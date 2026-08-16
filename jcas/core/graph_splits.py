"""Deterministic leakage-resistant split assignment for AV2 graph manifests.

The split unit is a connected component of candidate-recording/content
identifiers.  A whole component is assigned by a stable hash, so assignment is
independent of row order, multiprocessing completion order, and partial
smoke-test builds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "val", "test")
SPLIT_STRATEGY = "recording_or_stride1_av_overlap_content_group_sha256_v4"
SPLIT_METADATA_CONTRACT_VERSION = 4
RECORDING_GROUP_COLUMN = "recording_group_id_v2"
RECORDING_GROUP_SOURCE_COLUMN = "recording_group_source_v2"
GROUP_COLUMNS = (
    RECORDING_GROUP_COLUMN,
    "content_sha1",
    "focal_trajectory_hash_0p1m",
)
TIME_COLUMNS = ("start_timestamp", "end_timestamp")
OVERLAP_FINGERPRINT_COLUMN = "av_overlap_segment_hashes_stride1_v2"
REQUIRED_METADATA_COLUMNS = (
    "scenario_id",
    "city",
    *GROUP_COLUMNS,
    RECORDING_GROUP_SOURCE_COLUMN,
    *TIME_COLUMNS,
    OVERLAP_FINGERPRINT_COLUMN,
)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        root = int(value)
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[value]) != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_group_metadata_contract(path: str | Path) -> dict[str, Any]:
    """Validate the sidecar emitted by build_split_metadata.py."""
    metadata_path = Path(path)
    sidecar_path = metadata_path.with_suffix(
        metadata_path.suffix + ".metadata.json"
    )
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            "group metadata contract is missing: "
            f"{sidecar_path}; regenerate it with build_split_metadata.py"
        )
    with sidecar_path.open(encoding="utf-8") as stream:
        contract = json.load(stream)
    if contract.get("formal_split_eligible") is not True:
        raise ValueError(
            "debug/limited group metadata cannot define a formal split"
        )
    if int(contract.get("metadata_contract_version", -1)) != (
        SPLIT_METADATA_CONTRACT_VERSION
    ):
        raise ValueError("group metadata does not use the v4 contract")
    if contract.get("split_strategy") != SPLIT_STRATEGY:
        raise ValueError("group metadata does not use the v4 split strategy")
    fingerprint = contract.get("fingerprint_definition")
    if (
        not isinstance(fingerprint, dict)
        or int(fingerprint.get("segment_frames", -1)) != 3
        or int(fingerprint.get("stride_frames", -1)) != 1
        or fingerprint.get("absolute_time_grid") is not True
        or fingerprint.get("coverage")
        != "every_consecutive_three_frame_segment"
    ):
        raise ValueError("formal v4 metadata requires stride-1 AV fingerprints")
    output_record = contract.get("output")
    expected = (
        str(output_record.get("sha256", "")).lower()
        if isinstance(output_record, dict)
        else ""
    )
    actual = _sha256_file(metadata_path)
    if expected != actual:
        raise RuntimeError(
            "group metadata SHA-256 does not match its metadata contract"
        )
    return {
        "path": str(sidecar_path),
        "sha256": _sha256_file(sidecar_path),
        "formal_split_eligible": True,
        "metadata_contract_version": SPLIT_METADATA_CONTRACT_VERSION,
        "split_strategy": SPLIT_STRATEGY,
    }


def read_group_metadata(path: str | Path) -> pd.DataFrame:
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"split group metadata does not exist: {metadata_path}")
    if metadata_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(metadata_path)
    else:
        frame = pd.read_csv(metadata_path, dtype={"scenario_id": str})
    return validate_group_metadata(frame)


def validate_group_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_METADATA_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"split group metadata is missing columns: {missing}")

    result = frame.loc[:, REQUIRED_METADATA_COLUMNS].copy()
    for column in (
        "scenario_id",
        "city",
        *GROUP_COLUMNS,
        RECORDING_GROUP_SOURCE_COLUMN,
        OVERLAP_FINGERPRINT_COLUMN,
    ):
        if result[column].isna().any():
            count = int(result[column].isna().sum())
            raise ValueError(
                f"split group metadata column {column!r} has {count} missing values"
            )
        result[column] = result[column].astype(str)
        if (result[column].str.len() == 0).any():
            raise ValueError(
                f"split group metadata column {column!r} has empty values"
            )
    for column in TIME_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any() or not np.isfinite(
            result[column].to_numpy(np.float64)
        ).all():
            raise ValueError(
                f"split group metadata column {column!r} must be finite numeric"
            )
    if bool((result["end_timestamp"] < result["start_timestamp"]).any()):
        raise ValueError("split group metadata contains a negative time interval")
    if result["scenario_id"].duplicated().any():
        duplicates = (
            result.loc[result["scenario_id"].duplicated(False), "scenario_id"]
            .head(5)
            .tolist()
        )
        raise ValueError(
            "split group metadata has duplicate scenario_id values, examples: "
            f"{duplicates}"
        )
    return result.sort_values("scenario_id").reset_index(drop=True)


def component_assignments(group_metadata: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic component ID for every metadata scenario."""
    metadata = validate_group_metadata(group_metadata)
    disjoint = _DisjointSet(len(metadata))
    cities = metadata["city"].tolist()

    for column in GROUP_COLUMNS:
        first_index: dict[tuple[str, str], int] = {}
        for index, value in enumerate(metadata[column].tolist()):
            key = (str(cities[index]), str(value))
            prior = first_index.setdefault(key, index)
            disjoint.union(index, prior)

    # Fixed 10-second buckets miss shifted 10.9-second windows. Joining every
    # same-city time overlap is too conservative because simultaneous,
    # unrelated recordings form giant chains. Instead, union windows that
    # share a short absolute-time/position fingerprint of the AV trajectory.
    first_fingerprint_index: dict[tuple[str, str], int] = {}
    for index, raw_tokens in enumerate(
        metadata[OVERLAP_FINGERPRINT_COLUMN].tolist()
    ):
        tokens = [token for token in str(raw_tokens).split("|") if token]
        if not tokens:
            raise ValueError(
                f"scenario {metadata.iloc[index]['scenario_id']} has no AV "
                "overlap fingerprints"
            )
        for token in tokens:
            key = (str(cities[index]), str(token))
            prior = first_fingerprint_index.setdefault(key, index)
            disjoint.union(index, prior)

    members: dict[int, list[int]] = {}
    for index in range(len(metadata)):
        members.setdefault(disjoint.find(index), []).append(index)

    component_id_by_index: dict[int, str] = {}
    recording_group_values = metadata[RECORDING_GROUP_COLUMN].tolist()
    for indices in members.values():
        # Candidate recording groups are the primary units.  Taking the
        # lexicographically smallest one gives a readable, order-independent
        # ID if a content/focal hash ever links adjacent recording groups.
        recording_groups = sorted(
            {recording_group_values[index] for index in indices}
        )
        component_id = str(recording_groups[0])
        for index in indices:
            component_id_by_index[index] = component_id

    result = metadata.copy()
    result["split_group_id"] = [
        component_id_by_index[index] for index in range(len(result))
    ]
    mixed_city = result.groupby("split_group_id", sort=False)["city"].nunique()
    if bool((mixed_city > 1).any()):
        bad = mixed_city[mixed_city > 1].index[:5].tolist()
        raise ValueError(f"split components cross city boundaries: {bad}")
    return result


def temporal_overlap_counts(frame: pd.DataFrame) -> tuple[int, int]:
    """Return all and cross-split same-city overlapping interval pairs."""
    if frame.empty:
        return 0, 0
    required = {"city", "split", *TIME_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"temporal overlap audit is missing columns: {missing}")
    overlap_pairs = 0
    cross_split_pairs = 0
    for _city, city_rows in frame.groupby("city", sort=True):
        ordered = city_rows.sort_values(
            ["start_timestamp", "end_timestamp", "scenario_id"]
        )
        active: list[tuple[float, str]] = []
        for row in ordered.itertuples(index=False):
            start = float(row.start_timestamp)
            active = [(end, split) for end, split in active if end >= start]
            overlap_pairs += len(active)
            cross_split_pairs += sum(
                str(split) != str(row.split) for _end, split in active
            )
            active.append((float(row.end_timestamp), str(row.split)))
    return int(overlap_pairs), int(cross_split_pairs)


def av_overlap_segment_hashes(
    frame: pd.DataFrame,
    *,
    segment_frames: int = 3,
    stride_frames: int = 1,
    position_quantization_m: float = 0.1,
) -> str:
    """Fingerprint short AV segments on a global 0.1-second time grid.

    Shifted windows from one recording share tokens, while simultaneous
    recordings at different positions do not. Segment starts are aligned to a
    global grid so the token set is independent of a scenario's local timestep
    origin.
    """
    if segment_frames < 2:
        raise ValueError("segment_frames must be at least 2")
    if stride_frames < 1:
        raise ValueError("stride_frames must be positive")
    if not np.isfinite(position_quantization_m) or position_quantization_m <= 0:
        raise ValueError("position_quantization_m must be finite and positive")
    required = {
        "track_id",
        "timestep",
        "position_x",
        "position_y",
        "start_timestamp",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"AV fingerprint input is missing columns: {missing}")
    av = frame[frame["track_id"].astype(str) == "AV"].copy()
    if av.empty:
        raise ValueError("scenario contains no AV trajectory")
    av["timestep"] = pd.to_numeric(av["timestep"], errors="coerce")
    av["position_x"] = pd.to_numeric(av["position_x"], errors="coerce")
    av["position_y"] = pd.to_numeric(av["position_y"], errors="coerce")
    av = av.sort_values("timestep").drop_duplicates("timestep", keep=False)
    values = av[["timestep", "position_x", "position_y"]].to_numpy(np.float64)
    if (
        values.shape[0] < segment_frames
        or not np.isfinite(values).all()
    ):
        raise ValueError("AV trajectory is too short or contains non-finite values")
    start_values = pd.to_numeric(
        frame["start_timestamp"], errors="coerce"
    ).dropna()
    if start_values.empty or not np.isfinite(start_values.to_numpy()).all():
        raise ValueError("scenario start_timestamp is missing or non-finite")
    start_bin = int(np.rint(float(start_values.iloc[0]) / 100_000_000.0))
    quantization = float(position_quantization_m)
    tokens: list[str] = []
    for offset in range(values.shape[0] - segment_frames + 1):
        segment = values[offset : offset + segment_frames]
        timesteps = segment[:, 0].astype(np.int64)
        if not np.array_equal(
            timesteps, np.arange(timesteps[0], timesteps[0] + segment_frames)
        ):
            continue
        global_start_bin = start_bin + int(timesteps[0])
        if global_start_bin % int(stride_frames) != 0:
            continue
        quantized_positions = np.rint(
            segment[:, 1:] / quantization
        ).astype("<i8")
        payload = np.concatenate(
            [
                np.asarray([global_start_bin], dtype="<i8"),
                quantized_positions.reshape(-1),
            ]
        )
        tokens.append(
            hashlib.blake2b(payload.tobytes(), digest_size=16).hexdigest()
        )
    if not tokens:
        raise ValueError("scenario AV trajectory produced no overlap fingerprints")
    return "|".join(sorted(set(tokens)))


def _hash_fraction(seed: int, city: str, group_id: str) -> float:
    payload = f"{int(seed)}\0{city}\0{group_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(1 << 64)


def _validate_ratios(
    train_ratio: float, val_ratio: float, test_ratio: float
) -> tuple[float, float, float]:
    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    if any(not np.isfinite(value) or value <= 0.0 for value in ratios):
        raise ValueError("train/val/test ratios must all be finite and positive")
    if not np.isclose(sum(ratios), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("train/val/test ratios must sum to 1")
    return ratios


def grouped_split_assignments(
    group_metadata: pd.DataFrame,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    """Assign whole connected components using stable hash intervals."""
    if int(seed) < 0:
        raise ValueError("split seed must be non-negative")
    train_ratio, val_ratio, test_ratio = _validate_ratios(
        train_ratio, val_ratio, test_ratio
    )
    components = component_assignments(group_metadata)
    group_city = components.groupby("split_group_id", sort=False)["city"].first()
    split_by_group: dict[str, str] = {}
    for group_id, city in group_city.items():
        fraction = _hash_fraction(int(seed), str(city), str(group_id))
        if fraction < train_ratio:
            split = "train"
        elif fraction < train_ratio + val_ratio:
            split = "val"
        else:
            split = "test"
        split_by_group[str(group_id)] = split
    components["split"] = components["split_group_id"].map(split_by_group)
    return components


def apply_grouped_splits(
    rows: Iterable[dict[str, Any]],
    group_metadata: pd.DataFrame,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply grouped splits to manifest rows and return an integrity audit."""
    result_rows = [dict(row) for row in rows]
    if not result_rows:
        return [], {
            "strategy": SPLIT_STRATEGY,
            "num_manifest_rows": 0,
            "num_built_rows": 0,
            "num_split_groups": 0,
            "leakage_violations": {},
        }

    manifest = pd.DataFrame(result_rows)
    if "scenario_id" not in manifest:
        raise ValueError("graph manifest rows have no scenario_id")
    manifest["scenario_id"] = manifest["scenario_id"].astype(str)
    if manifest["scenario_id"].duplicated().any():
        raise ValueError("graph manifest has duplicate scenario_id values")

    assignments = grouped_split_assignments(
        group_metadata,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    assignment_columns = ["scenario_id", "city", "split_group_id", "split"]
    merged = manifest.merge(
        assignments[assignment_columns],
        on="scenario_id",
        how="left",
        suffixes=("", "_group"),
        validate="one_to_one",
    )
    missing = merged["split_group_id"].isna()
    if bool(missing.any()):
        examples = merged.loc[missing, "scenario_id"].head(5).tolist()
        raise ValueError(
            f"{int(missing.sum())} manifest scenarios lack split metadata; "
            f"examples: {examples}"
        )
    if "city" in manifest:
        city_mismatch = (
            merged["city"].notna()
            & (merged["city"].astype(str) != merged["city_group"].astype(str))
        )
        if bool(city_mismatch.any()):
            examples = merged.loc[city_mismatch, "scenario_id"].head(5).tolist()
            raise ValueError(f"manifest/metadata city mismatch, examples: {examples}")
        merged = merged.drop(columns=["city_group"])
    else:
        merged = merged.rename(columns={"city_group": "city"})

    if "split_group" in merged.columns:
        # When the incoming manifest already has a split column, pandas keeps
        # the freshly assigned split under the suffixed name.
        merged["split"] = merged["split_group"]
        merged = merged.drop(columns=["split_group"])
    elif "split" not in merged.columns:
        raise ValueError("grouped split assignment produced no split column")
    built_mask = merged["status"].isin(["built", "exists"])
    if "graph_path" in merged:
        built_mask &= merged["graph_path"].notna()
    built = merged.loc[built_mask].copy()

    metadata_for_built = assignments[
        assignments["scenario_id"].isin(set(built["scenario_id"]))
    ].copy()
    leakage: dict[str, int] = {}
    for column in GROUP_COLUMNS:
        split_counts = metadata_for_built.groupby(
            ["city", column], sort=False
        )["split"].nunique()
        leakage[column] = int((split_counts > 1).sum())
    exploded_fingerprints = metadata_for_built[
        ["city", "split", OVERLAP_FINGERPRINT_COLUMN]
    ].copy()
    exploded_fingerprints[OVERLAP_FINGERPRINT_COLUMN] = exploded_fingerprints[
        OVERLAP_FINGERPRINT_COLUMN
    ].str.split("|")
    exploded_fingerprints = exploded_fingerprints.explode(
        OVERLAP_FINGERPRINT_COLUMN
    )
    fingerprint_split_counts = exploded_fingerprints.groupby(
        ["city", OVERLAP_FINGERPRINT_COLUMN], sort=False
    )["split"].nunique()
    leakage["full_window_stride1_av_fingerprint"] = int(
        (fingerprint_split_counts > 1).sum()
    )
    leakage["split_group_id"] = int(
        (
            metadata_for_built.groupby("split_group_id", sort=False)["split"].nunique()
            > 1
        ).sum()
    )
    overlap_pairs, cross_split_overlap_pairs = temporal_overlap_counts(
        metadata_for_built
    )
    if any(leakage.values()):
        raise RuntimeError(f"grouped split leakage invariant failed: {leakage}")

    split_counts = {
        name: int((built["split"] == name).sum()) for name in SPLIT_NAMES
    }
    group_sizes = built.groupby("split_group_id", sort=False).size()
    largest_groups = group_sizes.sort_values(ascending=False).head(10)
    city_split_counts = (
        built.groupby(["city", "split"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SPLIT_NAMES, fill_value=0)
    )
    audit = {
        "strategy": SPLIT_STRATEGY,
        "seed": int(seed),
        "ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
        "num_manifest_rows": int(len(merged)),
        "num_built_rows": int(len(built)),
        "num_split_groups": int(merged["split_group_id"].nunique()),
        "built_component_sizes": {
            "median": float(group_sizes.median()) if len(group_sizes) else 0.0,
            "p95": (
                float(group_sizes.quantile(0.95))
                if len(group_sizes)
                else 0.0
            ),
            "p99": (
                float(group_sizes.quantile(0.99))
                if len(group_sizes)
                else 0.0
            ),
            "max": int(group_sizes.max()) if len(group_sizes) else 0,
            "largest": {
                str(group_id): int(size)
                for group_id, size in largest_groups.items()
            },
        },
        "temporal_overlap_pairs": int(overlap_pairs),
        "cross_split_temporal_overlap_pairs_diagnostic": int(
            cross_split_overlap_pairs
        ),
        "full_window_overlap_audit": {
            "fingerprint_stride_frames": 1,
            "shared_fingerprint_cross_split_violations": int(
                leakage["full_window_stride1_av_fingerprint"]
            ),
            "all_same_city_temporal_overlap_pairs": int(overlap_pairs),
            "cross_split_same_city_temporal_overlap_pairs_diagnostic": int(
                cross_split_overlap_pairs
            ),
            "note": (
                "Time overlap alone is diagnostic because independent AV2 "
                "recordings can share city/time; stride-1 trajectory identity "
                "is the fail-closed leakage invariant."
            ),
        },
        "built_split_counts": split_counts,
        "built_city_split_counts": {
            str(city): {name: int(row[name]) for name in SPLIT_NAMES}
            for city, row in city_split_counts.iterrows()
        },
        "leakage_violations": leakage,
    }
    return merged.to_dict("records"), audit
