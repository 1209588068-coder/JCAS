"""Two-frame leakage-resistant split protocol for the v5 data generation.

The graph bytes remain the source-bound graph-v4 artifacts.  V5 changes only
the split unit: connected components additionally union scenarios sharing any
same-city, absolute-time, consecutive two-frame AV trajectory fingerprint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from jcas.core.graph_splits import (
    GROUP_COLUMNS,
    RECORDING_GROUP_COLUMN,
    RECORDING_GROUP_SOURCE_COLUMN,
    SPLIT_NAMES,
    TIME_COLUMNS,
    _hash_fraction,
    _sha256_file,
    _validate_ratios,
    temporal_overlap_counts,
)


V5_SPLIT_STRATEGY = (
    "recording_or_stride1_twoframe_av_overlap_content_group_sha256_v5"
)
V5_SPLIT_METADATA_CONTRACT_VERSION = 5
V5_OVERLAP_FINGERPRINT_COLUMN = "av_overlap_segment_hashes_two_frame_stride1_v3"
V5_COMPONENT_COLUMN = "two_frame_split_group_id_v5"
V5_REQUIRED_METADATA_COLUMNS = (
    "scenario_id",
    "city",
    *GROUP_COLUMNS,
    RECORDING_GROUP_SOURCE_COLUMN,
    *TIME_COLUMNS,
    V5_COMPONENT_COLUMN,
)


def validate_v5_group_metadata_contract(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    sidecar_path = metadata_path.with_suffix(
        metadata_path.suffix + ".metadata.json"
    )
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"v5 group metadata contract is missing: {sidecar_path}")
    with sidecar_path.open(encoding="utf-8") as stream:
        contract = json.load(stream)
    if contract.get("formal_split_eligible") is not True:
        raise ValueError("debug/limited v5 metadata cannot define a formal split")
    if int(contract.get("metadata_contract_version", -1)) != (
        V5_SPLIT_METADATA_CONTRACT_VERSION
    ):
        raise ValueError("group metadata does not use the v5 contract")
    if contract.get("split_strategy") != V5_SPLIT_STRATEGY:
        raise ValueError("group metadata does not use the v5 split strategy")
    fingerprint = contract.get("fingerprint_definition")
    if (
        not isinstance(fingerprint, dict)
        or int(fingerprint.get("segment_frames", -1)) != 2
        or int(fingerprint.get("stride_frames", -1)) != 1
        or fingerprint.get("absolute_time_grid") is not True
        or fingerprint.get("coverage")
        != "every_consecutive_two_frame_segment"
        or float(fingerprint.get("position_quantization_m", -1.0)) != 0.1
    ):
        raise ValueError("formal v5 metadata requires the frozen two-frame rule")
    source_audit = contract.get("source_verification")
    output_record = contract.get("output")
    rows = int(output_record.get("rows", -1)) if isinstance(output_record, dict) else -1
    if (
        not isinstance(source_audit, dict)
        or source_audit.get("parquet_sha1_verified") is not True
        or int(source_audit.get("verified_scenarios", -1)) != rows
        or int(source_audit.get("verification_errors", -1)) != 0
    ):
        raise ValueError("v5 metadata lacks complete source parquet verification")
    expected = (
        str(output_record.get("sha256", "")).lower()
        if isinstance(output_record, dict)
        else ""
    )
    if expected != _sha256_file(metadata_path):
        raise RuntimeError("v5 group metadata SHA-256 does not match its contract")
    return {
        "path": str(sidecar_path),
        "sha256": _sha256_file(sidecar_path),
        "formal_split_eligible": True,
        "metadata_contract_version": V5_SPLIT_METADATA_CONTRACT_VERSION,
        "split_strategy": V5_SPLIT_STRATEGY,
        "component_column": V5_COMPONENT_COLUMN,
    }


def validate_v5_group_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(V5_REQUIRED_METADATA_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"v5 split group metadata is missing columns: {missing}")
    result = frame.loc[:, V5_REQUIRED_METADATA_COLUMNS].copy()
    text_columns = (
        "scenario_id",
        "city",
        *GROUP_COLUMNS,
        RECORDING_GROUP_SOURCE_COLUMN,
        V5_COMPONENT_COLUMN,
    )
    for column in text_columns:
        if result[column].isna().any():
            raise ValueError(f"v5 metadata column {column!r} contains missing values")
        result[column] = result[column].astype(str)
        if result[column].str.len().eq(0).any():
            raise ValueError(f"v5 metadata column {column!r} contains empty values")
    for column in TIME_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"v5 metadata column {column!r} must be numeric")
    if result["scenario_id"].duplicated().any():
        raise ValueError("v5 group metadata contains duplicate scenario IDs")
    if (result["end_timestamp"] < result["start_timestamp"]).any():
        raise ValueError("v5 group metadata contains negative time intervals")
    return result.sort_values("scenario_id").reset_index(drop=True)


def read_v5_group_metadata(path: str | Path) -> pd.DataFrame:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    frame = (
        pd.read_parquet(metadata_path)
        if metadata_path.suffix.lower() == ".parquet"
        else pd.read_csv(metadata_path, dtype={"scenario_id": str})
    )
    return validate_v5_group_metadata(frame)


def v5_component_assignments(group_metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = validate_v5_group_metadata(group_metadata)
    explicit = metadata[
        metadata[RECORDING_GROUP_SOURCE_COLUMN].astype(str).str.startswith(
            "parquet_column:"
        )
    ]
    conflicts = (
        explicit.groupby(V5_COMPONENT_COLUMN)[RECORDING_GROUP_COLUMN]
        .nunique()
        .gt(1)
        .sum()
    )
    if int(conflicts):
        raise RuntimeError(
            "v5 two-frame components merge distinct explicit recording IDs"
        )
    result = metadata.copy()
    result["split_group_id"] = result[V5_COMPONENT_COLUMN].astype(str)
    if result.groupby("split_group_id")["city"].nunique().gt(1).any():
        raise RuntimeError("v5 split components cross city boundaries")
    return result


def v5_grouped_split_assignments(
    group_metadata: pd.DataFrame,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    train_ratio, val_ratio, test_ratio = _validate_ratios(
        train_ratio, val_ratio, test_ratio
    )
    components = v5_component_assignments(group_metadata)
    groups = components.groupby("split_group_id", sort=False)["city"].first()
    split_by_group: dict[str, str] = {}
    for group_id, city in groups.items():
        fraction = _hash_fraction(int(seed), str(city), str(group_id))
        split_by_group[str(group_id)] = (
            "train"
            if fraction < train_ratio
            else "val"
            if fraction < train_ratio + val_ratio
            else "test"
        )
    components["split"] = components["split_group_id"].map(split_by_group)
    return components


def apply_v5_grouped_splits(
    rows: Iterable[dict[str, Any]],
    group_metadata: pd.DataFrame,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = pd.DataFrame([dict(row) for row in rows])
    if manifest.empty or "scenario_id" not in manifest:
        raise ValueError("v5 graph manifest is empty or lacks scenario_id")
    manifest["scenario_id"] = manifest["scenario_id"].astype(str)
    if manifest["scenario_id"].duplicated().any():
        raise ValueError("v5 graph manifest contains duplicate scenario IDs")
    assignments = v5_grouped_split_assignments(
        group_metadata,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    merged = manifest.drop(columns=["split_group_id", "split"], errors="ignore").merge(
        assignments[["scenario_id", "city", "split_group_id", "split"]],
        on="scenario_id",
        how="left",
        suffixes=("", "_group"),
        validate="one_to_one",
    )
    if merged["split_group_id"].isna().any():
        raise ValueError("v5 group metadata does not cover every manifest scenario")
    if "city_group" in merged:
        mismatch = merged["city"].astype(str).ne(merged["city_group"].astype(str))
        if mismatch.any():
            raise ValueError("v5 graph manifest and group metadata city mismatch")
        merged = merged.drop(columns=["city_group"])
    built_mask = merged["status"].isin(["built", "exists"])
    if "graph_path" in merged:
        built_mask &= merged["graph_path"].notna()
    built = merged.loc[built_mask].copy()
    metadata_for_built = assignments[
        assignments["scenario_id"].isin(set(built["scenario_id"]))
    ].copy()

    leakage: dict[str, int] = {}
    for column in GROUP_COLUMNS:
        counts = metadata_for_built.groupby(["city", column])["split"].nunique()
        leakage[column] = int(counts.gt(1).sum())
    component_counts = metadata_for_built.groupby(V5_COMPONENT_COLUMN)[
        "split"
    ].nunique()
    leakage["two_frame_stride1_av_fingerprint"] = int(
        component_counts.gt(1).sum()
    )
    leakage["split_group_id"] = int(
        metadata_for_built.groupby("split_group_id")["split"].nunique().gt(1).sum()
    )
    if any(leakage.values()):
        raise RuntimeError(f"v5 split leakage invariant failed: {leakage}")

    overlap_pairs, cross_split_overlap_pairs = temporal_overlap_counts(
        metadata_for_built
    )
    group_sizes = built.groupby("split_group_id").size()
    largest = group_sizes.sort_values(ascending=False).head(10)
    city_counts = (
        built.groupby(["city", "split"]).size().unstack(fill_value=0)
        .reindex(columns=SPLIT_NAMES, fill_value=0)
    )
    audit = {
        "strategy": V5_SPLIT_STRATEGY,
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
            "median": float(group_sizes.median()),
            "p95": float(group_sizes.quantile(0.95)),
            "p99": float(group_sizes.quantile(0.99)),
            "max": int(group_sizes.max()),
            "largest": {str(key): int(value) for key, value in largest.items()},
        },
        "temporal_overlap_pairs": int(overlap_pairs),
        "cross_split_temporal_overlap_pairs_diagnostic": int(
            cross_split_overlap_pairs
        ),
        "two_frame_overlap_audit": {
            "segment_frames": 2,
            "stride_frames": 1,
            "position_quantization_m": 0.1,
            "shared_fingerprint_cross_split_violations": int(
                leakage["two_frame_stride1_av_fingerprint"]
            ),
        },
        "built_split_counts": {
            name: int(built["split"].eq(name).sum()) for name in SPLIT_NAMES
        },
        "built_city_split_counts": {
            str(city): {name: int(row[name]) for name in SPLIT_NAMES}
            for city, row in city_counts.iterrows()
        },
        "explicit_recording_id_component_conflicts": 0,
        "leakage_violations": leakage,
    }
    return merged.to_dict("records"), audit
