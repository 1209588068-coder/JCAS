#!/usr/bin/env python3
"""Strict train-only cross-fitting manifests for the v4.2 gray-box method."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from jcas import PROJECT_ROOT
from jcas.core.poison import sha256_file
from jcas.core.shadow_folds import load_shadow_fold_manifest


STRICT_CROSSFIT_PROTOCOL = "strict_crossfit_inner_validation_v1"
STRICT_CROSSFIT_CONTRACT_VERSION = "v4.2_strict_crossfit_contract_v1"
V5_STRICT_CROSSFIT_CONTRACT_VERSION = "v5_strict_crossfit_contract_v1"
SUPPORTED_STRICT_CROSSFIT_CONTRACT_VERSIONS = {
    STRICT_CROSSFIT_CONTRACT_VERSION,
    V5_STRICT_CROSSFIT_CONTRACT_VERSION,
}
STRICT_CROSSFIT_RELEASE_ID = "v4_2_pretraining_strict_crossfit_20260811"
REPOSITORY_ROOT = PROJECT_ROOT
FIT_COLUMNS = (
    "scenario_id",
    "split_group_id",
    "city",
    "source_split",
    "shadow_fold",
    "split",
    "graph_sha256",
)
SCORE_COLUMNS = (
    "scenario_id",
    "split_group_id",
    "city",
    "source_split",
    "shadow_fold",
    "split",
    "graph_sha256",
)


def _stable_priority(seed: int, city: str, group_id: str) -> int:
    payload = f"{int(seed)}\0{city}\0{group_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _label_ratio(frame: pd.DataFrame) -> float | None:
    required = {"num_risk_positive_edges", "num_supervision_edges"}
    if not required.issubset(frame.columns):
        return None
    positive = pd.to_numeric(
        frame["num_risk_positive_edges"], errors="coerce"
    )
    supervised = pd.to_numeric(
        frame["num_supervision_edges"], errors="coerce"
    )
    if positive.isna().any() or supervised.isna().any():
        raise ValueError("strict cross-fit label audit contains non-numeric values")
    denominator = float(supervised.sum())
    return float(positive.sum() / denominator) if denominator else None


def _inner_validation_groups(
    complement: pd.DataFrame,
    *,
    seed: int,
    fraction: float,
) -> set[str]:
    selected: set[str] = set()
    for city, city_frame in complement.groupby("city", sort=True):
        groups = (
            city_frame.groupby("split_group_id", sort=True)
            .size()
            .rename("rows")
            .reset_index()
        )
        if len(groups) < 2:
            raise ValueError(
                f"city {city!r} has fewer than two complement groups; "
                "cannot create disjoint inner train/validation"
            )
        records = groups.to_dict("records")
        records.sort(
            key=lambda item: (
                _stable_priority(
                    int(seed), str(city), str(item["split_group_id"])
                ),
                str(item["split_group_id"]),
            )
        )
        target_rows = float(len(city_frame)) * float(fraction)
        running = 0
        city_selected: list[dict[str, Any]] = []
        for item in records:
            include_error = abs(running + int(item["rows"]) - target_rows)
            exclude_error = abs(running - target_rows)
            if include_error < exclude_error or not city_selected:
                city_selected.append(item)
                running += int(item["rows"])
        if len(city_selected) == len(records):
            city_selected.pop()
        if not city_selected:
            city_selected.append(records[0])
        selected.update(str(item["split_group_id"]) for item in city_selected)
    return selected


def build_strict_crossfit_manifests(
    graph_manifest: pd.DataFrame,
    outer_folds: pd.DataFrame,
    *,
    heldout_fold: int,
    num_folds: int = 3,
    inner_seed: int = 20260820,
    inner_validation_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create one fit manifest and one never-fit held-out score manifest."""
    if int(heldout_fold) < 0 or int(heldout_fold) >= int(num_folds):
        raise ValueError("heldout_fold is outside the outer fold range")
    if int(inner_seed) < 0:
        raise ValueError("inner_seed must be non-negative")
    if not np.isfinite(inner_validation_fraction) or not (
        0.0 < float(inner_validation_fraction) < 0.5
    ):
        raise ValueError("inner_validation_fraction must be within (0, 0.5)")
    required_graph = {
        "scenario_id",
        "split",
        "split_group_id",
        "city",
        "graph_sha256",
        "status",
        "graph_path",
    }
    missing = sorted(required_graph - set(graph_manifest.columns))
    if missing:
        raise ValueError(f"graph manifest is missing strict cross-fit fields: {missing}")
    required_folds = {
        "scenario_id",
        "split_group_id",
        "city",
        "source_split",
        "shadow_fold",
        "graph_sha256",
    }
    missing = sorted(required_folds - set(outer_folds.columns))
    if missing:
        raise ValueError(f"outer fold manifest is missing fields: {missing}")
    original_train = graph_manifest[
        graph_manifest["split"].eq("train")
        & graph_manifest["status"].isin(["built", "exists"])
        & graph_manifest["graph_path"].notna()
    ].copy()
    fold_columns = outer_folds[
        ["scenario_id", "split_group_id", "city", "shadow_fold", "graph_sha256"]
    ].rename(
        columns={
            "split_group_id": "outer_split_group_id",
            "city": "outer_city",
            "graph_sha256": "outer_graph_sha256",
        }
    )
    merged = original_train.merge(
        fold_columns, on="scenario_id", how="left", validate="one_to_one"
    )
    if len(merged) != len(outer_folds) or merged["shadow_fold"].isna().any():
        raise ValueError("outer folds do not cover exactly the built original train split")
    if not merged["split_group_id"].astype(str).eq(
        merged["outer_split_group_id"].astype(str)
    ).all():
        raise ValueError("outer fold split_group_id does not match graph manifest")
    if not merged["city"].astype(str).eq(merged["outer_city"].astype(str)).all():
        raise ValueError("outer fold city does not match graph manifest")
    if not merged["graph_sha256"].astype(str).str.lower().eq(
        merged["outer_graph_sha256"].astype(str).str.lower()
    ).all():
        raise ValueError("outer fold graph SHA-256 does not match graph manifest")
    merged = merged.drop(
        columns=["outer_split_group_id", "outer_city", "outer_graph_sha256"]
    )
    merged["shadow_fold"] = merged["shadow_fold"].astype(np.int64)
    merged["source_split"] = "train"

    score = merged[merged["shadow_fold"] == int(heldout_fold)].copy()
    complement = merged[merged["shadow_fold"] != int(heldout_fold)].copy()
    if score.empty or complement.empty:
        raise ValueError("strict cross-fit score or complement pool is empty")
    inner_val_groups = _inner_validation_groups(
        complement,
        seed=int(inner_seed) + int(heldout_fold),
        fraction=float(inner_validation_fraction),
    )
    complement["split"] = np.where(
        complement["split_group_id"].astype(str).isin(inner_val_groups),
        "val",
        "train",
    )
    score["split"] = "score"
    fit = complement[list(FIT_COLUMNS)].sort_values("scenario_id").reset_index(
        drop=True
    )
    score = score[list(SCORE_COLUMNS)].sort_values("scenario_id").reset_index(
        drop=True
    )

    fit_scenarios = set(fit["scenario_id"].astype(str))
    score_scenarios = set(score["scenario_id"].astype(str))
    train_groups = set(
        fit.loc[fit["split"] == "train", "split_group_id"].astype(str)
    )
    val_groups = set(
        fit.loc[fit["split"] == "val", "split_group_id"].astype(str)
    )
    score_groups = set(score["split_group_id"].astype(str))
    intersection_audit = {
        "heldout_scenario_inner_train": int(
            len(score_scenarios & set(fit.loc[fit["split"] == "train", "scenario_id"].astype(str)))
        ),
        "heldout_scenario_inner_val": int(
            len(score_scenarios & set(fit.loc[fit["split"] == "val", "scenario_id"].astype(str)))
        ),
        "heldout_group_fit": int(len(score_groups & (train_groups | val_groups))),
        "inner_train_group_inner_val": int(len(train_groups & val_groups)),
    }
    if any(intersection_audit.values()):
        raise RuntimeError("strict cross-fit manifests are not disjoint")
    if fit_scenarios | score_scenarios != set(outer_folds["scenario_id"].astype(str)):
        raise RuntimeError("strict cross-fit manifests do not cover the outer folds")
    if not {"train", "val"}.issubset(set(fit["split"])):
        raise RuntimeError("strict cross-fit fit manifest lacks train or inner validation")

    city_counts = {
        str(city): {
            str(split): int(len(rows))
            for split, rows in city_frame.groupby("split", sort=True)
        }
        for city, city_frame in pd.concat([fit, score]).groupby("city", sort=True)
    }
    graph_lookup = graph_manifest.set_index("scenario_id", drop=False)
    label_audit: dict[str, Any] = {}
    for split_name, scenarios in (
        ("inner_train", fit.loc[fit["split"] == "train", "scenario_id"]),
        ("inner_val", fit.loc[fit["split"] == "val", "scenario_id"]),
        ("heldout_score", score["scenario_id"]),
    ):
        label_audit[split_name] = _label_ratio(
            graph_lookup.loc[scenarios.astype(str).tolist()]
        )
    if label_audit["inner_train"] is not None and label_audit["inner_val"] is not None:
        label_audit["inner_val_minus_train"] = float(
            label_audit["inner_val"] - label_audit["inner_train"]
        )
    audit = {
        "version": STRICT_CROSSFIT_CONTRACT_VERSION,
        "surrogate_protocol": STRICT_CROSSFIT_PROTOCOL,
        "heldout_fold": int(heldout_fold),
        "num_folds": int(num_folds),
        "inner_split_seed": int(inner_seed) + int(heldout_fold),
        "inner_validation_fraction": float(inner_validation_fraction),
        "fit_rows": int(len(fit)),
        "inner_train_rows": int((fit["split"] == "train").sum()),
        "inner_val_rows": int((fit["split"] == "val").sum()),
        "heldout_score_rows": int(len(score)),
        "fit_groups": int(fit["split_group_id"].nunique()),
        "heldout_score_groups": int(score["split_group_id"].nunique()),
        "city_split_scenario_counts": city_counts,
        "risk_positive_edge_ratio_audit": label_audit,
        "intersections": intersection_audit,
        "original_validation_graphs_used": 0,
        "original_test_graphs_used": 0,
        "heldout_graphs_used_for_gradient_updates": 0,
        "heldout_graphs_used_for_checkpoint_selection": 0,
        "heldout_graphs_used_for_threshold_selection": 0,
        "heldout_graphs_used_for_hyperparameter_selection": 0,
    }
    return fit, score, audit


def _resolve_recorded_path(recorded: str, owner: Path) -> Path:
    requested = Path(str(recorded))
    for candidate in (requested, owner.parent / requested.name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot resolve strict cross-fit asset: {recorded}")


def _repository_asset(recorded: str) -> Path:
    requested = Path(str(recorded))
    resolved = (
        requested.resolve()
        if requested.is_absolute()
        else (REPOSITORY_ROOT / requested).resolve()
    )
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError("v4.2 pretraining asset escapes repository") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _archive_member_name(recorded: str) -> str:
    normalized = PurePosixPath(str(recorded).replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError("strict-crossfit archive asset path is unsafe")
    return normalized.as_posix().removeprefix("./")


def _verify_sha_manifest(
    path: Path,
    *,
    source_archive: Path | None = None,
) -> int:
    """Verify assets against the live tree or their frozen source archive.

    Code assets are historical method inputs.  A later experiment version is
    expected to change the live source tree, so an available frozen archive is
    the durable trust target.  Data/contract manifests remain verified against
    their live, immutable files.
    """
    entries = 0
    archive: tarfile.TarFile | None = None
    archive_members: dict[str, tarfile.TarInfo] = {}
    if source_archive is not None:
        archive = tarfile.open(source_archive, mode="r:gz")
        for member in archive.getmembers():
            name = _archive_member_name(member.name)
            if name in archive_members:
                archive.close()
                raise ValueError(
                    f"duplicate strict-crossfit source archive member: {name}"
                )
            archive_members[name] = member
    try:
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = line.split("  ", 1)
            if (
                len(parts) != 2
                or len(parts[0]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in parts[0]
                )
            ):
                raise ValueError(
                    f"invalid strict-crossfit SHA manifest line {number}: {path}"
                )
            if archive is None:
                asset = _repository_asset(parts[1])
                observed = sha256_file(asset)
            else:
                member_name = _archive_member_name(parts[1])
                member = archive_members.get(member_name)
                if member is None or not member.isfile():
                    raise FileNotFoundError(
                        f"strict-crossfit source archive lacks: {parts[1]}"
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(
                        f"cannot read strict-crossfit archive asset: {parts[1]}"
                    )
                with stream:
                    observed = _sha256_stream(stream)
            if observed != parts[0]:
                raise RuntimeError(
                    f"strict-crossfit frozen asset changed: {parts[1]}"
                )
            entries += 1
    finally:
        if archive is not None:
            archive.close()
    if entries == 0:
        raise ValueError("strict-crossfit SHA manifest is empty")
    return entries


def load_strict_crossfit_release(
    path: str | Path,
    *,
    expected_release_sha256: str,
    expected_graph_manifest_sha256: str,
    expected_contract_path: str | Path,
) -> dict[str, Any]:
    """Verify an external pretraining anchor and every strict-crossfit asset."""
    release_path = _repository_asset(str(path))
    expected_release = str(expected_release_sha256).lower()
    if (
        len(expected_release) != 64
        or any(
            character not in "0123456789abcdef" for character in expected_release
        )
        or sha256_file(release_path) != expected_release
    ):
        raise RuntimeError("strict-crossfit pretraining release trust anchor mismatch")
    with release_path.open(encoding="utf-8") as stream:
        release = json.load(stream)
    if not isinstance(release, dict):
        raise ValueError("strict-crossfit pretraining release root must be an object")
    release_id = str(release.get("release_id", ""))
    supported_statuses = {
        "pretraining_frozen_before_v4_2_surrogate_training",
        "pretraining_frozen_before_v5_surrogate_training",
    }
    if not release_id or release.get("status") not in supported_statuses:
        raise ValueError("strict-crossfit pretraining release declaration is unsupported")
    if int(release.get("training_results_present_before_freeze", -1)) != 0:
        raise RuntimeError("strict-crossfit release was not frozen before surrogate training")
    if int(release.get("training_checkpoints_present_before_freeze", -1)) != 0:
        raise RuntimeError("strict-crossfit release was frozen after a checkpoint existed")
    if release.get("surrogate_protocol") != STRICT_CROSSFIT_PROTOCOL:
        raise ValueError("strict-crossfit release uses another surrogate protocol")
    graph_record = release.get("graph_manifest")
    if not isinstance(graph_record, dict) or str(
        graph_record.get("sha256", "")
    ).lower() != str(expected_graph_manifest_sha256).lower():
        raise RuntimeError("strict-crossfit release uses another graph manifest")

    source_archive_audit: dict[str, str] | None = None
    source_path: Path | None = None
    source_record = release.get("source_archive")
    if source_record is not None:
        if not isinstance(source_record, dict):
            raise ValueError("strict-crossfit source archive record is malformed")
        source_path = _repository_asset(str(source_record.get("path", "")))
        source_sha = sha256_file(source_path)
        if source_sha != str(source_record.get("sha256", "")).lower():
            raise RuntimeError("strict-crossfit source archive SHA-256 mismatch")
        source_archive_audit = {
            "path": str(source_path),
            "sha256": source_sha,
        }

    verified_manifests: dict[str, dict[str, Any]] = {}
    for key in ("code_asset_manifest", "strict_crossfit_asset_manifest"):
        record = release.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"strict-crossfit release lacks {key}")
        manifest_path = _repository_asset(str(record.get("path", "")))
        if sha256_file(manifest_path) != str(record.get("sha256", "")).lower():
            raise RuntimeError(f"strict-crossfit {key} SHA-256 mismatch")
        entries = _verify_sha_manifest(
            manifest_path,
            source_archive=(source_path if key == "code_asset_manifest" else None),
        )
        if entries != int(record.get("entries", -1)):
            raise ValueError(f"strict-crossfit {key} entry count mismatch")
        verified_manifests[key] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "entries": entries,
        }

    base_poison_audit: dict[str, str | int] | None = None
    poison_record = release.get("base_poison_manifest")
    if release.get("status") == "pretraining_frozen_before_v5_surrogate_training":
        if not isinstance(poison_record, dict):
            raise ValueError("v5 strict release lacks its base poison manifest")
        poison_path = _repository_asset(str(poison_record.get("path", "")))
        poison_metadata_path = _repository_asset(
            str(poison_record.get("metadata_path", ""))
        )
        if sha256_file(poison_path) != str(poison_record.get("sha256", "")).lower():
            raise RuntimeError("v5 base poison manifest SHA-256 mismatch")
        if sha256_file(poison_metadata_path) != str(
            poison_record.get("metadata_sha256", "")
        ).lower():
            raise RuntimeError("v5 base poison metadata SHA-256 mismatch")
        with poison_path.open(encoding="utf-8") as stream:
            poison_rows = sum(1 for _line in stream) - 1
        if poison_rows != int(poison_record.get("rows", -1)):
            raise ValueError("v5 base poison manifest row count mismatch")
        base_poison_audit = {
            "path": str(poison_path),
            "sha256": sha256_file(poison_path),
            "metadata_path": str(poison_metadata_path),
            "metadata_sha256": sha256_file(poison_metadata_path),
            "rows": int(poison_rows),
        }

    contract_path = Path(expected_contract_path).resolve()
    contract_sha = sha256_file(contract_path)
    matching = [
        item
        for item in release.get("fold_contracts", [])
        if Path(str(item.get("contract_path", ""))).resolve() == contract_path
        and str(item.get("contract_sha256", "")).lower() == contract_sha
    ]
    if len(matching) != 1:
        raise RuntimeError("strict fold contract is absent from pretraining release")
    return {
        "path": str(release_path),
        "sha256": expected_release,
        "release_id": release_id,
        "contract_sha256": contract_sha,
        "code_asset_manifest": verified_manifests["code_asset_manifest"],
        "strict_crossfit_asset_manifest": verified_manifests[
            "strict_crossfit_asset_manifest"
        ],
        "source_archive": source_archive_audit,
        "base_poison_manifest": base_poison_audit,
    }


def load_strict_crossfit_contract(
    path: str | Path,
    *,
    expected_graph_manifest_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load and fail-closed validate one strict fold fit/score contract."""
    contract_path = Path(path)
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    with contract_path.open(encoding="utf-8") as stream:
        contract = json.load(stream)
    if not isinstance(contract, dict):
        raise ValueError("strict cross-fit contract root must be an object")
    if contract.get("version") not in SUPPORTED_STRICT_CROSSFIT_CONTRACT_VERSIONS:
        raise ValueError("strict cross-fit contract version is unsupported")
    if contract.get("surrogate_protocol") != STRICT_CROSSFIT_PROTOCOL:
        raise ValueError("strict cross-fit protocol is unsupported")
    graph_record = contract.get("graph_manifest", {})
    if str(graph_record.get("sha256", "")).lower() != str(
        expected_graph_manifest_sha256
    ).lower():
        raise RuntimeError("strict cross-fit contract uses another graph manifest")
    fit_record = contract.get("fit_manifest", {})
    score_record = contract.get("score_manifest", {})
    heldout = int(contract.get("heldout_fold", -1))
    num_folds = int(contract.get("num_folds", -1))
    if num_folds < 2 or heldout < 0 or heldout >= num_folds:
        raise ValueError("strict cross-fit heldout fold declaration is invalid")
    inner_seed = int(contract.get("inner_split_seed", -1))
    inner_fraction = float(contract.get("inner_validation_fraction", -1.0))
    if inner_seed < 0 or not np.isfinite(inner_fraction) or not (
        0.0 < inner_fraction < 0.5
    ):
        raise ValueError("strict cross-fit inner split declaration is invalid")
    for key in (
        "original_validation_graphs_used",
        "original_test_graphs_used",
        "heldout_graphs_used_for_gradient_updates",
        "heldout_graphs_used_for_checkpoint_selection",
        "heldout_graphs_used_for_threshold_selection",
        "heldout_graphs_used_for_hyperparameter_selection",
    ):
        if int(contract.get(key, -1)) != 0:
            raise ValueError(f"strict cross-fit contract violates {key}")
    recorded_intersections = contract.get("intersections")
    if not isinstance(recorded_intersections, dict):
        raise ValueError("strict cross-fit contract lacks intersection audit")
    expected_intersection_keys = {
        "heldout_scenario_inner_train",
        "heldout_scenario_inner_val",
        "heldout_group_fit",
        "inner_train_group_inner_val",
    }
    if set(recorded_intersections) != expected_intersection_keys or any(
        int(recorded_intersections[key]) != 0
        for key in expected_intersection_keys
    ):
        raise ValueError("strict cross-fit recorded intersections are nonzero")
    fit_path = _resolve_recorded_path(str(fit_record.get("path", "")), contract_path)
    score_path = _resolve_recorded_path(str(score_record.get("path", "")), contract_path)
    if sha256_file(fit_path) != str(fit_record.get("sha256", "")).lower():
        raise RuntimeError("strict fit manifest SHA-256 mismatch")
    if sha256_file(score_path) != str(score_record.get("sha256", "")).lower():
        raise RuntimeError("strict score manifest SHA-256 mismatch")
    fit = pd.read_csv(
        fit_path,
        dtype={"scenario_id": str, "split_group_id": str, "graph_sha256": str},
    )
    score = pd.read_csv(
        score_path,
        dtype={"scenario_id": str, "split_group_id": str, "graph_sha256": str},
    )
    for frame, columns, label in (
        (fit, FIT_COLUMNS, "fit"),
        (score, SCORE_COLUMNS, "score"),
    ):
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"strict {label} manifest is missing columns: {missing}")
        if frame.empty or frame["scenario_id"].duplicated().any():
            raise ValueError(f"strict {label} manifest is empty or duplicated")
        if not frame["source_split"].eq("train").all():
            raise ValueError(f"strict {label} manifest contains a forbidden split")
        if not frame["graph_sha256"].str.lower().str.fullmatch(
            r"[0-9a-f]{64}", na=False
        ).all():
            raise ValueError(f"strict {label} manifest contains invalid graph hashes")
    if set(fit["split"]) != {"train", "val"}:
        raise ValueError("strict fit manifest must contain inner train and val")
    if not score["split"].eq("score").all():
        raise ValueError("strict score manifest must contain only score rows")
    if not score["shadow_fold"].astype(int).eq(heldout).all():
        raise ValueError("strict score manifest does not match heldout fold")
    if fit["shadow_fold"].astype(int).eq(heldout).any():
        raise ValueError("heldout fold appears in strict fit manifest")
    train_groups = set(
        fit.loc[fit["split"] == "train", "split_group_id"].astype(str)
    )
    val_groups = set(fit.loc[fit["split"] == "val", "split_group_id"].astype(str))
    score_groups = set(score["split_group_id"].astype(str))
    if train_groups & val_groups or score_groups & (train_groups | val_groups):
        raise ValueError("strict cross-fit group intersections are nonzero")
    if set(fit["scenario_id"].astype(str)) & set(score["scenario_id"].astype(str)):
        raise ValueError("strict fit and score scenarios overlap")
    if int(fit_record.get("rows", -1)) != len(fit) or int(
        score_record.get("rows", -1)
    ) != len(score):
        raise ValueError("strict cross-fit row count mismatch")
    recorded_counts = {
        "fit_rows": len(fit),
        "inner_train_rows": int(fit["split"].eq("train").sum()),
        "inner_val_rows": int(fit["split"].eq("val").sum()),
        "heldout_score_rows": len(score),
    }
    for key, expected in recorded_counts.items():
        if int(contract.get(key, -1)) != int(expected):
            raise ValueError(f"strict cross-fit contract {key} mismatch")
    if int(contract.get("fit_groups", -1)) != int(
        fit["split_group_id"].nunique()
    ) or int(contract.get("heldout_score_groups", -1)) != int(
        score["split_group_id"].nunique()
    ):
        raise ValueError("strict cross-fit contract group count mismatch")

    outer_record = contract.get("outer_fold_manifest")
    if not isinstance(outer_record, dict):
        raise ValueError("strict cross-fit contract lacks its outer fold binding")
    outer_path = _resolve_recorded_path(
        str(outer_record.get("path", "")), contract_path
    )
    outer, outer_audit = load_shadow_fold_manifest(
        outer_path,
        expected_graph_manifest_sha256=expected_graph_manifest_sha256,
        expected_num_folds=num_folds,
    )
    if str(outer_record.get("sha256", "")).lower() != str(
        outer_audit["sha256"]
    ).lower():
        raise RuntimeError("strict outer fold manifest SHA-256 mismatch")
    if str(outer_record.get("metadata_sha256", "")).lower() != str(
        outer_audit["metadata_sha256"]
    ).lower():
        raise RuntimeError("strict outer fold metadata SHA-256 mismatch")
    combined = pd.concat([fit, score], ignore_index=True).sort_values(
        "scenario_id"
    ).reset_index(drop=True)
    expected_outer = outer.sort_values("scenario_id").reset_index(drop=True)
    if combined["scenario_id"].astype(str).tolist() != expected_outer[
        "scenario_id"
    ].astype(str).tolist():
        raise ValueError("strict fit/score union differs from the outer folds")
    for column in (
        "split_group_id",
        "city",
        "source_split",
        "shadow_fold",
        "graph_sha256",
    ):
        actual = combined[column].astype(str).str.lower()
        expected = expected_outer[column].astype(str).str.lower()
        if not actual.eq(expected).all():
            raise ValueError(
                f"strict fit/score {column} differs from the outer folds"
            )
    expected_score_ids = set(
        expected_outer.loc[
            expected_outer["shadow_fold"].astype(int) == heldout,
            "scenario_id",
        ].astype(str)
    )
    if set(score["scenario_id"].astype(str)) != expected_score_ids:
        raise ValueError("strict score rows are not exactly the held-out fold")
    audit = {
        **contract,
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "fit_manifest_path": str(fit_path),
        "fit_manifest_sha256": sha256_file(fit_path),
        "score_manifest_path": str(score_path),
        "score_manifest_sha256": sha256_file(score_path),
    }
    return (
        fit.sort_values("scenario_id").reset_index(drop=True),
        score.sort_values("scenario_id").reset_index(drop=True),
        audit,
    )


def effective_strict_fit_manifest(
    graph_manifest: pd.DataFrame,
    fit_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Return only complement inner-train/inner-val rows for model fitting."""
    original_train = graph_manifest[graph_manifest["split"] == "train"].copy()
    binding = fit_manifest[
        ["scenario_id", "split_group_id", "graph_sha256", "shadow_fold", "split"]
    ].rename(
        columns={
            "split_group_id": "strict_split_group_id",
            "graph_sha256": "strict_graph_sha256",
            "split": "strict_split",
        }
    )
    merged = original_train.merge(
        binding, on="scenario_id", how="inner", validate="one_to_one"
    )
    if len(merged) != len(fit_manifest):
        raise ValueError("strict fit manifest is not a subset of original train")
    if not merged["split_group_id"].astype(str).eq(
        merged["strict_split_group_id"].astype(str)
    ).all():
        raise ValueError("strict fit split_group_id does not match graph manifest")
    if not merged["graph_sha256"].astype(str).str.lower().eq(
        merged["strict_graph_sha256"].astype(str).str.lower()
    ).all():
        raise ValueError("strict fit graph SHA-256 does not match graph manifest")
    merged["original_split"] = "train"
    merged["split"] = merged["strict_split"]
    merged = merged.drop(
        columns=["strict_split_group_id", "strict_graph_sha256", "strict_split"]
    )
    return merged.sort_values("scenario_id").reset_index(drop=True)
