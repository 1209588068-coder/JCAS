#!/usr/bin/env python3
"""Train-only group folds for the v4 gray-box shadow-model experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jcas.core.poison import sha256_file


SHADOW_FOLD_VERSION = "v4_train_group_folds_v1"
SHADOW_FOLD_COLUMNS = (
    "scenario_id",
    "split_group_id",
    "city",
    "source_split",
    "shadow_fold",
    "graph_sha256",
)


def _stable_priority(seed: int, group_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def build_shadow_fold_frame(
    graph_manifest: pd.DataFrame,
    *,
    num_folds: int = 3,
    seed: int = 20260810,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign original-train split groups to balanced deterministic folds."""
    if int(num_folds) < 2:
        raise ValueError("num_folds must be at least two")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    required = {
        "scenario_id",
        "split",
        "split_group_id",
        "city",
        "graph_sha256",
        "status",
        "graph_path",
    }
    missing = sorted(required - set(graph_manifest.columns))
    if missing:
        raise ValueError(f"graph manifest is missing shadow-fold fields: {missing}")
    built = graph_manifest[
        graph_manifest["status"].isin(["built", "exists"])
        & graph_manifest["graph_path"].notna()
    ].copy()
    train = built[built["split"] == "train"].copy()
    if train.empty:
        raise ValueError("graph manifest has no built original-train rows")
    if train["scenario_id"].duplicated().any():
        raise ValueError("original-train scenario IDs must be unique")
    if train["split_group_id"].isna().any():
        raise ValueError("original-train rows contain missing split_group_id")

    groups = (
        train.groupby(["city", "split_group_id"], sort=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    assignment: dict[str, int] = {}
    fold_totals = np.zeros(int(num_folds), dtype=np.int64)
    city_fold_counts: dict[str, list[int]] = {}
    for city, city_groups in groups.groupby("city", sort=True):
        counts = np.zeros(int(num_folds), dtype=np.int64)
        records = city_groups.to_dict("records")
        records.sort(
            key=lambda item: (
                -int(item["rows"]),
                _stable_priority(int(seed), str(item["split_group_id"])),
                str(item["split_group_id"]),
            )
        )
        for item in records:
            minimum_city = int(counts.min())
            candidates = np.flatnonzero(counts == minimum_city)
            if candidates.size > 1:
                candidate_totals = fold_totals[candidates]
                candidates = candidates[candidate_totals == candidate_totals.min()]
            if candidates.size > 1:
                offset = _stable_priority(
                    int(seed) + 1, str(item["split_group_id"])
                ) % int(candidates.size)
                fold = int(candidates[int(offset)])
            else:
                fold = int(candidates[0])
            size = int(item["rows"])
            assignment[str(item["split_group_id"])] = fold
            counts[fold] += size
            fold_totals[fold] += size
        city_fold_counts[str(city)] = [int(value) for value in counts]

    train["source_split"] = "train"
    train["shadow_fold"] = train["split_group_id"].astype(str).map(assignment)
    if train["shadow_fold"].isna().any():
        raise RuntimeError("not every original-train group received a shadow fold")
    train["shadow_fold"] = train["shadow_fold"].astype(np.int64)
    frame = train[list(SHADOW_FOLD_COLUMNS)].sort_values("scenario_id").reset_index(
        drop=True
    )
    group_cross_fold = int(
        frame.groupby("split_group_id")["shadow_fold"].nunique().gt(1).sum()
    )
    scenario_counts = {
        str(fold): int((frame["shadow_fold"] == fold).sum())
        for fold in range(int(num_folds))
    }
    group_counts = {
        str(fold): int(
            frame.loc[frame["shadow_fold"] == fold, "split_group_id"].nunique()
        )
        for fold in range(int(num_folds))
    }
    audit = {
        "version": SHADOW_FOLD_VERSION,
        "source_split": "train",
        "num_folds": int(num_folds),
        "seed": int(seed),
        "rows": int(len(frame)),
        "split_groups": int(frame["split_group_id"].nunique()),
        "scenario_counts": scenario_counts,
        "group_counts": group_counts,
        "city_fold_scenario_counts": city_fold_counts,
        "group_cross_fold_violations": group_cross_fold,
        "non_train_source_rows": int((frame["source_split"] != "train").sum()),
    }
    if group_cross_fold != 0 or audit["non_train_source_rows"] != 0:
        raise RuntimeError("shadow fold construction violated train-only grouping")
    return frame, audit


def load_shadow_fold_manifest(
    path: str | Path,
    *,
    expected_graph_manifest_sha256: str,
    expected_num_folds: int | None = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a frozen fold assignment and validate its binding sidecar."""
    manifest_path = Path(path)
    metadata_path = manifest_path.with_suffix(manifest_path.suffix + ".metadata.json")
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("shadow fold manifest or metadata sidecar is missing")
    frame = pd.read_csv(
        manifest_path,
        dtype={"scenario_id": str, "split_group_id": str, "graph_sha256": str},
    )
    missing = sorted(set(SHADOW_FOLD_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"shadow fold manifest is missing columns: {missing}")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("shadow fold metadata must be an object")
    if str(metadata.get("version")) != SHADOW_FOLD_VERSION:
        raise ValueError("shadow fold version is not supported")
    if str(metadata.get("manifest_sha256", "")).lower() != sha256_file(
        manifest_path
    ):
        raise RuntimeError("shadow fold manifest SHA-256 does not match metadata")
    if str(metadata.get("graph_manifest_sha256", "")).lower() != str(
        expected_graph_manifest_sha256
    ).lower():
        raise RuntimeError("shadow folds are bound to a different graph manifest")
    if expected_num_folds is not None and int(metadata.get("num_folds", -1)) != int(
        expected_num_folds
    ):
        raise ValueError("shadow fold count does not match the requested protocol")
    if frame.empty or frame["scenario_id"].duplicated().any():
        raise ValueError("shadow fold manifest is empty or has duplicate scenarios")
    if not bool(frame["source_split"].eq("train").all()):
        raise ValueError("shadow folds may contain only original-train scenarios")
    fold_values = pd.to_numeric(frame["shadow_fold"], errors="coerce")
    if fold_values.isna().any() or not np.equal(fold_values, np.floor(fold_values)).all():
        raise ValueError("shadow_fold must contain integers")
    frame["shadow_fold"] = fold_values.astype(np.int64)
    num_folds = int(metadata["num_folds"])
    if not bool(frame["shadow_fold"].between(0, num_folds - 1).all()):
        raise ValueError("shadow_fold values are outside the declared range")
    if int(frame.groupby("split_group_id")["shadow_fold"].nunique().gt(1).sum()) != 0:
        raise ValueError("one split_group_id appears in multiple shadow folds")
    hashes = frame["graph_sha256"].astype(str).str.lower()
    if not bool(hashes.str.fullmatch(r"[0-9a-f]{64}", na=False).all()):
        raise ValueError("shadow folds contain invalid graph SHA-256 values")
    frame["graph_sha256"] = hashes
    if int(metadata.get("rows", -1)) != len(frame):
        raise ValueError("shadow fold row count does not match metadata")
    return frame.sort_values("scenario_id").reset_index(drop=True), {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "version": str(metadata["version"]),
        "num_folds": num_folds,
        "seed": int(metadata["seed"]),
        "rows": int(len(frame)),
        "group_cross_fold_violations": 0,
        "source_split": "train",
    }


def effective_shadow_manifest(
    graph_manifest: pd.DataFrame,
    fold_frame: pd.DataFrame,
    *,
    heldout_fold: int,
    num_folds: int,
) -> pd.DataFrame:
    """Map original train to effective train/val; drop original val/test."""
    if int(heldout_fold) < 0 or int(heldout_fold) >= int(num_folds):
        raise ValueError("heldout_fold is outside the shadow fold range")
    original_train = graph_manifest[graph_manifest["split"] == "train"].copy()
    fold_columns = fold_frame[
        ["scenario_id", "split_group_id", "shadow_fold", "graph_sha256"]
    ].rename(
        columns={
            "split_group_id": "shadow_split_group_id",
            "graph_sha256": "shadow_graph_sha256",
        }
    )
    merged = original_train.merge(
        fold_columns, on="scenario_id", how="left", validate="one_to_one"
    )
    if merged["shadow_fold"].isna().any() or len(merged) != len(fold_frame):
        raise ValueError("shadow folds do not cover exactly the original train split")
    if not bool(
        merged["split_group_id"].astype(str).eq(
            merged["shadow_split_group_id"].astype(str)
        ).all()
    ):
        raise ValueError("shadow split_group_id does not match graph manifest")
    if not bool(
        merged["graph_sha256"].astype(str).str.lower().eq(
            merged["shadow_graph_sha256"].astype(str).str.lower()
        ).all()
    ):
        raise ValueError("shadow graph SHA-256 does not match graph manifest")
    merged["original_split"] = "train"
    merged["split"] = np.where(
        merged["shadow_fold"].astype(int) == int(heldout_fold), "val", "train"
    )
    merged = merged.drop(columns=["shadow_split_group_id", "shadow_graph_sha256"])
    return merged.sort_values("scenario_id").reset_index(drop=True)
