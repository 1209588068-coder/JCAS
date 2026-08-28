#!/usr/bin/env python3
"""Create the train-only slow-K10/moving-K4 trajectory schedule manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from jcas.core.motion_normalized_features import (
    MOTION_NORMALIZED_EDGE_FEATURE_MODE,
)
from jcas.core.motion_schedule_trigger import (
    MOTION_REGIME_K4_K10_SCHEDULE,
    SCHEDULE_WINDOWS,
    scheduled_window,
)
from jcas.core.poison import load_poison_manifest, sha256_file
from jcas.core.risk_labels import RiskLabelConfig
from jcas.workflows.poison_manifest import (
    load_graph_manifest,
    resolve_graph_manifest,
)
from jcas.workflows.trainer import validate_graph_manifest_contract


VERSION = "v6_3_motion_regime_k4_k10_schedule_manifest_v1"
BASE_MANIFEST = Path("record/v6_2/poison_rate005_motion_prioritized.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the validation-only v6.3 slow-K10/moving-K4 manifest"
        )
    )
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--graph-manifest", required=True)
    parser.add_argument("--base-manifest", default=str(BASE_MANIFEST))
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_scheduled_manifest(base: pd.DataFrame) -> pd.DataFrame:
    required = {
        "scenario_id",
        "motion_regime",
        "perturb_window",
        "ramp_style",
        "velocity_mode",
    }
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"base manifest is missing columns: {missing}")
    result = base.copy()
    result["trigger_schedule_id"] = MOTION_REGIME_K4_K10_SCHEDULE
    result["perturb_window"] = (
        result["motion_regime"].astype(str).map(scheduled_window).astype(int)
    )
    if result["scenario_id"].duplicated().any():
        raise ValueError("scheduled manifest contains duplicate scenarios")
    if not result["ramp_style"].astype(str).eq("minimum_jerk").all():
        raise ValueError("scheduled manifest changed the ramp style")
    if not result["velocity_mode"].astype(str).eq("residual").all():
        raise ValueError("scheduled manifest changed velocity handling")
    return result.sort_values("scenario_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if (output.exists() or metadata_path.exists()) and not args.force:
        raise FileExistsError("scheduled manifest output already exists")
    graph_dir = Path(args.graph_dir)
    graph_manifest_path = resolve_graph_manifest(
        graph_dir, args.graph_manifest
    )
    validate_graph_manifest_contract(graph_manifest_path)
    graph_manifest = load_graph_manifest(graph_dir, graph_manifest_path)
    if int(graph_manifest["split"].eq("test").sum()) <= 0:
        raise RuntimeError("graph manifest has no frozen test split")
    label_config = RiskLabelConfig(
        label_mode="dynamic_risk",
        risk_base_distance_m=5.0,
        risk_reaction_time_s=1.0,
        risk_safe_decel_mps2=4.0,
    )
    base_path = Path(args.base_manifest)
    base, base_sha = load_poison_manifest(
        base_path,
        label_config,
        expected_split="train",
        require_strict_label=True,
        require_metadata_binding=True,
        expected_graph_manifest_sha256=sha256_file(graph_manifest_path),
    )
    base_metadata_path = base_path.with_suffix(base_path.suffix + ".metadata.json")
    base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    if base_metadata.get("experiment") != (
        "v6_2_data_only_motion_prioritized_pair_v1"
    ):
        raise RuntimeError("unexpected v6.2 base manifest")
    if base_metadata.get("edge_feature_mode") != (
        MOTION_NORMALIZED_EDGE_FEATURE_MODE
    ):
        raise RuntimeError("v6.2 base manifest has the wrong edge representation")

    frame = build_scheduled_manifest(base)
    if len(frame) != len(base) or not frame["scenario_id"].equals(
        base.sort_values("scenario_id").reset_index(drop=True)["scenario_id"]
    ):
        raise RuntimeError("scheduled manifest changed the poison scenario set")
    counts = {
        regime: int(frame["motion_regime"].astype(str).eq(regime).sum())
        for regime in SCHEDULE_WINDOWS
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(frame, output)
    metadata = {
        "scope": "offline_authorized_av2_model_robustness",
        "experiment": VERSION,
        "status": "frozen_train_only_pilot_manifest",
        "development_only": True,
        "formal_test_eligible": False,
        "test_access_authorized": False,
        "training_manifest_eligible": True,
        "split": "train",
        "requested_poison_scenario_rate": 0.05,
        "poisoned_scenarios": int(len(frame)),
        "scenario_ids_preserved": True,
        "pair_targets_preserved": True,
        "selection_authority": "frozen_v6_2_train_only_manifest",
        "selection_objective": "retain_v6_2_pairs_change_only_temporal_window",
        "trigger_schedule_id": MOTION_REGIME_K4_K10_SCHEDULE,
        "trigger_schedule_windows": dict(SCHEDULE_WINDOWS),
        "trigger_schedule_counts": counts,
        "motion_speed_threshold_mps": 0.5,
        "edge_feature_mode": MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        "orientation_policy": base_metadata["orientation_policy"],
        "allocation_policy": base_metadata["allocation_policy"],
        "fixed_allocation_alpha": base_metadata["fixed_allocation_alpha"],
        "require_strict_label": True,
        "strict_crossfit_required": False,
        "label_unit": base_metadata["label_unit"],
        "label_config": base_metadata["label_config"],
        "label_config_hash": base_metadata["label_config_hash"],
        "ordinary_bce_unchanged": True,
        "graph_model_unchanged": True,
        "displacement_and_allocation_unchanged": True,
        "original_validation_used": False,
        "original_test_used": False,
        "model_outputs_used": False,
        "victim_queries": 0,
        "manifest_path": str(output),
        "manifest_sha256": sha256_file(output),
        "base_manifest_path": str(base_path),
        "base_manifest_sha256": base_sha,
        "base_metadata_path": str(base_metadata_path),
        "base_metadata_sha256": sha256_file(base_metadata_path),
        "graph_manifest_path": str(graph_manifest_path),
        "graph_manifest_sha256": sha256_file(graph_manifest_path),
    }
    _atomic_json(metadata, metadata_path)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
