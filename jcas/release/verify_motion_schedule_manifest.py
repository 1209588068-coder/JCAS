#!/usr/bin/env python3
"""Independently verify every row of the v6.3 train-only schedule manifest."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from jcas.core.motion_normalized_features import (
    MOTION_NORMALIZED_EDGE_FEATURE_MODE,
)
from jcas.core.motion_schedule_trigger import (
    MOTION_REGIME_K4_K10_SCHEDULE,
    motion_regime_from_speed,
    scheduled_window,
)
from jcas.core.poison import apply_manifest_row, load_poison_manifest, sha256_file
from jcas.core.risk_labels import RiskLabelConfig
from jcas.workflows.poison_manifest import (
    load_graph_manifest,
    resolve_graph_manifest,
    resolve_verified_graph_path,
)
from jcas.workflows.trainer import validate_graph_manifest_contract


VERSION = "v6_3_motion_regime_schedule_verification_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the complete K10/K4 train-only schedule manifest"
    )
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--graph-manifest", required=True)
    parser.add_argument("--poison-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def verify_manifest(
    graph_dir: Path,
    graph_manifest: Any,
    poison_rows: Any,
    label_config: RiskLabelConfig,
) -> dict[str, Any]:
    train_rows = {
        str(row.scenario_id): row
        for row in graph_manifest[graph_manifest["split"].eq("train")].itertuples(
            index=False
        )
    }
    counts: dict[str, int] = defaultdict(int)
    maxima: dict[str, float] = defaultdict(float)
    for row in poison_rows.itertuples(index=False):
        scenario_id = str(row.scenario_id)
        graph_row = train_rows.get(scenario_id)
        if graph_row is None:
            raise RuntimeError("scheduled scenario is absent from the train split")
        path = resolve_verified_graph_path(graph_row, graph_dir)
        with np.load(path, allow_pickle=True) as graph:
            velocities = np.asarray(
                graph["observed_velocities_filled"], dtype=np.float64
            )
            endpoint_speeds = np.linalg.norm(
                velocities[[int(row.src), int(row.dst)], -1], axis=1
            )
            regime = motion_regime_from_speed(float(endpoint_speeds.min()))
            if regime != str(row.motion_regime):
                raise RuntimeError("stored motion regime differs from graph data")
            if int(row.perturb_window) != scheduled_window(regime):
                raise RuntimeError("stored perturb window differs from schedule")
            x_node, edge_attr, labels, target_mask, audit = apply_manifest_row(
                graph,
                row,
                label_config,
                require_strict_label=True,
                edge_feature_mode=MOTION_NORMALIZED_EDGE_FEATURE_MODE,
                experimental_trigger_schedule=(
                    MOTION_REGIME_K4_K10_SCHEDULE
                ),
            )
            if x_node.shape != graph["x_node"].shape:
                raise RuntimeError("scheduled node feature shape changed")
            if edge_attr.shape[1] != 42:
                raise RuntimeError("scheduled edge feature dimension is not 42")
            if int(target_mask.sum()) != 2:
                raise RuntimeError("scheduled manifest did not label both directions")
            if int(np.sum(labels[target_mask] == 1)) != 2:
                raise RuntimeError("scheduled target labels were not applied")
            if audit.get("trigger_schedule_id") != (
                MOTION_REGIME_K4_K10_SCHEDULE
            ):
                raise RuntimeError("scheduled transform audit ID changed")
            if int(audit["trigger_spec"]["perturb_window"]) != int(
                row.perturb_window
            ):
                raise RuntimeError("scheduled transform used the wrong window")
            expected_applied = min(
                0.2, 0.9 * float(audit["current_distance_m"])
            )
            if not np.isclose(
                float(audit["applied_displacement_m"]),
                expected_applied,
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError("scheduled planned displacement changed")
            if not np.isclose(
                float(audit["applied_relative_displacement_m"]),
                expected_applied,
                atol=1e-3,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "scheduled relative displacement changed: "
                    f"scenario={scenario_id}, expected={expected_applied}, "
                    "observed="
                    f"{audit['applied_relative_displacement_m']}, "
                    f"distance={audit['current_distance_m']}"
                )
            counts[regime] += 1
            for node in audit["nodes"]:
                for key in (
                    "max_induced_speed_mps",
                    "max_induced_acc_mps2",
                    "max_induced_jerk_mps3",
                    "terminal_displacement_m",
                ):
                    maxima[key] = max(maxima[key], float(node[key]))
    return {
        "verified_rows": int(len(poison_rows)),
        "schedule_counts": dict(sorted(counts.items())),
        "observed_maxima": dict(sorted(maxima.items())),
        "graph_files_verified": int(len(poison_rows)),
        "row_schedule_recomputed": True,
        "motion_regime_recomputed_from_graph": True,
        "transform_reapplied": True,
        "validation_graphs_opened": 0,
        "test_graphs_opened": 0,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError("verification output already exists")
    graph_dir = Path(args.graph_dir)
    graph_manifest_path = resolve_graph_manifest(
        graph_dir, args.graph_manifest
    )
    validate_graph_manifest_contract(graph_manifest_path)
    graph_manifest = load_graph_manifest(graph_dir, graph_manifest_path)
    label_config = RiskLabelConfig(
        label_mode="dynamic_risk",
        risk_base_distance_m=5.0,
        risk_reaction_time_s=1.0,
        risk_safe_decel_mps2=4.0,
    )
    poison_path = Path(args.poison_manifest)
    poison_rows, poison_sha = load_poison_manifest(
        poison_path,
        label_config,
        expected_split="train",
        require_strict_label=True,
        require_metadata_binding=True,
        expected_graph_manifest_sha256=sha256_file(graph_manifest_path),
        expected_trigger_schedule=MOTION_REGIME_K4_K10_SCHEDULE,
    )
    audit = verify_manifest(
        graph_dir, graph_manifest, poison_rows, label_config
    )
    payload = {
        "scope": "offline_authorized_av2_model_robustness",
        "version": VERSION,
        "status": "ok",
        "development_only": True,
        "formal_test_eligible": False,
        "trigger_schedule_id": MOTION_REGIME_K4_K10_SCHEDULE,
        "poison_manifest": {
            "path": str(poison_path),
            "sha256": poison_sha,
        },
        "graph_manifest": {
            "path": str(graph_manifest_path),
            "sha256": sha256_file(graph_manifest_path),
        },
        **audit,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(payload, output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
