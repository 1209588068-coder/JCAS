#!/usr/bin/env python3
"""Build a motion-regime-prioritized 5% training manifest.

The generator preserves the exact 5,776 scenarios frozen by v6.0.  Within
each scenario it enumerates every data-eligible unordered negative pair and
prefers the requested motion regime before applying a stable hash.  The
default remains the historical v6.2-D moving-pair policy.  No model,
probability, gradient, validation row, or test row is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jcas.core.motion_normalized_features import (
    MOTION_NORMALIZED_EDGE_FEATURE_MODE,
)
from jcas.core.poison import (
    ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
    FIXED_SYMMETRIC_BIEND_ALPHA,
    ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
    PAIR_LABEL_UNIT,
    POISON_MANIFEST_COLUMNS,
    eligible_negative_pair_groups,
    fixed_symmetric_bi_endpoint_allocation,
    load_poison_manifest,
    select_pair_orientation,
    sha256_file,
)
from jcas.core.risk_labels import (
    RiskLabelConfig,
    label_config_dict,
    label_config_hash,
    labels_for_graph,
)
from jcas.workflows.poison_manifest import (
    load_graph_manifest,
    resolve_graph_manifest,
    resolve_verified_graph_path,
)
from jcas.workflows.trainer import validate_graph_manifest_contract


VERSION = "v6_2_data_only_motion_prioritized_pair_v1"
SLOW_PRIORITIZED_VERSION = "v6_4_data_only_slow_prioritized_pair_v1"
SELECTION_SEED = 20260621
POISON_RATE = 0.05
MOTION_SPEED_THRESHOLD_MPS = 0.5
BASE_MANIFEST = Path("record/v6/poison_rate005_fixed_symmetric.csv")
SLOW_MOTION_REGIME = "slow_lt_0p5"
MOVING_MOTION_REGIME = "moving_ge_0p5"
MOTION_REGIMES = (SLOW_MOTION_REGIME, MOVING_MOTION_REGIME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the train-only v6.2-D motion-prioritized manifest"
        )
    )
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--graph-manifest", required=True)
    parser.add_argument("--base-manifest", default=str(BASE_MANIFEST))
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument(
        "--preferred-regime",
        choices=MOTION_REGIMES,
        default=MOVING_MOTION_REGIME,
        help=(
            "Motion regime preferred within each frozen train scenario. "
            "The other regime is used only when no preferred pair exists."
        ),
    )
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


def motion_regime(min_endpoint_speed_mps: float) -> str:
    speed = float(min_endpoint_speed_mps)
    if not np.isfinite(speed) or speed < 0.0:
        raise ValueError("minimum endpoint speed must be finite and non-negative")
    return (
        MOVING_MOTION_REGIME
        if speed >= MOTION_SPEED_THRESHOLD_MPS
        else SLOW_MOTION_REGIME
    )


def pair_priority(
    seed: int,
    scenario_id: str,
    src: int,
    dst: int,
    regime: str,
) -> str:
    if int(seed) < 0:
        raise ValueError("selection seed must be non-negative")
    canonical = (min(int(src), int(dst)), max(int(src), int(dst)))
    payload = (
        f"v6.2-motion-prioritized-v1:{int(seed)}:{scenario_id}:"
        f"{regime}:{canonical[0]}:{canonical[1]}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def choose_motion_candidate(
    candidates: list[dict[str, Any]],
    *,
    preferred_regime: str = MOVING_MOTION_REGIME,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("a scenario must have at least one eligible pair")
    if str(preferred_regime) not in MOTION_REGIMES:
        raise ValueError(f"unsupported preferred motion regime: {preferred_regime!r}")
    preferred = [
        item
        for item in candidates
        if item["motion_regime"] == str(preferred_regime)
    ]
    pool = preferred if preferred else candidates
    return min(
        pool,
        key=lambda item: (
            str(item["selection_priority_sha256"]),
            min(int(item["src"]), int(item["dst"])),
            max(int(item["src"]), int(item["dst"])),
        ),
    )


def _scenario_candidates(
    graph: Any,
    pair_groups: list[np.ndarray],
    *,
    scenario_id: str,
    selection_seed: int,
) -> list[dict[str, Any]]:
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    velocities = np.asarray(
        graph["observed_velocities_filled"], dtype=np.float64
    )
    candidates: list[dict[str, Any]] = []
    for group in pair_groups:
        edge_id, orientation = select_pair_orientation(
            graph,
            np.asarray(group, dtype=np.int64),
            policy=ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
            perturb_window=10,
        )
        src = int(edge_index[0, edge_id])
        dst = int(edge_index[1, edge_id])
        endpoint_speeds = np.linalg.norm(
            velocities[[src, dst], -1], axis=1
        )
        minimum_speed = float(endpoint_speeds.min())
        regime = motion_regime(minimum_speed)
        candidates.append(
            {
                "scenario_id": str(scenario_id),
                "src": src,
                "dst": dst,
                "src_track_id": str(graph["node_track_ids"][src]),
                "dst_track_id": str(graph["node_track_ids"][dst]),
                "orientation_policy": (
                    ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED
                ),
                "orientation_destination_mean_speed_mps": float(
                    orientation["destination_mean_speed_mps"]
                ),
                "orientation_candidate_count": int(
                    orientation["candidate_orientations"]
                ),
                "src_endpoint_speed_mps": float(endpoint_speeds[0]),
                "dst_endpoint_speed_mps": float(endpoint_speeds[1]),
                "min_endpoint_speed_mps": minimum_speed,
                "motion_regime": regime,
                "selection_priority_sha256": pair_priority(
                    selection_seed,
                    scenario_id,
                    src,
                    dst,
                    regime,
                ),
            }
        )
    return candidates


def _base_regime(graph: Any, row: Any) -> str:
    velocities = np.asarray(
        graph["observed_velocities_filled"], dtype=np.float64
    )
    endpoint_speeds = np.linalg.norm(
        velocities[[int(row.src), int(row.dst)], -1], axis=1
    )
    return motion_regime(float(endpoint_speeds.min()))


def generate_manifest(
    graph_dir: Path,
    graph_manifest: pd.DataFrame,
    base_manifest: pd.DataFrame,
    label_config: RiskLabelConfig,
    *,
    selection_seed: int,
    preferred_regime: str = MOVING_MOTION_REGIME,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if int(selection_seed) < 0:
        raise ValueError("selection seed must be non-negative")
    if str(preferred_regime) not in MOTION_REGIMES:
        raise ValueError(f"unsupported preferred motion regime: {preferred_regime!r}")
    train_rows = graph_manifest[graph_manifest["split"].eq("train")]
    graph_rows = {
        str(row.scenario_id): row
        for row in train_rows.itertuples(index=False)
    }
    base_scenarios = set(base_manifest["scenario_id"].astype(str))
    if len(base_scenarios) != len(base_manifest):
        raise ValueError("base manifest contains duplicate scenarios")
    missing = sorted(base_scenarios - set(graph_rows))
    if missing:
        raise ValueError(
            f"base manifest scenarios are absent from train split: {missing[:3]}"
        )

    selected_rows: list[dict[str, Any]] = []
    base_motion_counts = {"slow_lt_0p5": 0, "moving_ge_0p5": 0}
    selected_motion_counts = {"slow_lt_0p5": 0, "moving_ge_0p5": 0}
    changed_pairs = 0
    scenarios_with_preferred_candidate = 0
    scenarios_with_slow_candidate = 0
    scenarios_with_moving_candidate = 0
    candidate_pair_count = 0
    config_hash = label_config_hash(label_config)

    for base_row in base_manifest.sort_values("scenario_id").itertuples(
        index=False
    ):
        scenario_id = str(base_row.scenario_id)
        graph_row = graph_rows[scenario_id]
        graph_path = resolve_verified_graph_path(graph_row, graph_dir)
        with np.load(graph_path, allow_pickle=True) as graph:
            label_bundle = labels_for_graph(graph, label_config)
            pair_groups = eligible_negative_pair_groups(
                graph,
                label_config,
                require_strict_label=True,
                perturb_window=10,
                require_bi_endpoint_contiguous=True,
                label_bundle=label_bundle,
            )
            candidates = _scenario_candidates(
                graph,
                pair_groups,
                scenario_id=scenario_id,
                selection_seed=int(selection_seed),
            )
            selected = choose_motion_candidate(
                candidates,
                preferred_regime=str(preferred_regime),
            )
            alpha, allocation = fixed_symmetric_bi_endpoint_allocation(
                graph,
                src=int(selected["src"]),
                dst=int(selected["dst"]),
                displacement_m=0.2,
            )
            base_regime = _base_regime(graph, base_row)

        if not np.isclose(
            alpha, FIXED_SYMMETRIC_BIEND_ALPHA, atol=1e-12, rtol=0.0
        ):
            raise RuntimeError("fixed symmetric allocation changed")
        selected_regime = str(selected["motion_regime"])
        base_motion_counts[base_regime] += 1
        selected_motion_counts[selected_regime] += 1
        has_slow_candidate = any(
            item["motion_regime"] == SLOW_MOTION_REGIME
            for item in candidates
        )
        has_moving_candidate = any(
            item["motion_regime"] == MOVING_MOTION_REGIME
            for item in candidates
        )
        scenarios_with_slow_candidate += int(has_slow_candidate)
        scenarios_with_moving_candidate += int(has_moving_candidate)
        scenarios_with_preferred_candidate += int(
            any(
                item["motion_regime"] == str(preferred_regime)
                for item in candidates
            )
        )
        candidate_pair_count += len(candidates)
        base_pair = frozenset((int(base_row.src), int(base_row.dst)))
        selected_pair = frozenset(
            (int(selected["src"]), int(selected["dst"]))
        )
        pair_changed = base_pair != selected_pair
        changed_pairs += int(pair_changed)

        selected_rows.append(
            {
                **selected,
                "split": "train",
                "displacement_m": 0.2,
                "perturb_window": 10,
                "ramp_style": "minimum_jerk",
                "velocity_mode": "residual",
                "poison_label": 1,
                "seed": int(selection_seed),
                "label_mode": label_config.label_mode,
                "label_config_hash": config_hash,
                "require_strict_label": True,
                "label_unit": PAIR_LABEL_UNIT,
                "allocation_policy": (
                    ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
                ),
                "allocation_alpha": FIXED_SYMMETRIC_BIEND_ALPHA,
                "allocation_total_feature_energy": float(
                    allocation["allocation_total_feature_energy"]
                ),
                "allocation_incident_edge_energy": float(
                    allocation["allocation_incident_edge_energy"]
                ),
                "allocation_endpoint_node_energy": float(
                    allocation["allocation_endpoint_node_energy"]
                ),
                "allocation_candidate_count": int(
                    allocation["allocation_candidate_count"]
                ),
                "allocation_non_target_incident_edges": int(
                    allocation["allocation_non_target_incident_edges"]
                ),
                "base_src": int(base_row.src),
                "base_dst": int(base_row.dst),
                "base_motion_regime": base_regime,
                "pair_changed_from_v6": bool(pair_changed),
                "eligible_pair_count": int(len(candidates)),
                "moving_pair_available": bool(
                    has_moving_candidate
                ),
                "slow_pair_available": bool(has_slow_candidate),
                "preferred_motion_regime": str(preferred_regime),
                "preferred_pair_available": bool(
                    selected_regime == str(preferred_regime)
                ),
            }
        )

    frame = pd.DataFrame(selected_rows)
    required_first = list(POISON_MANIFEST_COLUMNS)
    frame = frame[
        required_first
        + sorted(
            column for column in frame.columns if column not in required_first
        )
    ].sort_values("scenario_id").reset_index(drop=True)
    if set(frame["scenario_id"].astype(str)) != base_scenarios:
        raise RuntimeError("v6.2 changed the frozen poisoned scenario set")
    if len(frame) != len(base_manifest):
        raise RuntimeError("v6.2 changed the poison scenario count")
    if selected_motion_counts[str(preferred_regime)] != (
        scenarios_with_preferred_candidate
    ):
        raise RuntimeError("motion-prioritized selection missed a preferred pair")

    experiment_version = (
        SLOW_PRIORITIZED_VERSION
        if str(preferred_regime) == SLOW_MOTION_REGIME
        else VERSION
    )

    metadata = {
        "scope": "offline_authorized_av2_model_robustness",
        "experiment": experiment_version,
        "status": "frozen_train_only_pilot_manifest",
        "development_only": True,
        "formal_test_eligible": False,
        "test_access_authorized": False,
        "training_manifest_eligible": True,
        "split": "train",
        "selection_authority": (
            "stored_train_trajectories_dynamic_labels_and_frozen_v6_scenarios"
        ),
        "selection_objective": (
            f"prefer_{str(preferred_regime)}_then_stable_hash_within_scenario_v1"
        ),
        "selection_seed": int(selection_seed),
        "preferred_motion_regime": str(preferred_regime),
        "motion_speed_threshold_mps": MOTION_SPEED_THRESHOLD_MPS,
        "requested_poison_scenario_rate": POISON_RATE,
        "eligible_scenarios": None,
        "poisoned_scenarios": int(len(frame)),
        "scenario_ids_preserved": True,
        "base_motion_counts": base_motion_counts,
        "selected_motion_counts": selected_motion_counts,
        "scenarios_with_moving_candidate": int(
            scenarios_with_moving_candidate
        ),
        "scenarios_with_slow_candidate": int(scenarios_with_slow_candidate),
        "scenarios_with_preferred_candidate": int(
            scenarios_with_preferred_candidate
        ),
        "candidate_pair_count": int(candidate_pair_count),
        "changed_pairs": int(changed_pairs),
        "changed_pair_rate": float(changed_pairs / len(frame)),
        "orientation_policy": (
            ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED
        ),
        "allocation_policy": ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
        "fixed_allocation_alpha": FIXED_SYMMETRIC_BIEND_ALPHA,
        "edge_feature_mode": MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        "require_strict_label": True,
        "strict_crossfit_required": False,
        "label_unit": PAIR_LABEL_UNIT,
        "label_config": label_config_dict(label_config),
        "label_config_hash": config_hash,
        "ordinary_bce_unchanged": True,
        "graph_model_unchanged": True,
        "trajectory_trigger_unchanged": True,
        "original_validation_used": False,
        "original_test_used": False,
        "model_outputs_used": False,
        "victim_queries": 0,
        "forbidden_information_used": [],
    }
    return frame, metadata


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if (output.exists() or metadata_path.exists()) and not args.force:
        raise FileExistsError("v6.2 manifest output already exists")
    graph_dir = Path(args.graph_dir)
    graph_manifest_path = resolve_graph_manifest(
        graph_dir, args.graph_manifest
    )
    graph_contract = validate_graph_manifest_contract(graph_manifest_path)
    graph_manifest = load_graph_manifest(graph_dir, graph_manifest_path)
    label_config = RiskLabelConfig(
        label_mode="dynamic_risk",
        risk_base_distance_m=5.0,
        risk_reaction_time_s=1.0,
        risk_safe_decel_mps2=4.0,
    )
    base_manifest_path = Path(args.base_manifest)
    base_manifest, base_manifest_sha = load_poison_manifest(
        base_manifest_path,
        label_config,
        expected_split="train",
        require_strict_label=True,
        require_metadata_binding=True,
        expected_graph_manifest_sha256=sha256_file(graph_manifest_path),
        require_strict_crossfit_binding=True,
    )
    frame, metadata = generate_manifest(
        graph_dir,
        graph_manifest,
        base_manifest,
        label_config,
        selection_seed=int(args.selection_seed),
        preferred_regime=str(args.preferred_regime),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(frame, output)
    base_metadata_path = base_manifest_path.with_suffix(
        base_manifest_path.suffix + ".metadata.json"
    )
    base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    eligible_scenarios = int(base_metadata.get("eligible_scenarios", 0))
    if eligible_scenarios <= 0:
        raise ValueError("base metadata has no positive eligible_scenarios")
    expected_rows = int(round(eligible_scenarios * POISON_RATE))
    if expected_rows != len(frame):
        raise RuntimeError(
            "base poison-rate/count contract is inconsistent: "
            f"round({eligible_scenarios} * {POISON_RATE})={expected_rows}, "
            f"rows={len(frame)}"
        )
    metadata.update(
        {
            "manifest_path": str(output),
            "manifest_sha256": sha256_file(output),
            "base_manifest_path": str(base_manifest_path),
            "base_manifest_sha256": base_manifest_sha,
            "base_metadata_path": str(base_metadata_path),
            "base_metadata_sha256": sha256_file(base_metadata_path),
            "eligible_scenarios": eligible_scenarios,
            "graph_manifest_path": str(graph_manifest_path),
            "graph_manifest_sha256": sha256_file(graph_manifest_path),
            "graph_manifest_contract": graph_contract,
        }
    )
    _atomic_json(metadata, metadata_path)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
