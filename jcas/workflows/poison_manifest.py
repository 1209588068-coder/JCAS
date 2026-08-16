#!/usr/bin/env python3
"""Generate and freeze a zero-query training poison manifest.

No model/checkpoint/probability/decision threshold is accepted by this CLI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jcas.core.poison import (
    ORIENTATION_POLICIES,
    ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
    PAIR_LABEL_UNIT,
    POISON_MANIFEST_COLUMNS,
    eligible_negative_pair_groups,
    select_pair_orientation,
    sha256_file,
)
from jcas.core.risk_labels import (
    RiskLabelConfig,
    label_config_dict,
    label_config_hash,
    labels_for_graph,
    selected_label_computable_mask,
)
from jcas.workflows.trainer import validate_graph_manifest_contract


def resolve_graph_manifest(
    graph_dir: Path, manifest_path: str | Path | None
) -> Path:
    if manifest_path is None:
        path = graph_dir / "manifest.csv"
    else:
        requested = Path(manifest_path)
        path = next(
            (
                candidate
                for candidate in (requested, graph_dir / requested)
                if candidate.exists()
            ),
            requested,
        )
    if not path.exists():
        raise FileNotFoundError(f"graph manifest does not exist: {path}")
    return path


def load_graph_manifest(
    graph_dir: Path, manifest_path: str | Path | None = None
) -> pd.DataFrame:
    path = resolve_graph_manifest(graph_dir, manifest_path)
    manifest = pd.read_csv(path, dtype={"scenario_id": str})
    manifest = manifest[manifest["status"].isin(["built", "exists"])].copy()
    manifest = manifest[manifest["graph_path"].notna()].sort_values("scenario_id")
    if manifest.empty:
        raise RuntimeError(f"graph manifest has no built rows: {path}")
    if not manifest["split"].isin(["train", "val", "test"]).all():
        raise ValueError(f"graph manifest has invalid or missing split values: {path}")
    if manifest["scenario_id"].duplicated().any():
        raise ValueError(f"graph manifest has duplicate scenario_id values: {path}")
    if "graph_sha256" not in manifest:
        raise ValueError(
            "graph manifest has no graph_sha256; use the verified v3 split"
        )
    hashes = manifest["graph_sha256"].astype(str).str.lower()
    if not bool(hashes.str.fullmatch(r"[0-9a-f]{64}", na=False).all()):
        raise ValueError("graph manifest contains invalid graph_sha256 values")
    manifest["graph_sha256"] = hashes
    return manifest


def resolve_graph_path(path_text: str, graph_dir: Path) -> Path:
    path = Path(path_text)
    for candidate in (path, graph_dir / path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve graph path: {path_text}")


def resolve_verified_graph_path(row: Any, graph_dir: Path) -> Path:
    path = resolve_graph_path(str(row.graph_path), graph_dir)
    actual = sha256_file(path)
    expected = str(row.graph_sha256).lower()
    if actual != expected:
        raise RuntimeError(
            f"graph SHA-256 mismatch for scenario {row.scenario_id}"
        )
    return path


def _atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a model-independent training poison manifest.")
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument(
        "--graph-manifest",
        default=None,
        help="Explicit split manifest; defaults to <graph-dir>/manifest.csv",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--poison-scenario-rate",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--orientation-policy",
        choices=ORIENTATION_POLICIES,
        default=ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
        help="Data-only rule for choosing which endpoint of a selected pair is moved",
    )
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--require-strict-label", action="store_true")
    parser.add_argument(
        "--label-mode",
        choices=["dynamic_risk"],
        default="dynamic_risk",
    )
    parser.add_argument("--risk-base-distance-m", type=float, default=5.0)
    parser.add_argument("--risk-reaction-time-s", type=float, default=1.0)
    parser.add_argument("--risk-safe-decel-mps2", type=float, default=4.0)
    parser.add_argument("--force", action="store_true", help="Explicitly replace an existing frozen manifest")
    return parser.parse_args()


def selected_scenario_indices(
    candidate_count: int, selected_count: int, seed: int
) -> np.ndarray:
    """Return the deterministic 5% scenario sample."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if selected_count < 1 or selected_count > candidate_count:
        raise ValueError("selected_count must be within candidate_count")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    scenario_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0x5343454E])
    )
    priority = scenario_rng.permutation(int(candidate_count))
    return np.sort(priority[: int(selected_count)])


def generate_manifest(
    graph_dir: Path,
    graph_manifest: pd.DataFrame,
    label_config: RiskLabelConfig,
    poison_scenario_rate: float,
    seed: int,
    require_strict_label: bool,
    orientation_policy: str = ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not np.isclose(
        float(poison_scenario_rate), 0.05, atol=1e-12, rtol=0.0
    ):
        raise ValueError("main-line poison_scenario_rate must equal 0.05")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    edge_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0x45444745])
    )
    candidates: list[dict[str, Any]] = []
    eligible_pairs = 0
    eligible_orientations = 0
    supervised_edges = 0
    train_rows = graph_manifest[graph_manifest["split"] == "train"].sort_values("scenario_id")

    for row in train_rows.itertuples(index=False):
        path = resolve_verified_graph_path(row, graph_dir)
        with np.load(path, allow_pickle=True) as graph:
            label_bundle = labels_for_graph(graph, label_config)
            pair_groups = eligible_negative_pair_groups(
                graph,
                label_config,
                require_strict_label=require_strict_label,
                perturb_window=10,
                label_bundle=label_bundle,
            )
            supervision = np.asarray(graph["supervision_edge_mask"], dtype=bool)
            if require_strict_label:
                supervision &= np.asarray(graph["label_strict_mask"], dtype=bool)
            supervision &= selected_label_computable_mask(
                graph, label_config, label_bundle
            )
            supervised_edges += int(supervision.sum())
            eligible_pairs += int(len(pair_groups))
            eligible_orientations += int(sum(len(group) for group in pair_groups))
            if not pair_groups:
                continue
            pair_group = pair_groups[int(edge_rng.integers(len(pair_groups)))]
            edge_id, orientation_audit = select_pair_orientation(
                graph,
                pair_group,
                policy=orientation_policy,
                perturb_window=10,
            )
            src = int(graph["edge_index"][0, edge_id])
            dst = int(graph["edge_index"][1, edge_id])
            candidates.append(
                {
                    "scenario_id": str(row.scenario_id),
                    "split": "train",
                    "src": src,
                    "dst": dst,
                    "src_track_id": str(graph["node_track_ids"][src]),
                    "dst_track_id": str(graph["node_track_ids"][dst]),
                    "orientation_policy": str(orientation_policy),
                    "orientation_destination_mean_speed_mps": float(
                        orientation_audit["destination_mean_speed_mps"]
                    ),
                    "orientation_candidate_count": int(
                        orientation_audit["candidate_orientations"]
                    ),
                }
            )

    if not candidates:
        raise RuntimeError("no training scenario contains an eligible negative edge")
    selected_count = max(1, int(round(len(candidates) * float(poison_scenario_rate))))
    selected_indices = selected_scenario_indices(
        len(candidates), selected_count, int(seed)
    )
    config_hash = label_config_hash(label_config)
    rows: list[dict[str, Any]] = []
    for index in selected_indices:
        item = dict(candidates[int(index)])
        item.update(
            {
                "displacement_m": 0.2,
                "perturb_window": 10,
                "ramp_style": "minimum_jerk",
                "velocity_mode": "residual",
                "poison_label": 1,
                "seed": int(seed),
                "label_mode": label_config.label_mode,
                "label_config_hash": config_hash,
                "require_strict_label": bool(require_strict_label),
                "label_unit": PAIR_LABEL_UNIT,
            }
        )
        rows.append(item)
    orientation_columns = [
        "orientation_policy",
        "orientation_destination_mean_speed_mps",
        "orientation_candidate_count",
    ]
    result = pd.DataFrame(
        rows,
        columns=[
            *POISON_MANIFEST_COLUMNS,
            *orientation_columns,
        ],
    ).sort_values("scenario_id")
    summary = {
        "selection_authority": "stored_data_and_ground_truth_only",
        "split": "train",
        "seed": int(seed),
        "edge_sampling_scheme": "one_uniform_eligible_unordered_pair_then_data_only_orientation_per_scenario_v3",
        "orientation_policy": str(orientation_policy),
        "scenario_sampling_scheme": "seeded_random_permutation_prefix_v1",
        "nested_across_rates_when_seed_and_eligibility_match": True,
        "requested_poison_scenario_rate": float(poison_scenario_rate),
        "eligible_scenarios": int(len(candidates)),
        "poisoned_scenarios": int(len(result)),
        "realized_poison_scenario_rate": float(len(result) / len(candidates)),
        "eligible_negative_pairs": int(eligible_pairs),
        "eligible_trigger_orientations": int(eligible_orientations),
        "supervised_train_edges": int(supervised_edges),
        "poisoned_pairs": int(len(result)),
        "poisoned_directed_edge_labels": int(2 * len(result)),
        "poison_pair_rate_among_eligible_pairs": float(
            len(result) / max(eligible_pairs, 1)
        ),
        "poison_directed_edge_rate_among_supervised_edges": float(
            (2 * len(result)) / max(supervised_edges, 1)
        ),
        "require_strict_label": bool(require_strict_label),
        "label_config": label_config_dict(label_config),
        "label_config_hash": config_hash,
        "forbidden_information_used": [],
    }
    return result, summary


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if (output.exists() or metadata_path.exists()) and not args.force:
        raise FileExistsError("frozen manifest already exists; use --force to replace it explicitly")
    label_config = RiskLabelConfig(
        label_mode=args.label_mode,
        risk_base_distance_m=args.risk_base_distance_m,
        risk_reaction_time_s=args.risk_reaction_time_s,
        risk_safe_decel_mps2=args.risk_safe_decel_mps2,
    )
    graph_dir = Path(args.graph_dir)
    graph_manifest_path = resolve_graph_manifest(graph_dir, args.graph_manifest)
    graph_manifest_contract = validate_graph_manifest_contract(
        graph_manifest_path
    )
    graph_manifest = load_graph_manifest(graph_dir, graph_manifest_path)
    frame, summary = generate_manifest(
        graph_dir,
        graph_manifest,
        label_config,
        args.poison_scenario_rate,
        args.seed,
        args.require_strict_label,
        args.orientation_policy,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv_write(frame, output)
    summary["manifest_path"] = str(output)
    summary["manifest_sha256"] = sha256_file(output)
    summary["graph_manifest_path"] = str(graph_manifest_path)
    summary["graph_manifest_sha256"] = sha256_file(graph_manifest_path)
    summary["graph_manifest_contract"] = graph_manifest_contract
    _atomic_json_write(summary, metadata_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
