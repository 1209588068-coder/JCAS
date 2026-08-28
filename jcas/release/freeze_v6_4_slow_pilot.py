#!/usr/bin/env python3
"""Freeze the v6.4-S low-speed-pair validation pilot before training.

The contract is development-only.  It binds the train-only slow-prioritized
manifest, the reused clean reference, the implementation, and the frozen
continuation gates.  It never authorizes test evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from jcas import PROJECT_ROOT
from jcas.core.motion_normalized_features import (
    MOTION_NORMALIZED_EDGE_FEATURE_MODE,
    edge_feature_protocol,
)
from jcas.core.poison import sha256_file
from jcas.release.source_integrity import (
    deterministic_source_archive,
    environment_summary,
)
from jcas.workflows.motion_prioritized_manifest import (
    SLOW_MOTION_REGIME,
    SLOW_PRIORITIZED_VERSION,
)


ROOT = PROJECT_ROOT
DATE_TAG = "20260828"
VERSION = "v6_4_slow_prioritized_pair_validation_pilot_v1"
DEFAULT_OUTPUT_DIR = Path("record/v6_4/contracts/pretraining")
GRAPH_MANIFEST = Path("graphs/av2mf_graph_v4/manifest_grouped_v5.csv")
GRAPH_CONTRACT = Path(str(GRAPH_MANIFEST) + ".metadata.json")
PARENT_MANIFEST = Path("record/v6/poison_rate005_fixed_symmetric.csv")
PARENT_METADATA = Path(str(PARENT_MANIFEST) + ".metadata.json")
PILOT_MANIFEST = Path("record/v6_4/poison_rate005_slow_prioritized.csv")
PILOT_METADATA = Path(str(PILOT_MANIFEST) + ".metadata.json")
CLEAN_RESULT = Path(
    "record/v6_1/clean_seed20260621/"
    "genconv_strict_seed20260621_20260827_174820/result.json"
)
CLEAN_CHECKPOINT = CLEAN_RESULT.with_name("best_model.pt")
V6_1_RELEASE = Path(
    "record/v6_1/contracts/v6_1a_pretraining_r2/"
    "v6_1a_pretraining_release_20260827.json"
)
V6_1_RELEASE_ANCHOR = V6_1_RELEASE.with_suffix(".sha256")

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path("pyproject.toml"),
    Path("requirements.txt"),
    Path("docs/V6_RUNBOOK.md"),
    Path("docs/V6_1_MOTION_NORMALIZED_PILOT.md"),
    Path("jcas/__init__.py"),
    Path("jcas/core/__init__.py"),
    Path("jcas/core/graph_splits.py"),
    Path("jcas/core/graph_splits_v5.py"),
    Path("jcas/core/models.py"),
    Path("jcas/core/motion_normalized_features.py"),
    Path("jcas/core/poison.py"),
    Path("jcas/core/risk_labels.py"),
    Path("jcas/core/trajectory_trigger.py"),
    Path("jcas/workflows/__init__.py"),
    Path("jcas/workflows/evaluator.py"),
    Path("jcas/workflows/motion_prioritized_manifest.py"),
    Path("jcas/workflows/poison_manifest.py"),
    Path("jcas/workflows/trainer.py"),
    Path("jcas/release/__init__.py"),
    Path("jcas/release/freeze_v6_4_slow_pilot.py"),
    Path("jcas/release/source_integrity.py"),
    Path("tests/test_blackbox_pipeline.py"),
    Path("tests/test_v6_2_motion_prioritized.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the v6.4-S validation-only pilot"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def _atomic_text(value: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _load_json(ROOT / PILOT_METADATA)
    if metadata.get("experiment") != SLOW_PRIORITIZED_VERSION:
        raise RuntimeError("unexpected slow-prioritized manifest experiment")
    if metadata.get("preferred_motion_regime") != SLOW_MOTION_REGIME:
        raise RuntimeError("pilot manifest does not prefer the frozen slow regime")
    if metadata.get("manifest_sha256") != sha256_file(ROOT / PILOT_MANIFEST):
        raise RuntimeError("pilot manifest SHA-256 binding failed")
    if int(metadata.get("poisoned_scenarios", -1)) != 5776:
        raise RuntimeError("pilot must preserve exactly 5,776 poisoned scenarios")
    if metadata.get("scenario_ids_preserved") is not True:
        raise RuntimeError("pilot changed the frozen poisoned scenario set")
    if metadata.get("original_validation_used") is not False:
        raise RuntimeError("pilot target selection accessed validation")
    if metadata.get("original_test_used") is not False:
        raise RuntimeError("pilot target selection accessed test")
    if metadata.get("model_outputs_used") is not False:
        raise RuntimeError("pilot target selection used model outputs")
    if metadata.get("ordinary_bce_unchanged") is not True:
        raise RuntimeError("pilot no longer uses ordinary BCE")
    counts = metadata.get("selected_motion_counts", {})
    if sum(int(value) for value in counts.values()) != 5776:
        raise RuntimeError("pilot motion counts do not sum to 5,776")
    preferred = int(metadata.get("scenarios_with_preferred_candidate", -1))
    if int(counts.get(SLOW_MOTION_REGIME, -1)) != preferred:
        raise RuntimeError("pilot failed to select every available slow pair")

    clean = _load_json(ROOT / CLEAN_RESULT)
    config = clean.get("config", {})
    if config.get("poison_manifest") is not None:
        raise RuntimeError("clean reference is not clean")
    if config.get("evaluate_test") is not False or clean.get("test_metrics") is not None:
        raise RuntimeError("clean reference accessed test")
    if config.get("edge_feature_mode") != MOTION_NORMALIZED_EDGE_FEATURE_MODE:
        raise RuntimeError("clean reference uses the wrong edge representation")
    return metadata, clean


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / Path(args.output_dir)).resolve()
    output_dir.relative_to(ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_manifest = output_dir / f"v6_4_code_assets_{DATE_TAG}.sha256"
    source_archive = output_dir / f"v6_4_source_snapshot_{DATE_TAG}.tar.gz"
    environment_path = output_dir / f"v6_4_environment_{DATE_TAG}.json"
    release_path = output_dir / f"v6_4_pretraining_release_{DATE_TAG}.json"
    anchor_path = output_dir / f"v6_4_pretraining_release_{DATE_TAG}.sha256"
    outputs = (code_manifest, source_archive, environment_path, release_path, anchor_path)
    if any(path.exists() for path in outputs):
        raise FileExistsError("v6.4 pretraining release already exists")

    required = SOURCE_PATHS + (
        GRAPH_MANIFEST,
        GRAPH_CONTRACT,
        PARENT_MANIFEST,
        PARENT_METADATA,
        PILOT_MANIFEST,
        PILOT_METADATA,
        CLEAN_RESULT,
        CLEAN_CHECKPOINT,
        V6_1_RELEASE,
        V6_1_RELEASE_ANCHOR,
    )
    missing = [path.as_posix() for path in required if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"v6.4 freeze inputs are missing: {missing}")
    metadata, clean = _verify_inputs()

    code_lines = [
        f"{sha256_file(ROOT / path)}  {path.as_posix()}"
        for path in sorted(SOURCE_PATHS, key=lambda item: item.as_posix())
    ]
    _atomic_text("\n".join(code_lines) + "\n", code_manifest)
    deterministic_source_archive(SOURCE_PATHS, source_archive)
    _atomic_json(environment_summary(), environment_path)

    release = {
        "scope": "offline_authorized_av2_model_robustness",
        "version": VERSION,
        "status": "frozen_before_single_seed_victim_training",
        "development_only": True,
        "formal_test_eligible": False,
        "test_access_authorized": False,
        "pilot_seed": 20260621,
        "method_change": (
            "within_the_frozen_v6_poisoned_scenarios_prefer_a_data_eligible_"
            "slow_pair_then_use_a_stable_hash"
        ),
        "threat_model": "data_only_poisoning_with_ordinary_bce",
        "unchanged_components": [
            "grouped_v5_graph_split",
            "dynamic_risk_labels_d0_5_tau_1_decel_4",
            "v6_poisoned_scenario_set_and_five_percent_rate",
            "relative_displacement_0p2m_k10_minimum_jerk_velocity_residual",
            "fixed_symmetric_biend_alpha_0p5",
            "genconv_architecture_and_ordinary_bce",
            "public_validation_target_pool",
        ],
        "edge_feature_protocol": edge_feature_protocol(
            MOTION_NORMALIZED_EDGE_FEATURE_MODE
        ),
        "training_protocol": {
            "model_name": "genconv",
            "seed": 20260621,
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.1,
            "norm": "layer",
            "decoder_hidden_dim": 128,
            "decoder_num_layers": 2,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "epochs": 50,
            "checkpoint_metric": "val_loss",
            "require_strict_label": True,
            "evaluate_test": False,
        },
        "continuation_gates": {
            "slow_lt_0p5_incremental_asr_min": 0.80,
            "overall_incremental_asr_min": 0.60,
            "slow_lt_0p5_clean_activation_max": 0.025,
            "adjacent_negative_fp_incremental_max": 0.015,
            "clean_pair_auc_min": 0.9849,
            "decision": "run_two_additional_seeds_only_if_all_gates_pass",
        },
        "validation_protocol": {
            "split": "val",
            "target_seed": 20260621,
            "motion_regime_boundary_mps": 0.5,
            "common_threshold_source": (
                "same_seed_v6_1_clean_validation_pair_threshold"
            ),
        },
        "pilot_manifest": {
            "path": PILOT_MANIFEST.as_posix(),
            "sha256": sha256_file(ROOT / PILOT_MANIFEST),
            "metadata_path": PILOT_METADATA.as_posix(),
            "metadata_sha256": sha256_file(ROOT / PILOT_METADATA),
            "base_motion_counts": metadata["base_motion_counts"],
            "selected_motion_counts": metadata["selected_motion_counts"],
            "changed_pairs": metadata["changed_pairs"],
        },
        "clean_reference": {
            "result_path": CLEAN_RESULT.as_posix(),
            "result_sha256": sha256_file(ROOT / CLEAN_RESULT),
            "checkpoint_path": CLEAN_CHECKPOINT.as_posix(),
            "checkpoint_sha256": sha256_file(ROOT / CLEAN_CHECKPOINT),
            "validation_pair_threshold": clean["val_pair_metrics"]["threshold"],
        },
        "v6_1_parent_release": {
            "path": V6_1_RELEASE.as_posix(),
            "sha256": sha256_file(ROOT / V6_1_RELEASE),
            "anchor_path": V6_1_RELEASE_ANCHOR.as_posix(),
            "anchor_sha256": sha256_file(ROOT / V6_1_RELEASE_ANCHOR),
        },
        "code_asset_manifest": {
            "path": code_manifest.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(code_manifest),
            "entries": len(code_lines),
        },
        "source_archive": {
            "path": source_archive.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(source_archive),
            "deterministic": True,
        },
        "environment": {
            "path": environment_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(environment_path),
        },
    }
    _atomic_json(release, release_path)
    release_sha = sha256_file(release_path)
    _atomic_text(
        f"{release_sha}  {release_path.relative_to(ROOT).as_posix()}\n",
        anchor_path,
    )
    print(f"v6.4 release: {release_path.relative_to(ROOT)}")
    print(f"external trust anchor SHA-256: {release_sha}")


if __name__ == "__main__":
    main()
