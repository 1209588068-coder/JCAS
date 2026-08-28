#!/usr/bin/env python3
"""Freeze the validation-only v6.3 K10/K4 schedule pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from jcas import PROJECT_ROOT
from jcas.core.motion_schedule_trigger import MOTION_REGIME_K4_K10_SCHEDULE
from jcas.core.poison import sha256_file
from jcas.release.source_integrity import (
    deterministic_source_archive,
    environment_summary,
)


ROOT = PROJECT_ROOT
DATE_TAG = "20260828"
VERSION = "v6_3_motion_regime_k4_k10_validation_pilot_v1"
DEFAULT_OUTPUT_DIR = Path("record/v6_3/contracts/pretraining")
GRAPH_MANIFEST = Path("graphs/av2mf_graph_v4/manifest_grouped_v5.csv")
GRAPH_CONTRACT = Path(str(GRAPH_MANIFEST) + ".metadata.json")
SCHEDULE_MANIFEST = Path("record/v6_3/poison_rate005_motion_k4_k10.csv")
SCHEDULE_METADATA = Path(str(SCHEDULE_MANIFEST) + ".metadata.json")
SCHEDULE_VERIFICATION = Path(
    "record/v6_3/verification/motion_schedule_verification.json"
)
V6_2_RELEASE = Path(
    "record/v6_2/contracts/pretraining/v6_2_pretraining_release_20260827.json"
)
V6_2_RELEASE_ANCHOR = V6_2_RELEASE.with_suffix(".sha256")
CLEAN_RESULT = Path(
    "record/v6_1/clean_seed20260621/"
    "genconv_strict_seed20260621_20260827_174820/result.json"
)
CLEAN_CHECKPOINT = CLEAN_RESULT.with_name("best_model.pt")

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path("pyproject.toml"),
    Path("requirements.txt"),
    Path("docs/V6_RUNBOOK.md"),
    Path("docs/V6_1_MOTION_NORMALIZED_PILOT.md"),
    Path("docs/V6_2_MOTION_PRIORITIZED_PILOT.md"),
    Path("docs/V6_3_MOTION_SCHEDULE_PILOT.md"),
    Path("jcas/__init__.py"),
    Path("jcas/core/models.py"),
    Path("jcas/core/motion_normalized_features.py"),
    Path("jcas/core/motion_schedule_trigger.py"),
    Path("jcas/core/poison.py"),
    Path("jcas/core/risk_labels.py"),
    Path("jcas/core/trajectory_trigger.py"),
    Path("jcas/workflows/evaluator.py"),
    Path("jcas/workflows/graph_builder.py"),
    Path("jcas/workflows/motion_prioritized_manifest.py"),
    Path("jcas/workflows/motion_schedule_manifest.py"),
    Path("jcas/workflows/poison_manifest.py"),
    Path("jcas/workflows/trainer.py"),
    Path("jcas/release/freeze_v6_3_pilot.py"),
    Path("jcas/release/source_integrity.py"),
    Path("jcas/release/verify_motion_schedule_manifest.py"),
    Path("tests/test_blackbox_pipeline.py"),
    Path("tests/test_v6_2_motion_prioritized.py"),
    Path("tests/test_v6_3_motion_schedule.py"),
    Path("tests/test_v6_fixed_symmetric_manifest.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the v6.3 validation-only pilot before training"
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


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / Path(args.output_dir)).resolve()
    output_dir.relative_to(ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_manifest = output_dir / f"v6_3_code_assets_{DATE_TAG}.sha256"
    source_archive = output_dir / f"v6_3_source_snapshot_{DATE_TAG}.tar.gz"
    environment_path = output_dir / f"v6_3_environment_{DATE_TAG}.json"
    release_path = output_dir / f"v6_3_pretraining_release_{DATE_TAG}.json"
    anchor_path = output_dir / f"v6_3_pretraining_release_{DATE_TAG}.sha256"
    outputs = (
        code_manifest,
        source_archive,
        environment_path,
        release_path,
        anchor_path,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("v6.3 pretraining release already exists")
    required = SOURCE_PATHS + (
        GRAPH_MANIFEST,
        GRAPH_CONTRACT,
        SCHEDULE_MANIFEST,
        SCHEDULE_METADATA,
        SCHEDULE_VERIFICATION,
        V6_2_RELEASE,
        V6_2_RELEASE_ANCHOR,
        CLEAN_RESULT,
        CLEAN_CHECKPOINT,
    )
    missing = [path.as_posix() for path in required if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"v6.3 freeze inputs are missing: {missing}")

    metadata = _load_json(ROOT / SCHEDULE_METADATA)
    verification = _load_json(ROOT / SCHEDULE_VERIFICATION)
    if metadata.get("trigger_schedule_id") != MOTION_REGIME_K4_K10_SCHEDULE:
        raise RuntimeError("v6.3 metadata schedule changed")
    if metadata.get("manifest_sha256") != sha256_file(ROOT / SCHEDULE_MANIFEST):
        raise RuntimeError("v6.3 manifest SHA-256 binding failed")
    if verification.get("status") != "ok":
        raise RuntimeError("v6.3 schedule verification did not pass")
    if verification.get("poison_manifest", {}).get("sha256") != (
        metadata["manifest_sha256"]
    ):
        raise RuntimeError("v6.3 verification used a different manifest")
    if int(verification.get("verified_rows", -1)) != int(
        metadata.get("poisoned_scenarios", -2)
    ):
        raise RuntimeError("v6.3 verification row count changed")
    if verification.get("schedule_counts") != metadata.get(
        "trigger_schedule_counts"
    ):
        raise RuntimeError("v6.3 verification schedule counts changed")

    code_lines = [
        f"{sha256_file(ROOT / path)}  {path.as_posix()}"
        for path in sorted(SOURCE_PATHS, key=lambda item: item.as_posix())
    ]
    _atomic_text("\n".join(code_lines) + "\n", code_manifest)
    deterministic_source_archive(SOURCE_PATHS, source_archive)
    _atomic_json(environment_summary(), environment_path)
    clean = _load_json(ROOT / CLEAN_RESULT)
    release = {
        "scope": "offline_authorized_av2_model_robustness",
        "version": VERSION,
        "status": "frozen_before_v6_3_single_seed_training",
        "development_only": True,
        "formal_test_eligible": False,
        "test_access_authorized": False,
        "pilot_seed": 20260621,
        "threat_model": "data_only_poisoning_with_ordinary_bce",
        "method_change": "slow_pair_K10_and_moving_pair_K4_only",
        "trigger_schedule_id": MOTION_REGIME_K4_K10_SCHEDULE,
        "unchanged_components": [
            "grouped_v5_graph_split",
            "dynamic_risk_labels_d0_5_tau_1_decel_4",
            "v6_2_poison_scenarios_and_target_pairs",
            "relative_displacement_0p2m",
            "fixed_symmetric_alpha_0p5",
            "minimum_jerk_and_velocity_residual",
            "genconv_and_ordinary_bce",
            "motion_normalized_42d_edge_representation",
            "public_validation_target_pool",
        ],
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
            "edge_feature_mode": "relative_motion_residual_dct8_v1",
        },
        "continuation_gates": {
            "common_threshold_incremental_asr_min": 0.6736,
            "moving_ge_0p5_incremental_asr_min": 0.45,
            "adjacent_negative_fp_incremental_max": 0.015,
            "clean_pair_auc_min": 0.9849,
            "decision": "continue_only_if_all_four_gates_pass",
        },
        "schedule_manifest": {
            "path": SCHEDULE_MANIFEST.as_posix(),
            "sha256": sha256_file(ROOT / SCHEDULE_MANIFEST),
            "metadata_path": SCHEDULE_METADATA.as_posix(),
            "metadata_sha256": sha256_file(ROOT / SCHEDULE_METADATA),
            "verification_path": SCHEDULE_VERIFICATION.as_posix(),
            "verification_sha256": sha256_file(ROOT / SCHEDULE_VERIFICATION),
            "counts": metadata["trigger_schedule_counts"],
        },
        "graph_contract": {
            "manifest_path": GRAPH_MANIFEST.as_posix(),
            "manifest_sha256": sha256_file(ROOT / GRAPH_MANIFEST),
            "metadata_path": GRAPH_CONTRACT.as_posix(),
            "metadata_sha256": sha256_file(ROOT / GRAPH_CONTRACT),
        },
        "clean_reference": {
            "result_path": CLEAN_RESULT.as_posix(),
            "result_sha256": sha256_file(ROOT / CLEAN_RESULT),
            "checkpoint_path": CLEAN_CHECKPOINT.as_posix(),
            "checkpoint_sha256": sha256_file(ROOT / CLEAN_CHECKPOINT),
            "validation_pair_threshold": clean["val_pair_metrics"]["threshold"],
        },
        "v6_2_parent_release": {
            "path": V6_2_RELEASE.as_posix(),
            "sha256": sha256_file(ROOT / V6_2_RELEASE),
            "anchor_path": V6_2_RELEASE_ANCHOR.as_posix(),
            "anchor_sha256": sha256_file(ROOT / V6_2_RELEASE_ANCHOR),
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
    release_hash = sha256_file(release_path)
    _atomic_text(
        f"{release_hash}  {release_path.relative_to(ROOT).as_posix()}\n",
        anchor_path,
    )
    print(json.dumps({"release": str(release_path), "sha256": release_hash}, indent=2))


if __name__ == "__main__":
    main()
