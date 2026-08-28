#!/usr/bin/env python3
"""Freeze the v6.2-D train/validation-only motion-prioritized pilot.

This release is deliberately not a formal-test contract.  It binds the
current implementation, the v6.0 parent manifest, the generated v6.2-D
manifest, the graph contract, the reused clean reference, the single pilot
seed, and the predeclared continuation gates before victim training.
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


ROOT = PROJECT_ROOT
DATE_TAG = "20260827"
VERSION = "v6_2_data_only_motion_prioritized_pair_pilot_v1"
DEFAULT_OUTPUT_DIR = Path("record/v6_2/contracts/pretraining")
GRAPH_MANIFEST = Path("graphs/av2mf_graph_v4/manifest_grouped_v5.csv")
GRAPH_CONTRACT = Path(str(GRAPH_MANIFEST) + ".metadata.json")
POISON_MANIFEST = Path("record/v6/poison_rate005_fixed_symmetric.csv")
POISON_METADATA = Path(str(POISON_MANIFEST) + ".metadata.json")
V6_2_MANIFEST = Path("record/v6_2/poison_rate005_motion_prioritized.csv")
V6_2_METADATA = Path(str(V6_2_MANIFEST) + ".metadata.json")
V6_1_RELEASE = Path(
    "record/v6_1/contracts/v6_1a_pretraining_r2/"
    "v6_1a_pretraining_release_20260827.json"
)
V6_1_RELEASE_ANCHOR = V6_1_RELEASE.with_suffix(".sha256")
CLEAN_RESULT = Path(
    "record/v6_1/clean_seed20260621/"
    "genconv_strict_seed20260621_20260827_174820/result.json"
)
CLEAN_CHECKPOINT = CLEAN_RESULT.with_name("best_model.pt")
PARENT_CONTRACT = Path("record/v6/contracts/v6_freeze_20260814.metadata.json")
PARENT_ASSETS = Path("record/v6/contracts/v6_frozen_assets_20260814.sha256")
PARENT_SOURCE = Path("record/v6/contracts/v6_source_snapshot_20260814.tar.gz")

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path("pyproject.toml"),
    Path("requirements.txt"),
    Path("docs/V6_RUNBOOK.md"),
    Path("docs/V6_1_MOTION_NORMALIZED_PILOT.md"),
    Path("docs/V6_2_MOTION_PRIORITIZED_PILOT.md"),
    Path("jcas/__init__.py"),
    Path("jcas/core/__init__.py"),
    Path("jcas/core/graph_splits.py"),
    Path("jcas/core/graph_splits_v5.py"),
    Path("jcas/core/models.py"),
    Path("jcas/core/motion_normalized_features.py"),
    Path("jcas/core/poison.py"),
    Path("jcas/core/risk_labels.py"),
    Path("jcas/core/shadow_folds.py"),
    Path("jcas/core/strict_shadow_folds.py"),
    Path("jcas/core/trajectory_trigger.py"),
    Path("jcas/workflows/__init__.py"),
    Path("jcas/workflows/evaluator.py"),
    Path("jcas/workflows/fixed_symmetric_manifest.py"),
    Path("jcas/workflows/graph_builder.py"),
    Path("jcas/workflows/motion_prioritized_manifest.py"),
    Path("jcas/workflows/poison_manifest.py"),
    Path("jcas/workflows/trainer.py"),
    Path("jcas/release/__init__.py"),
    Path("jcas/release/finalize_v6.py"),
    Path("jcas/release/freeze_code_release.py"),
    Path("jcas/release/freeze_v6.py"),
    Path("jcas/release/freeze_v6_2_pilot.py"),
    Path("jcas/release/metrics_integrity.py"),
    Path("jcas/release/source_integrity.py"),
    Path("tests/test_blackbox_pipeline.py"),
    Path("tests/test_v6_2_motion_prioritized.py"),
    Path("tests/test_v6_fixed_symmetric_manifest.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the v6.2-D validation-only pilot before training"
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


def _parent_allowlist(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(
                f"invalid parent asset line {line_number}: {line!r}"
            )
        digest, recorded_path = parts
        entries[recorded_path] = digest.lower()
    return entries


def _verify_parent_asset(path: Path, allowlist: dict[str, str]) -> str:
    key = path.as_posix()
    expected = allowlist.get(key)
    if expected is None:
        raise RuntimeError(f"parent v6 contract does not bind {key}")
    actual = sha256_file(ROOT / path)
    if actual != expected:
        raise RuntimeError(f"parent-bound asset changed: {key}")
    return actual


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _verify_pilot_assets() -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = _load_json(ROOT / V6_2_METADATA)
    if metadata.get("experiment") != (
        "v6_2_data_only_motion_prioritized_pair_v1"
    ):
        raise RuntimeError("unexpected v6.2 manifest experiment")
    if metadata.get("manifest_sha256") != sha256_file(ROOT / V6_2_MANIFEST):
        raise RuntimeError("v6.2 manifest SHA-256 binding failed")
    if int(metadata.get("poisoned_scenarios", -1)) != 5776:
        raise RuntimeError("v6.2 manifest must contain 5,776 scenarios")
    if int(metadata.get("eligible_scenarios", -1)) != 115526:
        raise RuntimeError("v6.2 eligible scenario count changed")
    if metadata.get("scenario_ids_preserved") is not True:
        raise RuntimeError("v6.2 did not preserve the v6 scenario set")
    if metadata.get("ordinary_bce_unchanged") is not True:
        raise RuntimeError("v6.2 manifest no longer has an ordinary-BCE contract")
    if metadata.get("original_validation_used") is not False:
        raise RuntimeError("v6.2 manifest selection accessed validation")
    if metadata.get("original_test_used") is not False:
        raise RuntimeError("v6.2 manifest selection accessed test")
    if metadata.get("model_outputs_used") is not False:
        raise RuntimeError("v6.2 manifest selection used model outputs")
    selected_counts = metadata.get("selected_motion_counts", {})
    if sum(int(value) for value in selected_counts.values()) != 5776:
        raise RuntimeError("v6.2 motion counts do not sum to 5,776")

    clean = _load_json(ROOT / CLEAN_RESULT)
    config = clean.get("config", {})
    if config.get("poison_manifest") is not None:
        raise RuntimeError("v6.2 clean reference is not clean")
    if config.get("evaluate_test") is not False:
        raise RuntimeError("v6.2 clean reference accessed test")
    if config.get("edge_feature_mode") != MOTION_NORMALIZED_EDGE_FEATURE_MODE:
        raise RuntimeError("v6.2 clean reference uses the wrong representation")
    if clean.get("test_metrics") is not None:
        raise RuntimeError("v6.2 clean reference contains test metrics")
    return metadata, clean


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / Path(args.output_dir)).resolve()
    output_dir.relative_to(ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_manifest = output_dir / f"v6_2_code_assets_{DATE_TAG}.sha256"
    source_archive = output_dir / f"v6_2_source_snapshot_{DATE_TAG}.tar.gz"
    environment_path = output_dir / f"v6_2_environment_{DATE_TAG}.json"
    release_path = output_dir / f"v6_2_pretraining_release_{DATE_TAG}.json"
    anchor_path = output_dir / f"v6_2_pretraining_release_{DATE_TAG}.sha256"
    outputs = (
        code_manifest,
        source_archive,
        environment_path,
        release_path,
        anchor_path,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("v6.2 pretraining release already exists")

    required = SOURCE_PATHS + (
        GRAPH_MANIFEST,
        GRAPH_CONTRACT,
        POISON_MANIFEST,
        POISON_METADATA,
        V6_2_MANIFEST,
        V6_2_METADATA,
        V6_1_RELEASE,
        V6_1_RELEASE_ANCHOR,
        CLEAN_RESULT,
        CLEAN_CHECKPOINT,
        PARENT_CONTRACT,
        PARENT_ASSETS,
        PARENT_SOURCE,
    )
    missing = [path.as_posix() for path in required if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"v6.2 freeze inputs are missing: {missing}")

    parent_allowlist = _parent_allowlist(ROOT / PARENT_ASSETS)
    parent_bound_assets = {
        path.as_posix(): _verify_parent_asset(path, parent_allowlist)
        for path in (
            GRAPH_MANIFEST,
            GRAPH_CONTRACT,
            POISON_MANIFEST,
            POISON_METADATA,
        )
    }
    manifest_metadata, clean_result = _verify_pilot_assets()

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
        "status": "frozen_before_v6_2_victim_training",
        "development_only": True,
        "formal_test_eligible": False,
        "test_access_authorized": False,
        "prior_validation_diagnostics_motivated_new_pilot": True,
        "current_manifest_selection_used_validation": False,
        "original_test_used": False,
        "pilot_seed": 20260621,
        "method_change": (
            "within_the_frozen_v6_poisoned_scenarios_prefer_a_data_eligible_"
            "moving_pair_then_use_a_stable_hash"
        ),
        "threat_model": "data_only_poisoning_with_ordinary_bce",
        "unchanged_components": [
            "graph_generation_and_grouped_v5_split",
            "dynamic_risk_labels_d0_5_tau_1_decel_4",
            "v6_0_poisoned_scenario_set_and_five_percent_rate",
            "fixed_symmetric_biend_alpha_0p5_trajectory_transform",
            "genconv_architecture_and_ordinary_bce_training",
            "public_validation_target_pool_definition",
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
            "common_threshold_incremental_asr_min": 0.6736,
            "moving_ge_0p5_incremental_asr_min": 0.45,
            "adjacent_negative_fp_incremental_max": 0.015,
            "clean_pair_auc_min": 0.9849,
            "decision": (
                "run_two_additional_seeds_only_if_all_four_gates_pass"
            ),
        },
        "validation_protocol": {
            "split": "val",
            "target_seed": 20260621,
            "orientation_policy": "lower_destination_mean_speed_v1",
            "allocation_policy": "fixed_symmetric_biend_v1",
            "motion_regime_boundary_mps": 0.5,
            "common_threshold_source": (
                "same_seed_v6_1a_clean_validation_pair_threshold"
            ),
        },
        "v6_2_manifest": {
            "path": V6_2_MANIFEST.as_posix(),
            "sha256": sha256_file(ROOT / V6_2_MANIFEST),
            "metadata_path": V6_2_METADATA.as_posix(),
            "metadata_sha256": sha256_file(ROOT / V6_2_METADATA),
            "selected_motion_counts": manifest_metadata[
                "selected_motion_counts"
            ],
            "changed_pairs": manifest_metadata["changed_pairs"],
        },
        "clean_reference": {
            "result_path": CLEAN_RESULT.as_posix(),
            "result_sha256": sha256_file(ROOT / CLEAN_RESULT),
            "checkpoint_path": CLEAN_CHECKPOINT.as_posix(),
            "checkpoint_sha256": sha256_file(ROOT / CLEAN_CHECKPOINT),
            "validation_pair_threshold": clean_result["val_pair_metrics"][
                "threshold"
            ],
        },
        "v6_1_parent_representation_release": {
            "path": V6_1_RELEASE.as_posix(),
            "sha256": sha256_file(ROOT / V6_1_RELEASE),
            "anchor_path": V6_1_RELEASE_ANCHOR.as_posix(),
            "anchor_sha256": sha256_file(ROOT / V6_1_RELEASE_ANCHOR),
        },
        "parent_v6_release": {
            "contract": {
                "path": PARENT_CONTRACT.as_posix(),
                "sha256": sha256_file(ROOT / PARENT_CONTRACT),
            },
            "asset_manifest": {
                "path": PARENT_ASSETS.as_posix(),
                "sha256": sha256_file(ROOT / PARENT_ASSETS),
            },
            "source_snapshot": {
                "path": PARENT_SOURCE.as_posix(),
                "sha256": sha256_file(ROOT / PARENT_SOURCE),
            },
        },
        "parent_bound_assets": parent_bound_assets,
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
    print(f"v6.2 release: {release_path.relative_to(ROOT)}")
    print(f"external trust anchor SHA-256: {release_sha}")
    print(f"code assets: {len(code_lines)}")


if __name__ == "__main__":
    main()
