#!/usr/bin/env python3
"""Independently recompute and freeze six formal v6.0 test evaluations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jcas import PROJECT_ROOT
from jcas.core.strict_shadow_folds import _verify_sha_manifest
from jcas.release import metrics_integrity as integrity


ROOT = PROJECT_ROOT
SEEDS = (20260621, 20260622, 20260623)
PACKAGE_VERSION = "v6.0_evaluation_integrity"
EVALUATOR_VERSION = "v6.0_pretest_release_metric_evidence"
METHOD_VERSION = "v6_fixed_symmetric_same_pair_training"
EXPECTED_VICTIM_ROLE = "v6_fixed_symmetric_victim"
EXPECTED_RELEASE_STATUS = "pre_frozen_before_v6_formal_test"
DEFAULT_PRETEST_RELEASE = Path(
    "record/v6/contracts/v6_pretest_release_20260814.json"
)
DEFAULT_TEST_DIR = Path("record/v6/test")
DEFAULT_OUTPUT_DIR = Path("record/v6/contracts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently verify v6.0 formal test evidence"
    )
    parser.add_argument("--pretest-release", default=str(DEFAULT_PRETEST_RELEASE))
    parser.add_argument("--pretest-release-sha256", required=True)
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def verify_pretest_release(path: Path, expected_sha256: str) -> dict[str, Any]:
    release_path = path.resolve()
    release_relative = release_path.relative_to(ROOT)
    if integrity.sha256_file(release_path) != str(expected_sha256).lower():
        raise RuntimeError("v6 pretest release does not match external trust anchor")
    release = integrity.load_json(release_path)
    if release.get("version") != PACKAGE_VERSION:
        raise ValueError("unexpected v6 pretest release version")
    if release.get("status") != EXPECTED_RELEASE_STATUS:
        raise ValueError("unexpected v6 pretest release status")
    if int(release.get("test_outputs_present_before_freeze", -1)) != 0:
        raise RuntimeError("v6 pretest release was created after test output")

    code_path, code_sha = integrity.verify_record(
        release.get("code_asset_manifest"), label="code asset manifest"
    )
    source_path, source_sha = integrity.verify_record(
        release.get("source_archive"), label="source archive"
    )
    code_entries = integrity.parse_sha_manifest(ROOT / code_path)
    verified_entries = _verify_sha_manifest(
        ROOT / code_path,
        source_archive=ROOT / source_path,
    )
    if verified_entries != len(code_entries):
        raise RuntimeError("v6 source archive does not cover its code manifest")
    evaluator_record = release.get("evaluator")
    finalizer_record = release.get("finalizer")
    if not isinstance(evaluator_record, dict) or not isinstance(finalizer_record, dict):
        raise ValueError("v6 release is missing evaluator or finalizer records")
    evaluator_path = integrity.repository_relative(
        str(evaluator_record.get("path", "")), label="evaluator"
    )
    finalizer_path = integrity.repository_relative(
        str(finalizer_record.get("path", "")), label="finalizer"
    )
    evaluator_sha = str(evaluator_record.get("sha256", "")).lower()
    finalizer_sha = str(finalizer_record.get("sha256", "")).lower()
    if evaluator_path.as_posix() != "eval_blackbox_poison.py":
        raise RuntimeError("v6 evaluator path is not canonical")
    expected_finalizer = Path("finalize_v6.py")
    if finalizer_path != expected_finalizer:
        raise RuntimeError("running finalizer is not the frozen v6 finalizer")
    if code_entries.get(evaluator_path.as_posix()) != evaluator_sha:
        raise RuntimeError("v6 evaluator is absent from code manifest")
    if code_entries.get(finalizer_path.as_posix()) != finalizer_sha:
        raise RuntimeError("v6 finalizer is absent from code manifest")

    formal_path, formal_sha = integrity.verify_record(
        release.get("formal_contract"), label="formal contract"
    )
    frozen_path, frozen_sha = integrity.verify_record(
        release.get("frozen_asset_manifest"), label="frozen asset manifest"
    )
    environment_path, environment_sha = integrity.verify_record(
        release.get("environment"), label="environment summary"
    )
    formal = integrity.load_json(ROOT / formal_path)
    if formal.get("version") != METHOD_VERSION:
        raise ValueError("v6 release is bound to another method")
    declared = formal.get("asset_hash_manifest", {})
    if declared.get("sha256") != frozen_sha:
        raise RuntimeError("v6 contract and asset manifest disagree")
    if integrity.repository_relative(
        str(declared.get("path", "")), label="frozen asset manifest"
    ) != frozen_path:
        raise RuntimeError("v6 contract records another asset manifest")
    return {
        "path": release_relative,
        "sha256": str(expected_sha256).lower(),
        "code_manifest_path": code_path,
        "code_manifest_sha256": code_sha,
        "evaluator_path": evaluator_path,
        "evaluator_sha256": evaluator_sha,
        "finalizer_path": finalizer_path,
        "finalizer_sha256": finalizer_sha,
        "formal_contract_path": formal_path,
        "formal_contract_sha256": formal_sha,
        "frozen_asset_manifest_path": frozen_path,
        "frozen_asset_manifest_sha256": frozen_sha,
        "source_archive_path": source_path,
        "source_archive_sha256": source_sha,
        "environment_path": environment_path,
        "environment_sha256": environment_sha,
    }


def _expected_target_rows(test_dir: Path) -> int:
    rows: set[int] = set()
    for seed in SEEDS:
        for kind in ("reference", "victim"):
            result = integrity.load_json(test_dir / f"{kind}_seed{seed}.json")
            target = result.get("target_manifest")
            if not isinstance(target, dict):
                raise ValueError("formal v6 result lacks target manifest")
            rows.add(int(target.get("rows", -1)))
    if len(rows) != 1 or next(iter(rows)) <= 0:
        raise ValueError("six v6 results do not declare one non-empty target count")
    return next(iter(rows))


def main() -> None:
    args = parse_args()
    release_audit = verify_pretest_release(
        Path(args.pretest_release), str(args.pretest_release_sha256)
    )
    test_dir = (
        ROOT / integrity.repository_relative(args.test_dir, label="test dir")
    ).resolve()
    output_dir = (
        ROOT / integrity.repository_relative(args.output_dir, label="output dir")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    test_asset_path = output_dir / "v6_final_test_assets_20260814.sha256"
    metadata_path = output_dir / "v6_final_test_20260814.metadata.json"
    final_release_path = output_dir / "v6_final_release_20260814.sha256"
    if any(path.exists() for path in (test_asset_path, metadata_path, final_release_path)):
        raise FileExistsError("v6 final outputs already exist")

    expected_rows = _expected_target_rows(test_dir)
    integrity.EVALUATOR_VERSION = EVALUATOR_VERSION
    integrity.METHOD_VERSION = METHOD_VERSION
    integrity.EXPECTED_TARGET_ROWS = expected_rows

    verified: dict[tuple[int, str], dict[str, Any]] = {}
    test_assets: set[Path] = set()
    target_hashes: set[str] = set()
    for seed in SEEDS:
        for kind, role in (
            ("reference", "clean_reference"),
            ("victim", EXPECTED_VICTIM_ROLE),
        ):
            record = integrity.verify_result(
                test_dir / f"{kind}_seed{seed}.json",
                seed=seed,
                expected_role=role,
                release_audit=release_audit,
            )
            verified[(seed, kind)] = record
            test_assets.update(record["assets"])
            target_hashes.add(record["target_sha256"])
    if len(target_hashes) != 1:
        raise ValueError("six v6 evaluations did not use one target pool")

    lines = [
        f"{integrity.sha256_file(ROOT / path)}  {path.as_posix()}"
        for path in sorted(test_assets, key=lambda item: item.as_posix())
    ]
    integrity.atomic_text("\n".join(lines) + "\n", test_asset_path)

    metric_keys = (
        "clean_activation_rate",
        "absolute_asr",
        "incremental_flip_rate_all_targets",
        "conditional_flip_rate",
        "target_probability_delta_mean",
        "nonincident_negative_fp_incremental",
        "adjacent_negative_fp_incremental",
        "adjacent_positive_suppression_incremental",
    )
    summary_values = {key: [] for key in metric_keys}
    pair_values = {key: [] for key in ("auc", "pr_auc", "f1")}
    reference_incremental: list[float] = []
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        reference = verified[(seed, "reference")]
        victim = verified[(seed, "victim")]
        reference_attack = {
            **reference["target_metrics"]["common"],
            **reference["prediction_metrics"]["collateral_common"],
        }
        victim_attack = {
            **victim["target_metrics"]["common"],
            **victim["prediction_metrics"]["collateral_common"],
        }
        victim_pair = victim["prediction_metrics"]["pair_common"]
        reference_value = float(
            reference_attack["incremental_flip_rate_all_targets"]
        )
        reference_incremental.append(reference_value)
        for key in metric_keys:
            summary_values[key].append(float(victim_attack[key]))
        for key in pair_values:
            pair_values[key].append(float(victim_pair[key]))
        per_seed[str(seed)] = {
            "common_threshold": float(victim_pair["threshold"]),
            "reference_incremental_flip_rate": reference_value,
            **{f"victim_{key}": float(victim_attack[key]) for key in metric_keys},
            **{f"victim_pair_{key}": float(victim_pair[key]) for key in pair_values},
            "reference_adjusted_incremental_effect": float(
                victim_attack["incremental_flip_rate_all_targets"]
                - reference_value
            ),
        }
    summary = {
        "reference_incremental_flip_rate": integrity.summarize(
            reference_incremental
        ),
        **{
            f"victim_{key}": integrity.summarize(values)
            for key, values in summary_values.items()
        },
        **{
            f"victim_pair_{key}": integrity.summarize(values)
            for key, values in pair_values.items()
        },
        "reference_adjusted_incremental_effect": integrity.summarize(
            [
                per_seed[str(seed)]["reference_adjusted_incremental_effect"]
                for seed in SEEDS
            ]
        ),
    }
    metadata = {
        "scope": "offline_authorized_av2_model_robustness",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "v6_formal_test_independently_recomputed",
        "version": PACKAGE_VERSION,
        "interpretation": "formal evaluation after validation-only v6 selection",
        "pretest_release": {
            "path": release_audit["path"].as_posix(),
            "sha256": release_audit["sha256"],
        },
        "evaluator": {
            "path": release_audit["evaluator_path"].as_posix(),
            "sha256": release_audit["evaluator_sha256"],
        },
        "test_asset_manifest": {
            "path": test_asset_path.relative_to(ROOT).as_posix(),
            "sha256": integrity.sha256_file(test_asset_path),
            "entries": len(test_assets),
        },
        "semantic_recomputation": {
            "verified_runs": 6,
            "target_metrics": True,
            "directed_clean_metrics": True,
            "unordered_pair_metrics": True,
            "collateral_2x2_transitions": True,
            "per_city_target_metrics": True,
            "verification_errors": 0,
        },
        "target_pool": {
            "rows": expected_rows,
            "common_sha256": next(iter(target_hashes)),
        },
        "per_seed": per_seed,
        "three_seed_summary": summary,
    }
    integrity.atomic_json(metadata, metadata_path)

    anchor_path = Path(args.pretest_release).with_suffix(".sha256")
    release_assets = {
        Path(__file__).resolve().relative_to(ROOT),
        release_audit["evaluator_path"],
        release_audit["path"],
        anchor_path.resolve().relative_to(ROOT),
        release_audit["code_manifest_path"],
        release_audit["source_archive_path"],
        test_asset_path.relative_to(ROOT),
        metadata_path.relative_to(ROOT),
    }
    release_lines = [
        f"{integrity.sha256_file(ROOT / path)}  {path.as_posix()}"
        for path in sorted(release_assets, key=lambda item: item.as_posix())
    ]
    integrity.atomic_text("\n".join(release_lines) + "\n", final_release_path)
    print("formal v6 results independently verified: 6/6")
    print(f"target rows: {expected_rows}")
    print(f"metadata: {metadata_path.relative_to(ROOT)}")
    print(f"release: {final_release_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
