#!/usr/bin/env python3
"""Freeze validation-selected v6.0 assets before any formal test access."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from jcas import PROJECT_ROOT
from jcas.core.poison import (
    ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
    ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
    sha256_file,
)


ROOT = PROJECT_ROOT
FREEZE_VERSION = "v6_fixed_symmetric_same_pair_training"
FREEZE_STATUS = "frozen_v6_validation_selected"
SELECTION_OBJECTIVE = "v5_gradient_pair_selection_then_fixed_symmetric_alpha_v6_0"
SEEDS = (20260621, 20260622, 20260623)
DATE_TAG = "20260814"

GRAPH_MANIFEST = Path("graphs/av2mf_graph_v4/manifest_grouped_v5.csv")
GRAPH_CONTRACT = Path(str(GRAPH_MANIFEST) + ".metadata.json")
GROUP_METADATA = Path("record/v5/duplicate_scene_check_overlap_v5.parquet")
GROUP_CONTRACT = Path(str(GROUP_METADATA) + ".metadata.json")
POISON_MANIFEST = Path("record/v6/poison_rate005_fixed_symmetric.csv")
POISON_METADATA = Path(str(POISON_MANIFEST) + ".metadata.json")
POISON_AUDIT = Path("record/v6/poison_rate005_fixed_symmetric.fixed_alpha_audit.csv")

CLEAN_RESULTS = {
    20260621: Path(
        "record/v5/clean_seed20260621/"
        "genconv_strict_seed20260621_20260812_123153/result.json"
    ),
    20260622: Path(
        "record/v5/clean_seed20260622/"
        "genconv_strict_seed20260622_20260812_135141/result.json"
    ),
    20260623: Path(
        "record/v5/clean_seed20260623/"
        "genconv_strict_seed20260623_20260812_161134/result.json"
    ),
}
VICTIM_RESULTS = {
    20260621: Path(
        "record/v6/victim_seed20260621/"
        "genconv_strict_seed20260621_20260813_154310/result.json"
    ),
    20260622: Path(
        "record/v6/victim_seed20260622/"
        "genconv_strict_seed20260622_20260813_220605/result.json"
    ),
    20260623: Path(
        "record/v6/victim_seed20260623/"
        "genconv_strict_seed20260623_20260813_231749/result.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze v6.0 after three-seed validation selection"
    )
    parser.add_argument("--output-dir", default="record/v6/contracts")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    absolute = ROOT / path if not path.is_absolute() else path
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    with absolute.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _relative(path: Path | str) -> Path:
    requested = Path(str(path))
    resolved = requested.resolve() if requested.is_absolute() else (ROOT / requested).resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"freeze asset escapes repository: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return relative


def _atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", path)


def _checkpoint_from_result(path: Path, result: dict[str, Any]) -> Path:
    record = result.get("checkpoint")
    if not isinstance(record, dict):
        raise ValueError(f"result has no checkpoint record: {path}")
    recorded = Path(str(record.get("path", "")))
    candidates = (
        _relative(recorded),
        _relative(path.parent / recorded.name),
    )
    checkpoint = candidates[0] if (ROOT / candidates[0]).is_file() else candidates[1]
    if sha256_file(ROOT / checkpoint) != str(record.get("sha256", "")):
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {checkpoint}")
    return checkpoint


def _training_result(
    path: Path,
    *,
    seed: int,
    clean: bool,
    poison_sha: str,
    poison_metadata_sha: str,
) -> tuple[dict[str, Any], Path]:
    result = _load_json(path)
    config = result.get("config")
    if not isinstance(config, dict) or int(config.get("seed", -1)) != seed:
        raise ValueError(f"training seed mismatch: {path}")
    expected = {
        "epochs": 50,
        "checkpoint_metric": "val_loss",
        "require_strict_label": True,
        "evaluate_test": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"training protocol mismatch {key}: {path}")
    if any(result.get(key) is not None for key in ("test_metrics", "test_pair_metrics", "test_stats")):
        raise ValueError(f"training result accessed test: {path}")
    if str(result.get("graph_manifest", {}).get("sha256", "")) != sha256_file(
        ROOT / GRAPH_MANIFEST
    ):
        raise RuntimeError(f"training graph manifest mismatch: {path}")
    poison = result.get("poison_manifest")
    if clean:
        if poison is not None:
            raise ValueError(f"clean result has a poison manifest: {path}")
    else:
        if not isinstance(poison, dict):
            raise ValueError(f"victim result has no poison manifest: {path}")
        required = {
            "sha256": poison_sha,
            "metadata_sha256": poison_metadata_sha,
            "rows": 5776,
            "applied_train_rows": 5776,
            "allocation_policy": ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
            "selection_objective": SELECTION_OBJECTIVE,
        }
        for key, value in required.items():
            if poison.get(key) != value:
                raise RuntimeError(f"victim poison binding mismatch {key}: {path}")
        if config.get("require_strict_crossfit_manifest") is not True:
            raise RuntimeError(f"victim did not require strict cross-fit binding: {path}")
        if int(result.get("train_stats", {}).get("poisoned_graphs", -1)) != 5776:
            raise RuntimeError(f"victim did not transform all selected graphs: {path}")
        if int(result.get("val_stats", {}).get("poisoned_graphs", -1)) != 0:
            raise RuntimeError(f"victim transformed validation graphs: {path}")
    return result, _checkpoint_from_result(path, result)


def _result_asset(path: Path, checkpoint: Path) -> dict[str, dict[str, str]]:
    return {
        "result": {"path": path.as_posix(), "sha256": sha256_file(ROOT / path)},
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": sha256_file(ROOT / checkpoint),
        },
    }


def _validation_assets(
    path: Path,
    *,
    expected_target_hash: str | None,
    expected_evaluator_hash: str | None,
) -> tuple[dict[str, Any], set[Path]]:
    result = _load_json(path)
    if result.get("split") != "val" or result.get("evaluation_complete") is not True:
        raise RuntimeError(f"validation result is incomplete: {path}")
    if result.get("formal_complete") is not False:
        raise RuntimeError(f"validation result claims formal completion: {path}")
    target = result.get("target_manifest")
    scores = result.get("target_scores")
    evaluator = result.get("evaluator")
    if not isinstance(target, dict) or not isinstance(scores, dict) or not isinstance(evaluator, dict):
        raise ValueError(f"validation result lacks frozen evidence: {path}")
    if int(target.get("rows", -1)) != 25515 or int(scores.get("rows", -1)) != 25515:
        raise RuntimeError(f"validation target count mismatch: {path}")
    target_path = _relative(str(target.get("path", "")))
    score_path = _relative(str(scores.get("path", "")))
    target_hash = sha256_file(ROOT / target_path)
    if target_hash != str(target.get("sha256", "")):
        raise RuntimeError(f"validation target SHA mismatch: {path}")
    if expected_target_hash is not None and target_hash != expected_target_hash:
        raise RuntimeError("validation target pools are not identical")
    if sha256_file(ROOT / score_path) != str(scores.get("sha256", "")):
        raise RuntimeError(f"validation score SHA mismatch: {path}")
    evaluator_hash = str(evaluator.get("sha256", ""))
    if expected_evaluator_hash is not None and evaluator_hash != expected_evaluator_hash:
        raise RuntimeError("validation evaluator hashes are not identical")
    if result.get("target_sampling", {}).get("allocation_policy") != (
        ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
    ):
        raise RuntimeError(f"validation allocation policy mismatch: {path}")
    assets = {path, target_path, score_path}
    evidence = result.get("metric_evidence")
    if not isinstance(evidence, dict) or evidence.get("complete") is not True:
        raise RuntimeError(f"validation metric evidence is incomplete: {path}")
    for key in ("clean_edge_scores", "clean_pair_scores", "collateral_transitions"):
        record = evidence.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"validation evidence lacks {key}: {path}")
        evidence_path = _relative(str(record.get("path", "")))
        if sha256_file(ROOT / evidence_path) != str(record.get("sha256", "")):
            raise RuntimeError(f"validation evidence SHA mismatch {key}: {path}")
        assets.add(evidence_path)
    return result, assets


def _collect_strict_provenance(metadata: dict[str, Any]) -> set[Path]:
    assets: set[Path] = set()
    for key in (
        "source_manifest_path",
        "source_metadata_path",
        "source_candidate_scores_path",
        "base_manifest_path",
        "fixed_alpha_audit_path",
    ):
        value = metadata.get(key)
        if value:
            assets.add(_relative(str(value)))
    shadow = metadata.get("shadow_fold_manifest")
    if isinstance(shadow, dict):
        for key in ("path", "metadata_path"):
            assets.add(_relative(str(shadow[key])))
    for surrogate in metadata.get("surrogates", []):
        if not isinstance(surrogate, dict):
            raise ValueError("strict surrogate record is malformed")
        for key in ("result_path", "checkpoint_path"):
            assets.add(_relative(str(surrogate[key])))
        protocol = surrogate.get("shadow_protocol")
        release = surrogate.get("strict_crossfit_release")
        if not isinstance(protocol, dict) or not isinstance(release, dict):
            raise ValueError("strict surrogate provenance is incomplete")
        for key in (
            "strict_contract_path",
            "surrogate_fit_manifest_path",
            "surrogate_score_manifest_path",
        ):
            assets.add(_relative(str(protocol[key])))
        for key in ("path",):
            assets.add(_relative(str(release[key])))
        for nested in (
            "code_asset_manifest",
            "strict_crossfit_asset_manifest",
            "source_archive",
            "base_poison_manifest",
        ):
            record = release.get(nested)
            if not isinstance(record, dict):
                raise ValueError(f"strict release lacks {nested}")
            for key in ("path", "metadata_path"):
                if record.get(key):
                    assets.add(_relative(str(record[key])))
    return assets


def _validation_record(path: Path) -> dict[str, str]:
    return {"path": path.as_posix(), "sha256": sha256_file(ROOT / path)}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.is_absolute():
        output_dir = output_dir.resolve().relative_to(ROOT)
    contract_path = output_dir / f"v6_freeze_{DATE_TAG}.metadata.json"
    manifest_path = output_dir / f"v6_frozen_assets_{DATE_TAG}.sha256"
    if contract_path.exists() or manifest_path.exists():
        raise FileExistsError("v6 freeze outputs already exist")
    test_dir = ROOT / "record/v6/test"
    if test_dir.is_dir() and any(test_dir.iterdir()):
        raise RuntimeError("v6 test outputs already exist; freeze must precede test")

    graph_metadata = _load_json(GRAPH_CONTRACT)
    verification = graph_metadata.get("graph_verification")
    split_audit = graph_metadata.get("split_audit")
    if not isinstance(verification, dict) or not isinstance(split_audit, dict):
        raise RuntimeError("graph contract lacks verification or split audit")
    required_verification = {
        "built_graphs_verified": 188439,
        "source_contract_graphs": 188439,
        "verification_errors": 0,
        "build_config_variants": 1,
    }
    for key, value in required_verification.items():
        if int(verification.get(key, -1)) != value:
            raise RuntimeError(f"graph verification mismatch: {key}")
    leakage = split_audit.get("leakage_violations")
    if not isinstance(leakage, dict) or any(int(value) != 0 for value in leakage.values()):
        raise RuntimeError("grouped_v5 leakage audit failed")

    poison_metadata = _load_json(POISON_METADATA)
    poison_sha = sha256_file(ROOT / POISON_MANIFEST)
    poison_metadata_sha = sha256_file(ROOT / POISON_METADATA)
    required_poison = {
        "manifest_sha256": poison_sha,
        "training_manifest_eligible": True,
        "development_only": True,
        "formal_test_eligible": False,
        "split": "train",
        "poisoned_scenarios": 5776,
        "selection_objective": SELECTION_OBJECTIVE,
        "allocation_policy": ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
        "strict_crossfit_required": True,
        "victim_queries": 0,
        "original_validation_used": False,
        "original_test_used": False,
    }
    for key, value in required_poison.items():
        if poison_metadata.get(key) != value:
            raise RuntimeError(f"v6 poison metadata mismatch: {key}")
    summary = poison_metadata.get("fixed_alpha_summary")
    if not isinstance(summary, dict) or int(
        summary.get("same_pair_preserved_rows", -1)
    ) != 5776 or int(summary.get("outside_source_pair_feature_budget_rows", -1)) != 2110:
        raise RuntimeError("v6 fixed-alpha audit summary changed")

    clean_assets: dict[str, Any] = {}
    victim_assets: dict[str, Any] = {}
    assets: set[Path] = {
        GRAPH_MANIFEST,
        GRAPH_CONTRACT,
        GROUP_METADATA,
        GROUP_CONTRACT,
        POISON_MANIFEST,
        POISON_METADATA,
        POISON_AUDIT,
    }
    assets.update(_collect_strict_provenance(poison_metadata))
    for seed in SEEDS:
        clean_result, clean_checkpoint = _training_result(
            CLEAN_RESULTS[seed],
            seed=seed,
            clean=True,
            poison_sha=poison_sha,
            poison_metadata_sha=poison_metadata_sha,
        )
        victim_result, victim_checkpoint = _training_result(
            VICTIM_RESULTS[seed],
            seed=seed,
            clean=False,
            poison_sha=poison_sha,
            poison_metadata_sha=poison_metadata_sha,
        )
        clean_assets[str(seed)] = _result_asset(CLEAN_RESULTS[seed], clean_checkpoint)
        victim_assets[str(seed)] = _result_asset(VICTIM_RESULTS[seed], victim_checkpoint)
        assets.update(
            {CLEAN_RESULTS[seed], clean_checkpoint, VICTIM_RESULTS[seed], victim_checkpoint}
        )

    target_hash: str | None = None
    evaluator_hash: str | None = None
    validation_records: dict[str, dict[str, Any]] = {
        "clean_reference": {},
        "victim": {},
    }
    baseline_records: dict[str, dict[str, Any]] = {
        "clean_reference": {},
        "victim": {},
    }
    v6_metrics: dict[str, list[float]] = {
        "incremental": [],
        "conditional": [],
        "absolute": [],
        "activation": [],
        "nonincident_fp": [],
        "adjacent_fp": [],
        "adjacent_suppression": [],
        "pair_auc": [],
    }
    baseline_incremental: list[float] = []
    for seed in SEEDS:
        for role in ("reference", "victim"):
            v6_path = Path(f"record/v6/validation/{role}_seed{seed}.json")
            result, evidence_assets = _validation_assets(
                v6_path,
                expected_target_hash=target_hash,
                expected_evaluator_hash=evaluator_hash,
            )
            if target_hash is None:
                target_hash = str(result["target_manifest"]["sha256"])
            if evaluator_hash is None:
                evaluator_hash = str(result["evaluator"]["sha256"])
            assets.update(evidence_assets)
            key = "clean_reference" if role == "reference" else "victim"
            validation_records[key][str(seed)] = _validation_record(v6_path)

            baseline_path = Path(
                f"record/v5_1_1/validation/{role}_seed{seed}.json"
            )
            baseline, baseline_assets = _validation_assets(
                baseline_path,
                expected_target_hash=target_hash,
                expected_evaluator_hash=evaluator_hash,
            )
            assets.update(baseline_assets)
            baseline_records[key][str(seed)] = _validation_record(baseline_path)
            if role == "victim":
                current = result["attack_metrics_common_threshold"]
                old = baseline["attack_metrics_common_threshold"]
                for name, field in (
                    ("incremental", "incremental_flip_rate_all_targets"),
                    ("conditional", "conditional_flip_rate"),
                    ("absolute", "absolute_asr"),
                    ("activation", "clean_activation_rate"),
                    ("nonincident_fp", "nonincident_negative_fp_incremental"),
                    ("adjacent_fp", "adjacent_negative_fp_incremental"),
                    (
                        "adjacent_suppression",
                        "adjacent_positive_suppression_incremental",
                    ),
                ):
                    v6_metrics[name].append(float(current[field]))
                v6_metrics["pair_auc"].append(float(result["clean_pair_metrics"]["auc"]))
                baseline_incremental.append(
                    float(old["incremental_flip_rate_all_targets"])
                )

    assets = {_relative(path) for path in assets}
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{sha256_file(ROOT / path)}  {path.as_posix()}"
        for path in sorted(assets, key=lambda item: item.as_posix())
    ]
    _atomic_text("\n".join(lines) + "\n", manifest_path)
    manifest_sha = sha256_file(ROOT / manifest_path)

    frozen_config = {
        "poison_rate": 0.05,
        "orientation_policy": ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
        "allocation_policy": ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
        "training_allocation_policy": ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
        "selection_objective": SELECTION_OBJECTIVE,
        "source_pair_selection_objective": "gradient_influence_v4_2",
        "allocation_grid": [0.5],
        "relative_displacement_m": 0.2,
        "perturb_window": 10,
        "ramp_style": "minimum_jerk",
        "velocity_mode": "residual",
        "target_seed": 20260621,
        "same_pair_single_variable_change": True,
        "outside_v5_pair_feature_budget_rows": 2110,
        "label_mode": "dynamic_risk",
        "risk_base_distance_m": 5.0,
        "risk_reaction_time_s": 1.0,
        "risk_safe_decel_mps2": 4.0,
        "graph_schema_version": 4,
        "build_contract_version": 3,
        "split_strategy": (
            "recording_or_stride1_twoframe_av_overlap_content_group_sha256_v5"
        ),
        "strict_crossfit_protocol": "strict_crossfit_inner_validation_v1",
    }
    split_counts = {
        key: int(value)
        for key, value in split_audit["built_split_counts"].items()
    }
    contract = {
        "scope": "offline_authorized_av2_model_robustness",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        ),
        "status": FREEZE_STATUS,
        "version": FREEZE_VERSION,
        "contract_schema_version": 4,
        "roles": {
            "clean": "clean_reference",
            "victim": "v6_fixed_symmetric_victim",
        },
        "frozen_config": frozen_config,
        "training_seeds": list(SEEDS),
        "asset_hash_manifest": {
            "path": manifest_path.as_posix(),
            "sha256": manifest_sha,
        },
        "graph_manifest": {
            "path": GRAPH_MANIFEST.as_posix(),
            "sha256": sha256_file(ROOT / GRAPH_MANIFEST),
            "contract_path": GRAPH_CONTRACT.as_posix(),
            "contract_sha256": sha256_file(ROOT / GRAPH_CONTRACT),
            "built_graphs_verified": 188439,
            "source_contract_graphs": 188439,
            "source_files_verified": int(verification["source_files_verified"]),
            "verification_errors": 0,
            "leakage_violations": leakage,
            "split_counts": split_counts,
        },
        "group_metadata": {
            "path": GROUP_METADATA.as_posix(),
            "sha256": sha256_file(ROOT / GROUP_METADATA),
            "contract_path": GROUP_CONTRACT.as_posix(),
            "contract_sha256": sha256_file(ROOT / GROUP_CONTRACT),
        },
        "poison_manifest": {
            "path": POISON_MANIFEST.as_posix(),
            "sha256": poison_sha,
            "metadata_path": POISON_METADATA.as_posix(),
            "metadata_sha256": poison_metadata_sha,
            "audit_path": POISON_AUDIT.as_posix(),
            "audit_sha256": sha256_file(ROOT / POISON_AUDIT),
            "rows": 5776,
        },
        "clean_assets": clean_assets,
        "victim_assets": victim_assets,
        "validation_assets": validation_records,
        "baseline_validation_assets": baseline_records,
        "validation_selection": {
            "split": "val",
            "target_seed": 20260621,
            "target_rows": 25515,
            "target_manifest_sha256": target_hash,
            "evaluator_sha256": evaluator_hash,
            "victim_incremental_asr_mean": float(np.mean(v6_metrics["incremental"])),
            "victim_incremental_asr_sample_std": float(
                np.std(v6_metrics["incremental"], ddof=1)
            ),
            "victim_conditional_flip_mean": float(np.mean(v6_metrics["conditional"])),
            "victim_absolute_asr_mean": float(np.mean(v6_metrics["absolute"])),
            "victim_clean_activation_mean": float(np.mean(v6_metrics["activation"])),
            "victim_pair_auc_mean": float(np.mean(v6_metrics["pair_auc"])),
            "victim_nonincident_negative_fp_incremental_mean": float(
                np.mean(v6_metrics["nonincident_fp"])
            ),
            "victim_adjacent_negative_fp_incremental_mean": float(
                np.mean(v6_metrics["adjacent_fp"])
            ),
            "victim_adjacent_positive_suppression_incremental_mean": float(
                np.mean(v6_metrics["adjacent_suppression"])
            ),
            "baseline_v5_1_1_incremental_asr_mean": float(
                np.mean(baseline_incremental)
            ),
            "incremental_asr_improvement_over_v5_1_1": float(
                np.mean(v6_metrics["incremental"])
                - np.mean(baseline_incremental)
            ),
            "method_or_hyperparameters_changed_after_validation": False,
        },
        "protocol": {
            "training_epochs": 50,
            "checkpoint_metric": "val_loss",
            "require_strict_label": True,
            "training_test_access": False,
            "validation_only_method_selection": True,
            "formal_test_evaluated": False,
        },
    }
    _atomic_json(contract, contract_path)
    print(
        json.dumps(
            {
                "contract_path": contract_path.as_posix(),
                "contract_sha256": sha256_file(ROOT / contract_path),
                "asset_manifest_path": manifest_path.as_posix(),
                "asset_manifest_sha256": manifest_sha,
                "asset_rows": len(lines),
                "validation_selection": contract["validation_selection"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
