#!/usr/bin/env python3
"""Independently verify and freeze the three-seed v6.4-S validation pilot.

This is a post-run validation release.  It does not authorize test access and
does not promote the development pilot to a formal test result.  The exact
training/evaluation implementation was already frozen by the v6.4 pretraining
release before the first victim model was trained.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jcas import PROJECT_ROOT
from jcas.core.motion_normalized_features import (
    MOTION_NORMALIZED_EDGE_FEATURE_MODE,
)
from jcas.core.poison import (
    ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
    sha256_file,
)
from jcas.core.strict_shadow_folds import _verify_sha_manifest
from jcas.release import metrics_integrity as integrity
from jcas.workflows.motion_prioritized_manifest import (
    SLOW_MOTION_REGIME,
    SLOW_PRIORITIZED_VERSION,
)


ROOT = PROJECT_ROOT
SEEDS = (20260621, 20260622, 20260623)
DATE_TAG = "20260828"
VERSION = "v6_4_slow_prioritized_pair_three_seed_validation_v1"
PRETRAINING_VERSION = "v6_4_slow_prioritized_pair_validation_pilot_v1"
SELECTION_OBJECTIVE = "prefer_slow_lt_0p5_then_stable_hash_within_scenario_v1"
EXPECTED_TARGET_ROWS = 25515
MOVING_MOTION_REGIME = "moving_ge_0p5"
MOTION_BOUNDARY_MPS = 0.5

DEFAULT_OUTPUT_DIR = Path("record/v6_4/contracts/final_validation")
PRETRAINING_RELEASE = Path(
    "record/v6_4/contracts/pretraining/v6_4_pretraining_release_20260828.json"
)
PRETRAINING_ANCHOR = PRETRAINING_RELEASE.with_suffix(".sha256")
GRAPH_MANIFEST = Path("graphs/av2mf_graph_v4/manifest_grouped_v5.csv")
GRAPH_CONTRACT = Path(str(GRAPH_MANIFEST) + ".metadata.json")
POISON_MANIFEST = Path("record/v6_4/poison_rate005_slow_prioritized.csv")
POISON_METADATA = Path(str(POISON_MANIFEST) + ".metadata.json")

CLEAN_RESULTS = {
    20260621: Path(
        "record/v6_1/clean_seed20260621/"
        "genconv_strict_seed20260621_20260827_174820/result.json"
    ),
    20260622: Path(
        "record/v6_4/clean_seed20260622/"
        "genconv_strict_seed20260622_20260828_091817/result.json"
    ),
    20260623: Path(
        "record/v6_4/clean_seed20260623/"
        "genconv_strict_seed20260623_20260828_143113/result.json"
    ),
}
VICTIM_RESULTS = {
    20260621: Path(
        "record/v6_4/victim_seed20260621/"
        "genconv_strict_seed20260621_20260828_042127/result.json"
    ),
    20260622: Path(
        "record/v6_4/victim_seed20260622/"
        "genconv_strict_seed20260622_20260828_103839/result.json"
    ),
    20260623: Path(
        "record/v6_4/victim_seed20260623/"
        "genconv_strict_seed20260623_20260828_155112/result.json"
    ),
}
VALIDATION_RESULTS = {
    seed: Path(f"record/v6_4/validation/victim_seed{seed}.json")
    for seed in SEEDS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the completed three-seed v6.4-S validation pilot"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _relative(value: Path | str) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"asset escapes repository: {value}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return relative


def _atomic_text(value: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", path)


def _anchor_digest(anchor_path: Path, expected_path: Path) -> str:
    lines = anchor_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1 or "  " not in lines[0]:
        raise ValueError("pretraining anchor must contain exactly one SHA record")
    digest, recorded = lines[0].split("  ", 1)
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("pretraining anchor has an invalid SHA-256")
    if _relative(recorded) != expected_path:
        raise RuntimeError("pretraining anchor points to another release")
    return digest


def _verify_pretraining_release() -> dict[str, Any]:
    release_path = _relative(PRETRAINING_RELEASE)
    anchor_path = _relative(PRETRAINING_ANCHOR)
    expected = _anchor_digest(ROOT / anchor_path, release_path)
    if sha256_file(ROOT / release_path) != expected:
        raise RuntimeError("pretraining release no longer matches its external anchor")
    release = _load_json(ROOT / release_path)
    if release.get("version") != PRETRAINING_VERSION:
        raise ValueError("unexpected v6.4 pretraining release version")
    if release.get("status") != "frozen_before_single_seed_victim_training":
        raise ValueError("unexpected v6.4 pretraining release status")
    if release.get("development_only") is not True:
        raise RuntimeError("v6.4 pretraining release is not development-only")
    if release.get("test_access_authorized") is not False:
        raise RuntimeError("v6.4 pretraining release unexpectedly authorizes test")

    records: dict[str, tuple[Path, str]] = {}
    for key in ("code_asset_manifest", "source_archive", "environment"):
        record = release.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"pretraining release lacks {key}")
        path = _relative(str(record.get("path", "")))
        digest = str(record.get("sha256", ""))
        if sha256_file(ROOT / path) != digest:
            raise RuntimeError(f"pretraining {key} SHA-256 mismatch")
        records[key] = (path, digest)
    verified = _verify_sha_manifest(
        ROOT / records["code_asset_manifest"][0],
        source_archive=ROOT / records["source_archive"][0],
    )
    code_entries = integrity.parse_sha_manifest(
        ROOT / records["code_asset_manifest"][0]
    )
    if verified != len(code_entries):
        raise RuntimeError("pretraining source archive is incomplete")
    evaluator_sha = code_entries.get("jcas/workflows/evaluator.py")
    trainer_sha = code_entries.get("jcas/workflows/trainer.py")
    if evaluator_sha is None or trainer_sha is None:
        raise RuntimeError("pretraining release does not bind trainer/evaluator")
    return {
        "path": release_path,
        "sha256": expected,
        "anchor_path": anchor_path,
        "anchor_sha256": sha256_file(ROOT / anchor_path),
        "release": release,
        "records": records,
        "evaluator_sha256": evaluator_sha,
        "trainer_sha256": trainer_sha,
    }


def _checkpoint_path(result_path: Path, result: dict[str, Any]) -> Path:
    checkpoint = result.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"training result lacks checkpoint: {result_path}")
    path = _relative(result_path.parent / "best_model.pt")
    if sha256_file(ROOT / path) != str(checkpoint.get("sha256", "")):
        raise RuntimeError(f"training checkpoint SHA-256 mismatch: {result_path}")
    return path


def _verify_training_result(
    result_path: Path,
    *,
    seed: int,
    clean: bool,
) -> tuple[dict[str, Any], Path]:
    result_path = _relative(result_path)
    result = _load_json(ROOT / result_path)
    config = result.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"training result lacks config: {result_path}")
    expected = {
        "seed": seed,
        "model_name": "genconv",
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
        "label_mode": "dynamic_risk",
        "risk_base_distance_m": 5.0,
        "risk_reaction_time_s": 1.0,
        "risk_safe_decel_mps2": 4.0,
        "evaluate_test": False,
        "edge_feature_mode": MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        "experimental_trigger_schedule": None,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"training protocol mismatch {result_path}/{key}: "
                f"{config.get(key)!r} != {value!r}"
            )
    for key in ("max_train_graphs", "max_val_graphs", "max_test_graphs"):
        if config.get(key) is not None:
            raise RuntimeError(f"training used truncation: {result_path}/{key}")
    if any(
        result.get(key) is not None
        for key in ("test_stats", "test_metrics", "test_pair_metrics")
    ):
        raise RuntimeError(f"training result accessed test: {result_path}")
    if len(result.get("history", [])) != 50:
        raise RuntimeError(f"training did not complete 50 epochs: {result_path}")
    graph = result.get("graph_manifest", {})
    if graph.get("sha256") != sha256_file(ROOT / GRAPH_MANIFEST):
        raise RuntimeError(f"training graph manifest mismatch: {result_path}")
    poison = result.get("poison_manifest")
    if clean:
        if poison is not None or config.get("poison_manifest") is not None:
            raise RuntimeError(f"clean result has poison data: {result_path}")
    else:
        if config.get("poison_manifest") != POISON_MANIFEST.as_posix():
            raise RuntimeError(f"victim uses another poison manifest: {result_path}")
        if not isinstance(poison, dict):
            raise RuntimeError(f"victim lacks poison provenance: {result_path}")
        required_poison = {
            "path": POISON_MANIFEST.as_posix(),
            "sha256": sha256_file(ROOT / POISON_MANIFEST),
            "metadata_path": POISON_METADATA.as_posix(),
            "metadata_sha256": sha256_file(ROOT / POISON_METADATA),
            "rows": 5776,
            "applied_train_rows": 5776,
            "allocation_policy": ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
            "selection_objective": SELECTION_OBJECTIVE,
            "edge_feature_mode": MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        }
        for key, value in required_poison.items():
            if poison.get(key) != value:
                raise RuntimeError(
                    f"victim poison binding mismatch {result_path}/{key}"
                )
        if int(result.get("train_stats", {}).get("poisoned_graphs", -1)) != 5776:
            raise RuntimeError(f"not all poison rows were applied: {result_path}")
        if int(result.get("val_stats", {}).get("poisoned_graphs", -1)) != 0:
            raise RuntimeError(f"validation data was transformed: {result_path}")
    return result, _checkpoint_path(result_path, result)


def _assert_path_and_sha(
    record: Any,
    expected_path: Path,
    *,
    label: str,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"validation result lacks {label}")
    recorded_path = _relative(str(record.get("path", "")))
    if recorded_path != expected_path:
        raise RuntimeError(f"validation {label} points to another asset")
    if str(record.get("sha256", "")) != sha256_file(ROOT / expected_path):
        raise RuntimeError(f"validation {label} SHA-256 mismatch")


def _verify_motion_groups(
    scores: pd.DataFrame,
    result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    integrity.require_columns(
        scores,
        {"scenario_id", "motion_regime", "min_endpoint_speed_mps"},
        label="motion target scores",
    )
    speed = integrity.numeric_array(
        scores, "min_endpoint_speed_mps", label="motion target scores"
    )
    expected = np.where(
        speed < MOTION_BOUNDARY_MPS,
        SLOW_MOTION_REGIME,
        MOVING_MOTION_REGIME,
    )
    observed = scores["motion_regime"].astype(str).to_numpy()
    if not np.array_equal(expected, observed):
        raise RuntimeError("motion-regime labels do not match endpoint speeds")
    if set(observed) != {SLOW_MOTION_REGIME, MOVING_MOTION_REGIME}:
        raise RuntimeError("validation motion-regime coverage is incomplete")

    own_threshold = float(result["thresholds"]["evaluated_model_own"])
    common_threshold = float(result["thresholds"]["common"])
    output: dict[str, dict[str, Any]] = {}
    for regime, frame in scores.groupby("motion_regime", sort=True):
        own = integrity.target_metrics(frame, own_threshold)
        common = integrity.target_metrics(frame, common_threshold)
        own.pop("target_probability_delta_mean")
        common.pop("target_probability_delta_mean")
        integrity.assert_metrics(
            f"per_motion_regime/{regime}",
            own,
            result["per_motion_regime"][str(regime)],
        )
        integrity.assert_metrics(
            f"per_motion_regime_common_threshold/{regime}",
            common,
            result["per_motion_regime_common_threshold"][str(regime)],
        )
        output[str(regime)] = common
    return output


def _conditional_low_speed_metrics(
    scores: pd.DataFrame,
    pairs: pd.DataFrame,
    transitions: pd.DataFrame,
    common_threshold: float,
) -> dict[str, Any]:
    low_scenarios = set(
        scores.loc[
            scores["motion_regime"].astype(str) == SLOW_MOTION_REGIME,
            "scenario_id",
        ].astype(str)
    )
    low_pairs = pairs[pairs["scenario_id"].astype(str).isin(low_scenarios)]
    low_transitions = transitions[
        transitions["scenario_id"].astype(str).isin(low_scenarios)
    ]
    labels = integrity.numeric_array(low_pairs, "label", label="low-speed pairs")
    probabilities = integrity.numeric_array(
        low_pairs, "pair_probability", label="low-speed pairs"
    )
    pair_metrics = integrity.classification_metrics(
        labels.astype(np.int8), probabilities, common_threshold
    )
    collateral = integrity.collateral_metrics(low_transitions, "common")
    return {
        "scenarios": len(low_scenarios),
        "clean_pair_metrics": pair_metrics,
        "collateral_metrics": collateral,
    }


def _verify_validation_result(
    path: Path,
    *,
    seed: int,
    evaluator_sha256: str,
    victim_result_path: Path,
    victim_checkpoint_path: Path,
    clean_result_path: Path,
    clean_checkpoint_path: Path,
) -> dict[str, Any]:
    path = _relative(path)
    result = _load_json(ROOT / path)
    if result.get("split") != "val":
        raise ValueError(f"not a validation result: {path}")
    if result.get("evaluation_complete") is not True:
        raise RuntimeError(f"validation result is incomplete: {path}")
    if result.get("formal_complete") is not False:
        raise RuntimeError(f"validation result claims formal test status: {path}")
    if result.get("evaluation_phase") != "development_validation":
        raise RuntimeError(f"unexpected evaluation phase: {path}")
    evaluator = result.get("evaluator", {})
    if evaluator.get("path") != "jcas/workflows/evaluator.py":
        raise RuntimeError(f"validation evaluator path mismatch: {path}")
    if evaluator.get("sha256") != evaluator_sha256:
        raise RuntimeError(f"validation evaluator was not pre-frozen: {path}")

    _assert_path_and_sha(
        result.get("checkpoint"), victim_checkpoint_path, label="victim checkpoint"
    )
    checkpoint = result["checkpoint"]
    if _relative(str(checkpoint.get("training_result", ""))) != victim_result_path:
        raise RuntimeError(f"validation uses another victim result: {path}")
    if checkpoint.get("training_result_sha256") != sha256_file(
        ROOT / victim_result_path
    ):
        raise RuntimeError(f"validation victim result SHA mismatch: {path}")

    clean = result.get("clean_reference")
    if not isinstance(clean, dict):
        raise ValueError(f"validation lacks clean reference: {path}")
    if _relative(str(clean.get("result_path", ""))) != clean_result_path:
        raise RuntimeError(f"validation uses another clean result: {path}")
    if clean.get("result_sha256") != sha256_file(ROOT / clean_result_path):
        raise RuntimeError(f"validation clean result SHA mismatch: {path}")
    if _relative(str(clean.get("checkpoint_path", ""))) != clean_checkpoint_path:
        raise RuntimeError(f"validation uses another clean checkpoint: {path}")
    if clean.get("checkpoint_sha256") != sha256_file(ROOT / clean_checkpoint_path):
        raise RuntimeError(f"validation clean checkpoint SHA mismatch: {path}")

    target_path, _ = integrity.verify_record(
        result.get("target_manifest"), label="target manifest"
    )
    score_path, _ = integrity.verify_record(
        result.get("target_scores"), label="target scores"
    )
    evidence = result.get("metric_evidence")
    if not isinstance(evidence, dict) or evidence.get("complete") is not True:
        raise RuntimeError(f"validation metric evidence is incomplete: {path}")
    directed_path, _ = integrity.verify_record(
        evidence.get("clean_edge_scores"), label="clean edge scores"
    )
    pair_path, _ = integrity.verify_record(
        evidence.get("clean_pair_scores"), label="clean pair scores"
    )
    transition_path, _ = integrity.verify_record(
        evidence.get("collateral_transitions"), label="collateral transitions"
    )

    target = pd.read_csv(ROOT / target_path)
    scores = pd.read_csv(ROOT / score_path)
    directed = pd.read_csv(ROOT / directed_path)
    pairs = pd.read_csv(ROOT / pair_path)
    transitions = pd.read_csv(ROOT / transition_path)
    integrity.EXPECTED_TARGET_ROWS = EXPECTED_TARGET_ROWS
    target_metrics = integrity.verify_target_assets(target, scores, result)
    prediction_metrics = integrity.verify_prediction_evidence(
        directed, pairs, transitions, scores, result
    )
    motion_metrics = _verify_motion_groups(scores, result)
    low_speed = _conditional_low_speed_metrics(
        scores,
        pairs,
        transitions,
        float(result["thresholds"]["common"]),
    )
    assets = {
        path,
        target_path,
        score_path,
        directed_path,
        pair_path,
        transition_path,
    }
    return {
        "document": result,
        "assets": assets,
        "target_sha256": str(result["target_manifest"]["sha256"]),
        "target_metrics": target_metrics,
        "prediction_metrics": prediction_metrics,
        "motion_metrics": motion_metrics,
        "low_speed": low_speed,
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
    }


def _result_row(seed: int, record: dict[str, Any]) -> dict[str, Any]:
    attack = {
        **record["target_metrics"]["common"],
        **record["prediction_metrics"]["collateral_common"],
    }
    pair = record["prediction_metrics"]["pair_common"]
    slow_target = record["motion_metrics"][SLOW_MOTION_REGIME]
    slow_pair = record["low_speed"]["clean_pair_metrics"]
    slow_collateral = record["low_speed"]["collateral_metrics"]
    return {
        "seed": seed,
        "target_rows": attack["targets"],
        "common_threshold": attack["threshold"],
        "clean_activation_rate": attack["clean_activation_rate"],
        "absolute_asr": attack["absolute_asr"],
        "incremental_asr": attack["incremental_flip_rate_all_targets"],
        "conditional_flip_rate": attack["conditional_flip_rate"],
        "target_probability_delta_mean": attack["target_probability_delta_mean"],
        "nonincident_incremental_fp": attack[
            "nonincident_negative_fp_incremental"
        ],
        "adjacent_incremental_fp": attack["adjacent_negative_fp_incremental"],
        "adjacent_positive_suppression": attack[
            "adjacent_positive_suppression_incremental"
        ],
        "clean_pair_auc": pair["auc"],
        "clean_pair_pr_auc": pair["pr_auc"],
        "clean_pair_f1": pair["f1"],
        "slow_target_rows": slow_target["targets"],
        "slow_clean_activation_rate": slow_target["clean_activation_rate"],
        "slow_absolute_asr": slow_target["absolute_asr"],
        "slow_incremental_asr": slow_target[
            "incremental_flip_rate_all_targets"
        ],
        "slow_conditional_flip_rate": slow_target["conditional_flip_rate"],
        "slow_adjacent_incremental_fp": slow_collateral[
            "adjacent_negative_fp_incremental"
        ],
        "slow_nonincident_incremental_fp": slow_collateral[
            "nonincident_negative_fp_incremental"
        ],
        "slow_clean_pair_auc": slow_pair["auc"],
    }


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / Path(args.output_dir)).resolve()
    output_dir.relative_to(ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_path = output_dir / f"v6_4_validation_assets_{DATE_TAG}.sha256"
    summary_path = output_dir / f"v6_4_validation_results_{DATE_TAG}.csv"
    metadata_path = output_dir / f"v6_4_validation_results_{DATE_TAG}.json"
    release_path = output_dir / f"v6_4_validation_release_{DATE_TAG}.json"
    anchor_path = release_path.with_suffix(".sha256")
    outputs = (assets_path, summary_path, metadata_path, release_path, anchor_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "v6.4 final validation release exists: "
            + ", ".join(path.as_posix() for path in existing)
        )

    pretraining = _verify_pretraining_release()
    poison_metadata = _load_json(ROOT / POISON_METADATA)
    if poison_metadata.get("experiment") != SLOW_PRIORITIZED_VERSION:
        raise RuntimeError("unexpected low-speed poison manifest method")
    if poison_metadata.get("manifest_sha256") != sha256_file(ROOT / POISON_MANIFEST):
        raise RuntimeError("low-speed poison manifest SHA-256 mismatch")
    if int(poison_metadata.get("poisoned_scenarios", -1)) != 5776:
        raise RuntimeError("low-speed poison manifest row count mismatch")

    assets: set[Path] = {
        _relative(PRETRAINING_RELEASE),
        _relative(PRETRAINING_ANCHOR),
        _relative(GRAPH_MANIFEST),
        _relative(GRAPH_CONTRACT),
        _relative(POISON_MANIFEST),
        _relative(POISON_METADATA),
        Path(__file__).resolve().relative_to(ROOT),
    }
    for path, _ in pretraining["records"].values():
        assets.add(path)

    verified: dict[int, dict[str, Any]] = {}
    target_hashes: set[str] = set()
    rows: list[dict[str, Any]] = []
    training_assets: dict[str, Any] = {}
    for seed in SEEDS:
        clean_result, clean_checkpoint = _verify_training_result(
            CLEAN_RESULTS[seed], seed=seed, clean=True
        )
        victim_result, victim_checkpoint = _verify_training_result(
            VICTIM_RESULTS[seed], seed=seed, clean=False
        )
        clean_path = _relative(CLEAN_RESULTS[seed])
        victim_path = _relative(VICTIM_RESULTS[seed])
        record = _verify_validation_result(
            VALIDATION_RESULTS[seed],
            seed=seed,
            evaluator_sha256=pretraining["evaluator_sha256"],
            victim_result_path=victim_path,
            victim_checkpoint_path=victim_checkpoint,
            clean_result_path=clean_path,
            clean_checkpoint_path=clean_checkpoint,
        )
        verified[seed] = record
        rows.append(_result_row(seed, record))
        target_hashes.add(record["target_sha256"])
        assets.update(record["assets"])
        assets.update(
            {clean_path, clean_checkpoint, victim_path, victim_checkpoint}
        )
        training_assets[str(seed)] = {
            "clean_result": clean_path.as_posix(),
            "clean_result_sha256": sha256_file(ROOT / clean_path),
            "clean_checkpoint": clean_checkpoint.as_posix(),
            "clean_checkpoint_sha256": sha256_file(ROOT / clean_checkpoint),
            "victim_result": victim_path.as_posix(),
            "victim_result_sha256": sha256_file(ROOT / victim_path),
            "victim_checkpoint": victim_checkpoint.as_posix(),
            "victim_checkpoint_sha256": sha256_file(ROOT / victim_checkpoint),
            "clean_val_loss": clean_result["losses"]["val"],
            "victim_val_loss": victim_result["losses"]["val"],
        }
    if len(target_hashes) != 1:
        raise RuntimeError("three v6.4 evaluations used different target pools")

    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(summary_path, index=False)
    numeric_columns = [
        column
        for column in summary_frame.columns
        if column not in {"seed", "target_rows", "slow_target_rows"}
    ]
    three_seed_summary = {
        column: _summary([float(value) for value in summary_frame[column]])
        for column in numeric_columns
    }
    gates = pretraining["release"]["continuation_gates"]
    per_seed_gate = {}
    for row in rows:
        per_seed_gate[str(row["seed"])] = {
            "slow_incremental_asr": row["slow_incremental_asr"]
            >= float(gates["slow_lt_0p5_incremental_asr_min"]),
            "overall_incremental_asr": row["incremental_asr"]
            >= float(gates["overall_incremental_asr_min"]),
            "slow_clean_activation": row["slow_clean_activation_rate"]
            <= float(gates["slow_lt_0p5_clean_activation_max"]),
            "adjacent_incremental_fp": row["adjacent_incremental_fp"]
            <= float(gates["adjacent_negative_fp_incremental_max"]),
            "clean_pair_auc": row["clean_pair_auc"]
            >= float(gates["clean_pair_auc_min"]),
        }
    asset_lines = [
        f"{sha256_file(ROOT / path)}  {path.as_posix()}"
        for path in sorted(assets, key=lambda item: item.as_posix())
    ]
    _atomic_text("\n".join(asset_lines) + "\n", assets_path)

    metadata = {
        "scope": "offline_authorized_av2_model_robustness",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "version": VERSION,
        "status": "postrun_three_seed_validation_frozen",
        "development_only": True,
        "formal_test_result": False,
        "test_accessed": False,
        "interpretation": (
            "three-seed development validation; the exact implementation was "
            "frozen before the first victim training run"
        ),
        "method": {
            "poison_rate": 0.05,
            "poisoned_train_scenarios": 5776,
            "pair_selection": SLOW_PRIORITIZED_VERSION,
            "slow_boundary_mps": MOTION_BOUNDARY_MPS,
            "edge_feature_mode": MOTION_NORMALIZED_EDGE_FEATURE_MODE,
            "relative_displacement_m": 0.2,
            "perturb_window": 10,
            "ramp_style": "minimum_jerk",
            "velocity_mode": "residual",
            "allocation_policy": ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
            "allocation_alpha": 0.5,
            "training_objective": "ordinary_bce",
        },
        "pretraining_release": {
            "path": pretraining["path"].as_posix(),
            "sha256": pretraining["sha256"],
            "anchor_path": pretraining["anchor_path"].as_posix(),
            "evaluator_sha256": pretraining["evaluator_sha256"],
            "trainer_sha256": pretraining["trainer_sha256"],
        },
        "training_assets": training_assets,
        "validation_target_pool": {
            "rows": EXPECTED_TARGET_ROWS,
            "sha256": next(iter(target_hashes)),
            "slow_target_rows": int(summary_frame["slow_target_rows"].iloc[0]),
        },
        "per_seed": {str(row["seed"]): row for row in rows},
        "three_seed_summary": three_seed_summary,
        "frozen_gates": gates,
        "per_seed_gate_results": per_seed_gate,
        "semantic_recomputation": {
            "target_metrics": True,
            "pair_auc_pr_auc_f1": True,
            "collateral_2x2_metrics": True,
            "motion_regime_membership": True,
            "low_speed_conditional_metrics": True,
            "errors": 0,
        },
        "asset_manifest": {
            "path": assets_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(assets_path),
            "entries": len(asset_lines),
        },
        "summary_csv": {
            "path": summary_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(summary_path),
            "rows": len(summary_frame),
        },
        "postrun_auditor": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    _atomic_json(metadata, metadata_path)
    release = {
        "scope": "offline_authorized_av2_model_robustness",
        "version": VERSION,
        "status": "frozen_postrun_validation_release",
        "formal_test_result": False,
        "test_accessed": False,
        "pretraining_release": {
            "path": pretraining["path"].as_posix(),
            "sha256": pretraining["sha256"],
        },
        "experiment_assets": {
            "path": assets_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(assets_path),
        },
        "results": {
            "metadata_path": metadata_path.relative_to(ROOT).as_posix(),
            "metadata_sha256": sha256_file(metadata_path),
            "csv_path": summary_path.relative_to(ROOT).as_posix(),
            "csv_sha256": sha256_file(summary_path),
        },
    }
    _atomic_json(release, release_path)
    release_sha = sha256_file(release_path)
    _atomic_text(
        f"{release_sha}  {release_path.relative_to(ROOT).as_posix()}\n",
        anchor_path,
    )
    print(f"v6.4 validation release: {release_path.relative_to(ROOT)}")
    print(f"external release SHA-256: {release_sha}")
    print("semantic recomputation errors: 0")


if __name__ == "__main__":
    main()
