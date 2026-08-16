#!/usr/bin/env python3
"""Independently verify and freeze v4.1.1 confirmatory re-evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from jcas import PROJECT_ROOT


ROOT = PROJECT_ROOT
SEEDS = (20260621, 20260622, 20260623)
PACKAGE_VERSION = "v4.1.1_evaluation_integrity"
EVALUATOR_VERSION = "v4.1.1_pretest_release_metric_evidence"
METHOD_VERSION = "v4_1_crossfit_bce_informative_graybox"
EXPECTED_TARGET_ROWS = 24450
DEFAULT_PRETEST_RELEASE = Path(
    "record/v4_1_1_graybox/contracts/v4_1_1_pretest_release_20260811.json"
)
DEFAULT_TEST_DIR = Path("record/v4_1_1_graybox/test")
DEFAULT_OUTPUT_DIR = Path("record/v4_1_1_graybox/contracts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently verify v4.1.1 formal re-evaluation assets"
    )
    parser.add_argument(
        "--pretest-release", default=str(DEFAULT_PRETEST_RELEASE)
    )
    parser.add_argument("--pretest-release-sha256", required=True)
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    atomic_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", path
    )


def repository_relative(value: str, *, label: str) -> Path:
    recorded = Path(str(value))
    if recorded.is_absolute():
        try:
            recorded = recorded.resolve().relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(f"{label} path escapes repository") from exc
    resolved = (ROOT / recorded).resolve()
    try:
        return resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes repository") from exc


def parse_sha_manifest(path: Path) -> dict[str, str]:
    pattern = re.compile(r"^([0-9a-f]{64})  (.+)$")
    entries: dict[str, str] = {}
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid SHA manifest line {number}: {path}")
        digest, value = match.groups()
        if value in entries:
            raise ValueError(f"duplicate SHA manifest path: {value}")
        entries[value] = digest
    if not entries:
        raise ValueError(f"empty SHA manifest: {path}")
    return entries


def verify_record(record: Any, *, label: str) -> tuple[Path, str]:
    if not isinstance(record, dict):
        raise ValueError(f"missing {label} record")
    relative = repository_relative(str(record.get("path", "")), label=label)
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {relative}")
    expected = str(record.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{label} has an invalid SHA-256")
    if sha256_file(path) != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {relative}")
    return relative, expected


def verify_pretest_release(path: Path, expected_sha256: str) -> dict[str, Any]:
    release_path = path.resolve()
    release_relative = release_path.relative_to(ROOT)
    expected = str(expected_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("pretest release trust anchor is not one SHA-256")
    if sha256_file(release_path) != expected:
        raise RuntimeError("pretest release does not match external trust anchor")
    release = load_json(release_path)
    if release.get("version") != PACKAGE_VERSION:
        raise ValueError("unexpected pretest release version")
    if release.get("status") != "pre_frozen_before_v4_1_1_re_evaluation":
        raise ValueError("unexpected pretest release status")
    if int(release.get("test_outputs_present_before_freeze", -1)) != 0:
        raise RuntimeError("pretest release was not frozen before outputs existed")

    code_path, code_sha = verify_record(
        release.get("code_asset_manifest"), label="code asset manifest"
    )
    code_entries = parse_sha_manifest(ROOT / code_path)
    for recorded_path, digest in code_entries.items():
        relative = repository_relative(recorded_path, label="code asset")
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"frozen source changed: {relative}")
    evaluator_path, evaluator_sha = verify_record(
        release.get("evaluator"), label="evaluator"
    )
    finalizer_path, finalizer_sha = verify_record(
        release.get("finalizer"), label="finalizer"
    )
    if evaluator_path.as_posix() != "eval_blackbox_poison.py":
        raise RuntimeError("evaluator does not use its canonical path")
    expected_finalizer = Path(__file__).resolve().relative_to(ROOT)
    if finalizer_path != expected_finalizer:
        raise RuntimeError("running finalizer is not the frozen finalizer")
    if code_entries.get(evaluator_path.as_posix()) != evaluator_sha:
        raise RuntimeError("evaluator is absent from frozen code manifest")
    if code_entries.get(finalizer_path.as_posix()) != finalizer_sha:
        raise RuntimeError("finalizer is absent from frozen code manifest")

    formal_path, formal_sha = verify_record(
        release.get("formal_contract"), label="formal contract"
    )
    frozen_path, frozen_sha = verify_record(
        release.get("frozen_asset_manifest"), label="frozen asset manifest"
    )
    source_path, source_sha = verify_record(
        release.get("source_archive"), label="source archive"
    )
    environment_path, environment_sha = verify_record(
        release.get("environment"), label="environment summary"
    )
    formal = load_json(ROOT / formal_path)
    if formal.get("version") != METHOD_VERSION:
        raise ValueError("pretest release is bound to another method")
    declared = formal.get("asset_hash_manifest", {})
    if declared.get("sha256") != frozen_sha:
        raise RuntimeError("formal contract and frozen asset manifest disagree")
    if repository_relative(
        str(declared.get("path", "")), label="frozen asset manifest"
    ) != frozen_path:
        raise RuntimeError("formal contract records another asset manifest")
    return {
        "path": release_relative,
        "sha256": expected,
        "release": release,
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


def require_columns(frame: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def numeric_array(frame: pd.DataFrame, column: str, *, label: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label}/{column} contains NaN or Inf")
    return values


def boolean_array(frame: pd.DataFrame, column: str, *, label: str) -> np.ndarray:
    values = frame[column]
    if values.dtype == bool:
        return values.to_numpy(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{label}/{column} contains invalid booleans")
    return normalized.eq("true").to_numpy(bool)


def unique_threshold(frame: pd.DataFrame, column: str, *, label: str) -> float:
    values = numeric_array(frame, column, label=label)
    unique = np.unique(values)
    if unique.size != 1 or not 0.0 < float(unique[0]) < 1.0:
        raise ValueError(f"{label}/{column} is not one valid threshold")
    return float(unique[0])


def assert_value(label: str, recomputed: Any, reported: Any) -> None:
    if recomputed is None or reported is None:
        if recomputed is not None or reported is not None:
            raise ValueError(f"{label} None mismatch")
        return
    if isinstance(recomputed, (bool, str, int, np.integer)):
        if reported != recomputed:
            raise ValueError(f"{label} mismatch: {reported!r} != {recomputed!r}")
        return
    left = float(recomputed)
    right = float(reported)
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError(f"{label} is not finite")
    if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: {right} != {left}")


def assert_metrics(
    label: str,
    recomputed: dict[str, Any],
    reported: Any,
) -> None:
    if not isinstance(reported, dict):
        raise ValueError(f"{label} is not an object")
    for key, value in recomputed.items():
        if key not in reported:
            raise ValueError(f"{label} has no {key}")
        assert_value(f"{label}/{key}", value, reported[key])


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = probabilities >= float(threshold)
    both = np.unique(labels).size >= 2
    return {
        "edges": int(labels.size),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "auc": float(roc_auc_score(labels, probabilities)) if both else None,
        "pr_auc": (
            float(average_precision_score(labels, probabilities)) if both else None
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold": float(threshold),
    }


def target_metrics(scores: pd.DataFrame, threshold: float) -> dict[str, Any]:
    clean = numeric_array(scores, "clean_pair_probability", label="target scores")
    triggered = numeric_array(
        scores, "triggered_pair_probability", label="target scores"
    )
    clean_negative = clean < float(threshold)
    triggered_positive = triggered >= float(threshold)
    flipped = clean_negative & triggered_positive
    eligible = int(clean_negative.sum())
    return {
        "threshold": float(threshold),
        "targets": int(len(scores)),
        "clean_activation_rate": float((~clean_negative).mean()),
        "absolute_asr": float(triggered_positive.mean()),
        "incremental_flip_rate_all_targets": float(flipped.mean()),
        "conditional_flip_rate": (
            float(flipped.sum() / eligible) if eligible else None
        ),
        "clean_negative_targets": eligible,
        "incremental_flips": int(flipped.sum()),
        "target_probability_delta_mean": float((triggered - clean).mean()),
    }


def collateral_metrics(
    transitions: pd.DataFrame, threshold_kind: str
) -> dict[str, Any]:
    selected = transitions[transitions["threshold_kind"] == threshold_kind]
    aggregate = selected.groupby("subset")[["n00", "n01", "n10", "n11", "total"]].sum()
    if set(aggregate.index) != {
        "nonincident_negative",
        "adjacent_negative",
        "adjacent_positive",
    }:
        raise ValueError(f"incomplete collateral subsets for {threshold_kind}")

    def rate(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    nonincident = aggregate.loc["nonincident_negative"]
    adjacent_negative = aggregate.loc["adjacent_negative"]
    adjacent_positive = aggregate.loc["adjacent_positive"]
    nonincident_total = int(nonincident["total"])
    adjacent_negative_total = int(adjacent_negative["total"])
    adjacent_positive_total = int(adjacent_positive["total"])
    return {
        "nonincident_negative_fp_absolute": rate(
            int(nonincident["n01"] + nonincident["n11"]), nonincident_total
        ),
        "nonincident_negative_fp_incremental": rate(
            int(nonincident["n01"]), nonincident_total
        ),
        "nonincident_negative_edges": nonincident_total,
        "nonincident_negative_status": "ok" if nonincident_total else "empty",
        "adjacent_negative_fp_absolute": rate(
            int(adjacent_negative["n01"] + adjacent_negative["n11"]),
            adjacent_negative_total,
        ),
        "adjacent_negative_fp_incremental": rate(
            int(adjacent_negative["n01"]), adjacent_negative_total
        ),
        "adjacent_negative_edges": adjacent_negative_total,
        "adjacent_negative_status": (
            "ok" if adjacent_negative_total else "empty"
        ),
        "adjacent_positive_suppression_absolute": rate(
            int(adjacent_positive["n00"] + adjacent_positive["n10"]),
            adjacent_positive_total,
        ),
        "adjacent_positive_suppression_incremental": rate(
            int(adjacent_positive["n10"]), adjacent_positive_total
        ),
        "adjacent_positive_edges": adjacent_positive_total,
        "adjacent_positive_status": (
            "ok" if adjacent_positive_total else "empty"
        ),
    }


def verify_target_assets(
    target: pd.DataFrame,
    scores: pd.DataFrame,
    result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    identity = ["scenario_id", "edge_id", "reverse_edge_id"]
    require_columns(target, set(identity), label="target manifest")
    require_columns(
        scores,
        set(identity)
        | {
            "clean_pair_probability",
            "triggered_pair_probability",
            "probability_delta",
            "evaluated_model_threshold",
            "evaluated_model_clean_positive",
            "evaluated_model_triggered_positive",
            "common_threshold",
            "common_clean_positive",
            "common_triggered_positive",
        },
        label="target scores",
    )
    if len(target) != EXPECTED_TARGET_ROWS or len(scores) != EXPECTED_TARGET_ROWS:
        raise ValueError("target or score row count mismatch")
    if target[identity].duplicated().any() or scores[identity].duplicated().any():
        raise ValueError("target identity is not unique")
    left = target[identity].astype({"scenario_id": str}).sort_values(identity)
    right = scores[identity].astype({"scenario_id": str}).sort_values(identity)
    if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
        raise ValueError("target and score identities do not match exactly")
    clean = numeric_array(scores, "clean_pair_probability", label="target scores")
    triggered = numeric_array(
        scores, "triggered_pair_probability", label="target scores"
    )
    delta = numeric_array(scores, "probability_delta", label="target scores")
    if ((clean < 0.0) | (clean > 1.0) | (triggered < 0.0) | (triggered > 1.0)).any():
        raise ValueError("target probabilities are outside [0, 1]")
    if not np.allclose(delta, triggered - clean, rtol=1e-10, atol=1e-12):
        raise ValueError("target probability deltas are inconsistent")
    own = unique_threshold(
        scores, "evaluated_model_threshold", label="target scores"
    )
    common = unique_threshold(scores, "common_threshold", label="target scores")
    assert_value("result own threshold", own, result["thresholds"]["evaluated_model_own"])
    assert_value("result common threshold", common, result["thresholds"]["common"])
    for prefix, threshold in (("evaluated_model", own), ("common", common)):
        clean_column = f"{prefix}_clean_positive"
        triggered_column = f"{prefix}_triggered_positive"
        if not np.array_equal(
            boolean_array(scores, clean_column, label="target scores"),
            clean >= threshold,
        ):
            raise ValueError(f"{clean_column} is inconsistent")
        if not np.array_equal(
            boolean_array(scores, triggered_column, label="target scores"),
            triggered >= threshold,
        ):
            raise ValueError(f"{triggered_column} is inconsistent")
    recomputed = {
        "evaluated_model": target_metrics(scores, own),
        "common": target_metrics(scores, common),
    }
    assert_metrics("attack_metrics", recomputed["evaluated_model"], result["attack_metrics"])
    assert_metrics(
        "attack_metrics_common_threshold",
        recomputed["common"],
        result["attack_metrics_common_threshold"],
    )
    for city, city_scores in scores.groupby("city", sort=True):
        own_city = target_metrics(city_scores, own)
        reported_own = result["per_city"][str(city)]
        expected_own = {
            "targets": own_city["targets"],
            "triggered_positive": int(
                (
                    numeric_array(
                        city_scores,
                        "triggered_pair_probability",
                        label="city target scores",
                    )
                    >= own
                ).sum()
            ),
            "incremental_flip": own_city["incremental_flips"],
            "absolute_asr": own_city["absolute_asr"],
            "incremental_flip_rate": own_city[
                "incremental_flip_rate_all_targets"
            ],
        }
        assert_metrics(f"per_city/{city}", expected_own, reported_own)
        common_city = target_metrics(city_scores, common)
        common_city.pop("target_probability_delta_mean")
        assert_metrics(
            f"per_city_common_threshold/{city}",
            common_city,
            result["per_city_common_threshold"][str(city)],
        )
    return recomputed


def verify_prediction_evidence(
    directed: pd.DataFrame,
    pairs: pd.DataFrame,
    transitions: pd.DataFrame,
    target_scores: pd.DataFrame,
    result: dict[str, Any],
) -> dict[str, Any]:
    require_columns(
        directed,
        {
            "scenario_id",
            "edge_id",
            "label",
            "probability",
            "evaluated_model_threshold",
        },
        label="directed evidence",
    )
    if directed[["scenario_id", "edge_id"]].duplicated().any():
        raise ValueError("directed evidence identity is not unique")
    directed_labels = numeric_array(directed, "label", label="directed evidence")
    directed_probs = numeric_array(
        directed, "probability", label="directed evidence"
    )
    if not np.isin(directed_labels, [0.0, 1.0]).all():
        raise ValueError("directed labels are not binary")
    if ((directed_probs < 0.0) | (directed_probs > 1.0)).any():
        raise ValueError("directed probabilities are outside [0, 1]")
    directed_threshold = unique_threshold(
        directed, "evaluated_model_threshold", label="directed evidence"
    )
    directed_metrics = classification_metrics(
        directed_labels.astype(np.int8), directed_probs, directed_threshold
    )
    assert_metrics("clean_test_metrics", directed_metrics, result["clean_test_metrics"])

    require_columns(
        pairs,
        {
            "scenario_id",
            "canonical_edge_id",
            "reverse_edge_id",
            "label",
            "pair_probability",
            "evaluated_model_threshold",
            "common_threshold",
        },
        label="pair evidence",
    )
    pair_identity = ["scenario_id", "canonical_edge_id", "reverse_edge_id"]
    if pairs[pair_identity].duplicated().any():
        raise ValueError("pair evidence identity is not unique")
    canonical = numeric_array(pairs, "canonical_edge_id", label="pair evidence")
    reverse = numeric_array(pairs, "reverse_edge_id", label="pair evidence")
    if not (canonical < reverse).all():
        raise ValueError("pair evidence does not use canonical edge ordering")
    pair_labels = numeric_array(pairs, "label", label="pair evidence")
    pair_probs = numeric_array(pairs, "pair_probability", label="pair evidence")
    if not np.isin(pair_labels, [0.0, 1.0]).all():
        raise ValueError("pair labels are not binary")
    if ((pair_probs < 0.0) | (pair_probs > 1.0)).any():
        raise ValueError("pair probabilities are outside [0, 1]")
    own = unique_threshold(
        pairs, "evaluated_model_threshold", label="pair evidence"
    )
    common = unique_threshold(pairs, "common_threshold", label="pair evidence")
    own_pair = classification_metrics(
        pair_labels.astype(np.int8), pair_probs, own
    )
    common_pair = classification_metrics(
        pair_labels.astype(np.int8), pair_probs, common
    )
    assert_metrics("clean_pair_metrics", own_pair, result["clean_pair_metrics"])
    assert_metrics(
        "clean_pair_metrics_common_threshold",
        common_pair,
        result["clean_pair_metrics_common_threshold"],
    )

    required_transition_columns = {
        "scenario_id",
        "threshold_kind",
        "threshold",
        "subset",
        "label",
        "n00",
        "n01",
        "n10",
        "n11",
        "total",
    }
    require_columns(
        transitions, required_transition_columns, label="collateral evidence"
    )
    identity = ["scenario_id", "threshold_kind", "subset"]
    if transitions[identity].duplicated().any():
        raise ValueError("collateral evidence identity is not unique")
    expected_rows = len(target_scores) * 2 * 3
    if len(transitions) != expected_rows:
        raise ValueError("collateral evidence row count mismatch")
    if set(transitions["threshold_kind"].astype(str)) != {
        "evaluated_model",
        "common",
    }:
        raise ValueError("collateral threshold kinds are incomplete")
    if set(transitions["subset"].astype(str)) != {
        "nonincident_negative",
        "adjacent_negative",
        "adjacent_positive",
    }:
        raise ValueError("collateral subsets are incomplete")
    target_scenarios = set(target_scores["scenario_id"].astype(str))
    for (kind, subset), frame in transitions.groupby(
        ["threshold_kind", "subset"]
    ):
        if set(frame["scenario_id"].astype(str)) != target_scenarios:
            raise ValueError(f"collateral scenario coverage mismatch: {kind}/{subset}")
    for column in ("label", "n00", "n01", "n10", "n11", "total"):
        values = numeric_array(transitions, column, label="collateral evidence")
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"collateral {column} is not a non-negative integer")
    totals = transitions[["n00", "n01", "n10", "n11"]].sum(axis=1)
    if not np.array_equal(totals.to_numpy(np.int64), transitions["total"].to_numpy(np.int64)):
        raise ValueError("collateral 2x2 cells do not sum to total")
    expected_labels = transitions["subset"].map(
        {
            "nonincident_negative": 0,
            "adjacent_negative": 0,
            "adjacent_positive": 1,
        }
    )
    if not np.array_equal(
        expected_labels.to_numpy(np.int64), transitions["label"].to_numpy(np.int64)
    ):
        raise ValueError("collateral subset labels are inconsistent")
    own_transition_threshold = unique_threshold(
        transitions[transitions["threshold_kind"] == "evaluated_model"],
        "threshold",
        label="collateral evaluated-model evidence",
    )
    common_transition_threshold = unique_threshold(
        transitions[transitions["threshold_kind"] == "common"],
        "threshold",
        label="collateral common evidence",
    )
    assert_value("collateral own threshold", own, own_transition_threshold)
    assert_value("collateral common threshold", common, common_transition_threshold)
    own_collateral = collateral_metrics(transitions, "evaluated_model")
    common_collateral = collateral_metrics(transitions, "common")
    assert_metrics("attack_metrics collateral", own_collateral, result["attack_metrics"])
    assert_metrics(
        "attack_metrics_common_threshold collateral",
        common_collateral,
        result["attack_metrics_common_threshold"],
    )
    return {
        "directed": directed_metrics,
        "pair": own_pair,
        "pair_common": common_pair,
        "collateral": own_collateral,
        "collateral_common": common_collateral,
    }


def verify_result(
    path: Path,
    *,
    seed: int,
    expected_role: str,
    release_audit: dict[str, Any],
) -> dict[str, Any]:
    result = load_json(path)
    if result.get("split") != "test":
        raise ValueError(f"not a test result: {path}")
    if result.get("evaluation_complete") is not True or result.get("formal_complete") is not True:
        raise ValueError(f"incomplete formal result: {path}")
    evaluator = result.get("evaluator", {})
    if evaluator.get("path") != release_audit["evaluator_path"].as_posix():
        raise ValueError(f"evaluator path mismatch: {path}")
    if evaluator.get("sha256") != release_audit["evaluator_sha256"]:
        raise ValueError(f"evaluator SHA mismatch: {path}")
    if evaluator.get("integrity_patch_version") != EVALUATOR_VERSION:
        raise ValueError(f"evaluator version mismatch: {path}")
    pretest = result.get("pretest_release", {})
    if pretest.get("path") != release_audit["path"].as_posix():
        raise ValueError(f"pretest release path mismatch: {path}")
    if pretest.get("sha256") != release_audit["sha256"]:
        raise ValueError(f"pretest release SHA mismatch: {path}")
    if pretest.get("verified_before_test_graph_access") is not True:
        raise ValueError(f"pretest release was not verified early: {path}")
    formal = result.get("formal_contract", {})
    if formal.get("version") != METHOD_VERSION:
        raise ValueError(f"formal method mismatch: {path}")
    if formal.get("sha256") != release_audit["formal_contract_sha256"]:
        raise ValueError(f"formal contract SHA mismatch: {path}")
    if formal.get("formal_contract_verified") is not True:
        raise ValueError(f"formal contract not verified: {path}")
    model_audit = formal.get("evaluated_model_asset_audit", {})
    if model_audit.get("role") != expected_role:
        raise ValueError(f"model role mismatch: {path}")
    if int(model_audit.get("seed", -1)) != seed:
        raise ValueError(f"model seed mismatch: {path}")
    if model_audit.get("frozen_asset_allowlist_match") is not True:
        raise ValueError(f"model is outside frozen allowlist: {path}")

    assets: list[Path] = [path.relative_to(ROOT)]
    target_path, _ = verify_record(result.get("target_manifest"), label="target manifest")
    score_path, _ = verify_record(result.get("target_scores"), label="target scores")
    evidence = result.get("metric_evidence", {})
    if evidence.get("complete") is not True:
        raise ValueError(f"metric evidence is incomplete: {path}")
    directed_path, _ = verify_record(
        evidence.get("clean_edge_scores"), label="clean edge scores"
    )
    pair_path, _ = verify_record(
        evidence.get("clean_pair_scores"), label="clean pair scores"
    )
    transition_path, _ = verify_record(
        evidence.get("collateral_transitions"),
        label="collateral transitions",
    )
    assets.extend(
        [target_path, score_path, directed_path, pair_path, transition_path]
    )
    target = pd.read_csv(ROOT / target_path)
    scores = pd.read_csv(ROOT / score_path)
    directed = pd.read_csv(ROOT / directed_path)
    pairs = pd.read_csv(ROOT / pair_path)
    transitions = pd.read_csv(ROOT / transition_path)
    records = (
        ("target_manifest", target),
        ("target_scores", scores),
        ("clean_edge_scores", directed),
        ("clean_pair_scores", pairs),
        ("collateral_transitions", transitions),
    )
    for key, frame in records:
        record = (
            result[key]
            if key in result
            else evidence[key]
        )
        if int(record.get("rows", -1)) != len(frame):
            raise ValueError(f"embedded row count mismatch: {path}/{key}")
    target_result = verify_target_assets(target, scores, result)
    prediction_result = verify_prediction_evidence(
        directed, pairs, transitions, scores, result
    )
    return {
        "document": result,
        "assets": assets,
        "target_sha256": str(result["target_manifest"]["sha256"]),
        "target_metrics": target_result,
        "prediction_metrics": prediction_result,
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    release_audit = verify_pretest_release(
        Path(args.pretest_release), str(args.pretest_release_sha256)
    )
    test_dir = (ROOT / repository_relative(args.test_dir, label="test dir")).resolve()
    output_dir = (ROOT / repository_relative(args.output_dir, label="output dir")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    test_asset_path = output_dir / "v4_1_1_final_test_assets_20260811.sha256"
    metadata_path = output_dir / "v4_1_1_final_test_20260811.metadata.json"
    release_path = output_dir / "v4_1_1_final_release_20260811.sha256"
    existing = [path for path in (test_asset_path, metadata_path, release_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "final outputs already exist; pass --force to replace: "
            + ", ".join(str(path) for path in existing)
        )

    verified: dict[tuple[int, str], dict[str, Any]] = {}
    test_assets: set[Path] = set()
    target_hashes: set[str] = set()
    for seed in SEEDS:
        for kind, role in (
            ("reference", "clean_reference"),
            ("victim", "v4_1_graybox_victim"),
        ):
            result_path = test_dir / f"{kind}_seed{seed}.json"
            record = verify_result(
                result_path,
                seed=seed,
                expected_role=role,
                release_audit=release_audit,
            )
            verified[(seed, kind)] = record
            test_assets.update(record["assets"])
            target_hashes.add(record["target_sha256"])
    if len(target_hashes) != 1:
        raise ValueError("six evaluations did not use one common target pool")

    asset_lines = [
        f"{sha256_file(ROOT / path)}  {path.as_posix()}"
        for path in sorted(test_assets, key=lambda item: item.as_posix())
    ]
    atomic_text("\n".join(asset_lines) + "\n", test_asset_path)

    per_seed: dict[str, Any] = {}
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
    summary_values: dict[str, list[float]] = {
        key: [] for key in metric_keys
    }
    pair_values: dict[str, list[float]] = {
        key: [] for key in ("auc", "pr_auc", "f1")
    }
    reference_incremental: list[float] = []
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
        ref_inc = float(reference_attack["incremental_flip_rate_all_targets"])
        reference_incremental.append(ref_inc)
        for key in metric_keys:
            summary_values[key].append(float(victim_attack[key]))
        for key in pair_values:
            pair_values[key].append(float(victim_pair[key]))
        per_seed[str(seed)] = {
            "common_threshold": float(victim_pair["threshold"]),
            "reference_incremental_flip_rate": ref_inc,
            **{f"victim_{key}": float(victim_attack[key]) for key in metric_keys},
            **{f"victim_pair_{key}": float(victim_pair[key]) for key in pair_values},
            "reference_adjusted_incremental_effect": float(
                victim_attack["incremental_flip_rate_all_targets"] - ref_inc
            ),
        }
    summary = {
        "reference_incremental_flip_rate": summarize(reference_incremental),
        **{
            f"victim_{key}": summarize(values)
            for key, values in summary_values.items()
        },
        **{
            f"victim_pair_{key}": summarize(values)
            for key, values in pair_values.items()
        },
        "reference_adjusted_incremental_effect": summarize(
            [
                per_seed[str(seed)]["reference_adjusted_incremental_effect"]
                for seed in SEEDS
            ]
        ),
    }
    metadata = {
        "scope": "offline_authorized_av2_model_robustness",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "v4_1_1_confirmatory_re_evaluation_complete",
        "version": PACKAGE_VERSION,
        "interpretation": (
            "re-evaluation under a pre-frozen v4.1.1 evaluation package; "
            "not a globally untouched blind test"
        ),
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
            "sha256": sha256_file(test_asset_path),
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
            "rows": EXPECTED_TARGET_ROWS,
            "common_sha256": next(iter(target_hashes)),
        },
        "per_seed": per_seed,
        "three_seed_summary": summary,
    }
    atomic_json(metadata, metadata_path)

    pretest_anchor_path = Path(args.pretest_release).with_suffix(".sha256")
    release_assets = {
        Path(__file__).resolve().relative_to(ROOT),
        release_audit["evaluator_path"],
        release_audit["path"],
        pretest_anchor_path.resolve().relative_to(ROOT),
        release_audit["code_manifest_path"],
        release_audit["source_archive_path"],
        test_asset_path.relative_to(ROOT),
        metadata_path.relative_to(ROOT),
    }
    release_lines = [
        f"{sha256_file(ROOT / path)}  {path.as_posix()}"
        for path in sorted(release_assets, key=lambda item: item.as_posix())
    ]
    atomic_text("\n".join(release_lines) + "\n", release_path)
    print(f"formal v4.1.1 results independently verified: 6/6")
    print(f"test assets: {test_asset_path.relative_to(ROOT)} ({len(test_assets)} entries)")
    print(f"metadata: {metadata_path.relative_to(ROOT)}")
    print(f"release: {release_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
