#!/usr/bin/env python3
"""Freeze the v6.0 same-pair, fixed-symmetric train manifest.

The input pair for every poisoned training scenario is inherited verbatim
from the frozen v5 C-3 manifest.  Only the allocation row is replaced by the
matching alpha=0.5 candidate that was already scored using train-only strict
cross-fitting.  No model inference or validation/test asset is used here.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jcas.core.poison import (
    ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
    FIXED_SYMMETRIC_BIEND_ALPHA,
    load_poison_manifest,
    sha256_file,
    validate_strict_crossfit_bindings,
)
from jcas.core.risk_labels import RiskLabelConfig


VERSION = "v6_0_same_pair_fixed_symmetric_training_manifest_v1"
SELECTION_OBJECTIVE = (
    "v5_gradient_pair_selection_then_fixed_symmetric_alpha_v6_0"
)
PAIR_KEYS = (
    "scenario_id",
    "src",
    "dst",
    "src_track_id",
    "dst_track_id",
)
STRICT_BINDING_COLUMNS = (
    "scenario_shadow_fold",
    "surrogate_heldout_fold",
    "surrogate_checkpoint_sha256",
    "surrogate_fit_manifest_sha256",
    "surrogate_score_manifest_sha256",
    "surrogate_protocol",
)
REQUIRED_CANDIDATE_COLUMNS = set(PAIR_KEYS) | set(STRICT_BINDING_COLUMNS) | {
    "shadow_fold",
    "alpha",
    "clean_probability_mean",
    "triggered_probability_mean",
    "delta_mean",
    "delta_std",
    "response_utility",
    "bce_informative_utility",
    "selection_utility",
    "total_feature_energy",
    "incident_edge_energy",
    "endpoint_node_energy",
    "non_target_incident_edges",
    "within_pair_feature_budget",
    "gradient_alignment_mean",
    "gradient_alignment_std",
    "gradient_alignment_robust",
    "gradient_effect_norm_mean",
    "target_probe_count_min",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze v6.0 same-pair alpha=0.5 poison manifest"
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-candidates", required=True)
    parser.add_argument("--source-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(
            temporary,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
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


def _strict_bool(series: pd.Series, *, name: str) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{name} contains invalid booleans")
    return normalized.eq("true").to_numpy(bool)


def build_fixed_symmetric_manifest(
    source: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Replace allocation only, preserving every frozen v5 selected pair."""
    missing_source = sorted(set(PAIR_KEYS) - set(source.columns))
    missing_candidates = sorted(REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns))
    if missing_source:
        raise ValueError(f"source manifest is missing columns: {missing_source}")
    if missing_candidates:
        raise ValueError(
            f"source candidate table is missing columns: {missing_candidates}"
        )
    if source.empty or source["scenario_id"].astype(str).duplicated().any():
        raise ValueError("source manifest scenarios must be non-empty and unique")

    source = source.copy()
    candidates = candidates.copy()
    for frame in (source, candidates):
        frame["scenario_id"] = frame["scenario_id"].astype(str)
        frame["src_track_id"] = frame["src_track_id"].astype(str)
        frame["dst_track_id"] = frame["dst_track_id"].astype(str)
        frame["src"] = pd.to_numeric(frame["src"], errors="raise").astype(np.int64)
        frame["dst"] = pd.to_numeric(frame["dst"], errors="raise").astype(np.int64)

    alpha = pd.to_numeric(candidates["alpha"], errors="coerce").to_numpy(float)
    if not np.isfinite(alpha).all():
        raise ValueError("candidate alpha contains non-finite values")
    fixed = candidates[
        np.isclose(
            alpha,
            FIXED_SYMMETRIC_BIEND_ALPHA,
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()
    if fixed.duplicated(list(PAIR_KEYS)).any():
        raise ValueError("candidate table has duplicate same-pair alpha=0.5 rows")

    lookup_columns = list(PAIR_KEYS) + [
        column for column in fixed.columns if column not in PAIR_KEYS
    ]
    request = source[list(PAIR_KEYS)].copy()
    request["_source_order"] = np.arange(len(request), dtype=np.int64)
    matched = request.merge(
        fixed[lookup_columns],
        on=list(PAIR_KEYS),
        how="left",
        validate="one_to_one",
        indicator=True,
    ).sort_values("_source_order")
    if not matched["_merge"].eq("both").all():
        missing = matched.loc[
            matched["_merge"] != "both", "scenario_id"
        ].astype(str).tolist()[:5]
        raise RuntimeError(
            "one or more frozen v5 pairs have no unique alpha=0.5 candidate: "
            f"{missing}"
        )
    matched = matched.drop(columns=["_merge"]).reset_index(drop=True)
    source = source.reset_index(drop=True)
    if not source[list(PAIR_KEYS)].equals(matched[list(PAIR_KEYS)]):
        raise RuntimeError("same-pair identity changed during alpha=0.5 lookup")

    for column in STRICT_BINDING_COLUMNS:
        if column in source.columns and not source[column].astype(str).reset_index(
            drop=True
        ).equals(matched[column].astype(str)):
            raise RuntimeError(f"strict cross-fit binding changed: {column}")

    output = source.copy()
    source_alpha = pd.to_numeric(output["allocation_alpha"], errors="raise").to_numpy(float)
    source_energy = pd.to_numeric(
        output["allocation_total_feature_energy"], errors="raise"
    ).to_numpy(float)
    output["allocation_policy"] = ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
    output["allocation_alpha"] = FIXED_SYMMETRIC_BIEND_ALPHA

    replacements = {
        "allocation_total_feature_energy": "total_feature_energy",
        "allocation_incident_edge_energy": "incident_edge_energy",
        "allocation_endpoint_node_energy": "endpoint_node_energy",
        "allocation_non_target_incident_edges": "non_target_incident_edges",
        "graybox_shadow_fold": "shadow_fold",
        "graybox_clean_probability_mean": "clean_probability_mean",
        "graybox_triggered_probability_mean": "triggered_probability_mean",
        "graybox_probability_delta_mean": "delta_mean",
        "graybox_probability_delta_std": "delta_std",
        "graybox_response_utility": "response_utility",
        "graybox_bce_informative_utility": "bce_informative_utility",
        "graybox_selection_utility": "selection_utility",
        "graybox_gradient_alignment_mean": "gradient_alignment_mean",
        "graybox_gradient_alignment_std": "gradient_alignment_std",
        "graybox_gradient_alignment_robust": "gradient_alignment_robust",
        "graybox_gradient_effect_norm_mean": "gradient_effect_norm_mean",
        "graybox_target_probe_count_min": "target_probe_count_min",
    }
    for destination, source_column in replacements.items():
        if destination not in output.columns:
            raise ValueError(f"source manifest is missing output column: {destination}")
        output[destination] = matched[source_column].to_numpy()
    output["graybox_selection_objective"] = SELECTION_OBJECTIVE
    output["v6_source_allocation_alpha"] = source_alpha
    output["v6_fixed_allocation_alpha"] = FIXED_SYMMETRIC_BIEND_ALPHA
    output["v6_source_pair_preserved"] = True
    within_budget = _strict_bool(
        matched["within_pair_feature_budget"],
        name="within_pair_feature_budget",
    )
    output["v6_within_source_pair_feature_budget"] = within_budget

    fixed_energy = pd.to_numeric(
        matched["total_feature_energy"], errors="raise"
    ).to_numpy(float)
    energy_ratio = np.divide(
        fixed_energy,
        source_energy,
        out=np.full(fixed_energy.shape, np.inf, dtype=float),
        where=source_energy > 0.0,
    )
    audit = pd.DataFrame(
        {
            "scenario_id": output["scenario_id"].astype(str),
            "src": output["src"].astype(np.int64),
            "dst": output["dst"].astype(np.int64),
            "src_track_id": output["src_track_id"].astype(str),
            "dst_track_id": output["dst_track_id"].astype(str),
            "source_allocation_alpha": source_alpha,
            "fixed_allocation_alpha": FIXED_SYMMETRIC_BIEND_ALPHA,
            "source_total_feature_energy": source_energy,
            "fixed_total_feature_energy": fixed_energy,
            "fixed_to_source_energy_ratio": energy_ratio,
            "within_source_pair_feature_budget": within_budget,
            "scenario_shadow_fold": output["scenario_shadow_fold"].astype(np.int64),
            "surrogate_checkpoint_sha256": output[
                "surrogate_checkpoint_sha256"
            ].astype(str),
        }
    )
    summary = {
        "rows": int(len(output)),
        "same_pair_preserved_rows": int(output["v6_source_pair_preserved"].sum()),
        "fixed_alpha": FIXED_SYMMETRIC_BIEND_ALPHA,
        "source_alpha_counts": {
            str(key): int(value)
            for key, value in sorted(Counter(source_alpha).items())
        },
        "within_source_pair_feature_budget_rows": int(within_budget.sum()),
        "outside_source_pair_feature_budget_rows": int((~within_budget).sum()),
        "outside_source_pair_feature_budget_rate": float((~within_budget).mean()),
        "fixed_total_feature_energy_mean": float(fixed_energy.mean()),
        "fixed_to_source_energy_ratio_median": float(np.median(energy_ratio)),
        "fixed_to_source_energy_ratio_p95": float(np.quantile(energy_ratio, 0.95)),
        "fixed_to_source_energy_ratio_max": float(np.max(energy_ratio)),
    }
    return output, audit, summary


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_manifest)
    candidates_path = Path(args.source_candidates)
    source_metadata_path = Path(args.source_metadata)
    output_path = Path(args.output)
    output_metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    audit_path = output_path.with_suffix(".fixed_alpha_audit.csv")
    existing = [
        path
        for path in (output_path, output_metadata_path, audit_path)
        if path.exists()
    ]
    if existing and not args.force:
        raise FileExistsError("v6.0 output exists; use a new path")

    source_metadata = _load_json(source_metadata_path)
    if sha256_file(source_path) != str(source_metadata.get("manifest_sha256", "")):
        raise RuntimeError("source manifest SHA-256 does not match metadata")
    if sha256_file(candidates_path) != str(
        source_metadata.get("candidate_scores_sha256", "")
    ):
        raise RuntimeError("source candidate SHA-256 does not match metadata")
    if source_metadata.get("strict_crossfit_required") is not True:
        raise RuntimeError("v6.0 source must be a strict cross-fit manifest")
    if str(source_metadata.get("selection_objective")) != "gradient_influence_v4_2":
        raise RuntimeError("v6.0 source must be the frozen v5 C-3 selection")

    label_config = RiskLabelConfig(**source_metadata["label_config"])
    source, source_hash = load_poison_manifest(
        source_path,
        label_config,
        expected_split="train",
        require_strict_label=True,
        require_metadata_binding=True,
        expected_graph_manifest_sha256=str(
            source_metadata["graph_manifest_sha256"]
        ),
        require_strict_crossfit_binding=True,
    )
    candidates = pd.read_csv(
        candidates_path,
        dtype={
            "scenario_id": str,
            "src_track_id": str,
            "dst_track_id": str,
        },
    )
    output, audit, summary = build_fixed_symmetric_manifest(source, candidates)
    strict_audit = validate_strict_crossfit_bindings(
        output,
        source_metadata,
        expected_graph_manifest_sha256=str(
            source_metadata["graph_manifest_sha256"]
        ),
        metadata_owner=source_metadata_path,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output, output_path)
    _atomic_csv(audit, audit_path)
    metadata = {
        "scope": "offline_authorized_av2_model_robustness",
        "experiment": VERSION,
        "status": "frozen_train_manifest",
        "development_only": True,
        "formal_test_eligible": False,
        "test_access_authorized": False,
        "training_manifest_eligible": True,
        "split": "train",
        "selection_authority": (
            "frozen_v5_gradient_selected_pair_with_train_only_fixed_alpha_lookup"
        ),
        "selection_objective": SELECTION_OBJECTIVE,
        "selection_utility_formula": (
            "pair inherited from v5 gradient_influence_v4_2; alpha fixed to 0.5"
        ),
        "same_pair_single_variable_change": True,
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": source_hash,
        "source_metadata_path": str(source_metadata_path),
        "source_metadata_sha256": sha256_file(source_metadata_path),
        "source_candidate_scores_path": str(candidates_path),
        "source_candidate_scores_sha256": sha256_file(candidates_path),
        "base_manifest_path": source_metadata["base_manifest_path"],
        "base_manifest_sha256": source_metadata["base_manifest_sha256"],
        "base_manifest_rows": int(source_metadata["base_manifest_rows"]),
        "scenario_ids_preserved": True,
        "unordered_pairs_preserved": True,
        "poison_labels_preserved": True,
        "poison_rate_preserved": True,
        "requested_poison_scenario_rate": float(
            source_metadata["requested_poison_scenario_rate"]
        ),
        "eligible_scenarios": int(source_metadata["eligible_scenarios"]),
        "poisoned_scenarios": int(len(output)),
        "pair_changed_rows": int(source_metadata["pair_changed_rows"]),
        "pair_changed_rate": float(source_metadata["pair_changed_rate"]),
        "orientation_policy": source_metadata["orientation_policy"],
        "allocation_policy": ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
        "allocation_alpha_grid": [FIXED_SYMMETRIC_BIEND_ALPHA],
        "alpha_counts": {str(FIXED_SYMMETRIC_BIEND_ALPHA): int(len(output))},
        "fixed_allocation_alpha": FIXED_SYMMETRIC_BIEND_ALPHA,
        "feature_budget_ratio_reference": float(
            source_metadata["feature_budget_ratio"]
        ),
        "feature_budget_status": (
            "fixed_alpha_override_reported_against_v5_pair_budget"
        ),
        "fixed_alpha_summary": summary,
        "require_strict_label": bool(source_metadata["require_strict_label"]),
        "label_unit": source_metadata["label_unit"],
        "label_config": source_metadata["label_config"],
        "label_config_hash": source_metadata["label_config_hash"],
        "graph_manifest_path": source_metadata["graph_manifest_path"],
        "graph_manifest_sha256": source_metadata["graph_manifest_sha256"],
        "graph_manifest_contract": source_metadata["graph_manifest_contract"],
        "shadow_fold_manifest": source_metadata["shadow_fold_manifest"],
        "strict_crossfit_required": True,
        "strict_crossfit_audit": strict_audit,
        "surrogates": source_metadata["surrogates"],
        "models_per_fold": source_metadata["models_per_fold"],
        "victim_queries": 0,
        "victim_parameters_or_gradients_used": False,
        "original_validation_used": False,
        "original_test_used": False,
        "forbidden_information_used": [],
        "fixed_alpha_audit_path": str(audit_path),
        "fixed_alpha_audit_sha256": sha256_file(audit_path),
        "fixed_alpha_audit_rows": int(len(audit)),
        "manifest_path": str(output_path),
        "manifest_sha256": sha256_file(output_path),
    }
    validate_strict_crossfit_bindings(
        output,
        metadata,
        expected_graph_manifest_sha256=str(metadata["graph_manifest_sha256"]),
        metadata_owner=output_metadata_path,
    )
    _atomic_json(metadata, output_metadata_path)

    # Exercise the exact training entry validator after the sidecar exists.
    load_poison_manifest(
        output_path,
        label_config,
        expected_split="train",
        require_strict_label=True,
        require_metadata_binding=True,
        expected_graph_manifest_sha256=str(metadata["graph_manifest_sha256"]),
        require_strict_crossfit_binding=True,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
