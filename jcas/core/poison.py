#!/usr/bin/env python3
"""Data-only utilities for the zero-query poisoning experiment.

Target eligibility and selection use graph fields and ground-truth labels only.
This module contains no victim-model scoring, probability, gradient, threshold,
checkpoint-selection, teacher, replay, or validation/test target-selection code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jcas.core.risk_labels import (
    RiskLabelConfig,
    label_config_hash,
    labels_for_graph,
    selected_label_computable_mask,
)


POISON_MANIFEST_COLUMNS = (
    "scenario_id",
    "split",
    "src",
    "dst",
    "src_track_id",
    "dst_track_id",
    "displacement_m",
    "perturb_window",
    "ramp_style",
    "velocity_mode",
    "poison_label",
    "seed",
    "label_mode",
    "label_config_hash",
    "require_strict_label",
    "label_unit",
)

PAIR_LABEL_UNIT = "unordered_pair_both_directions_v1"
ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED = (
    "lower_destination_mean_speed_v1"
)
ORIENTATION_POLICIES = (
    ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
)
ALLOCATION_POLICY_SINGLE_DESTINATION_V1 = "single_destination_v1"
ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2 = (
    "min_incident_feature_energy_v2"
)
ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1 = (
    "fixed_symmetric_biend_v1"
)
ALLOCATION_POLICY_CROSSFIT_SURROGATE_PAIR_ALPHA_V4 = (
    "crossfit_surrogate_pair_alpha_v4"
)
ALLOCATION_POLICIES = (
    ALLOCATION_POLICY_SINGLE_DESTINATION_V1,
    ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2,
    ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
    ALLOCATION_POLICY_CROSSFIT_SURROGATE_PAIR_ALPHA_V4,
)
ALLOCATION_ALPHA_GRID_V2 = (0.0, 0.25, 0.5, 0.75, 1.0)
FIXED_SYMMETRIC_BIEND_ALPHA = 0.5
ALLOCATION_AUDIT_COLUMNS = (
    "allocation_policy",
    "allocation_alpha",
    "allocation_total_feature_energy",
    "allocation_incident_edge_energy",
    "allocation_endpoint_node_energy",
    "allocation_candidate_count",
    "allocation_non_target_incident_edges",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supervision_mask(graph: Any, require_strict_label: bool = False) -> np.ndarray:
    edge_count = int(np.asarray(graph["edge_index"]).shape[1])
    mask = np.asarray(graph["supervision_edge_mask"], dtype=bool).copy()
    if mask.shape != (edge_count,):
        raise ValueError("supervision_edge_mask shape does not match edge_index")
    if "label_computable_mask" in graph:
        mask &= np.asarray(graph["label_computable_mask"], dtype=bool)
    if require_strict_label:
        mask &= np.asarray(graph["label_strict_mask"], dtype=bool)
    return mask


def trigger_contiguous_mask(graph: Any, perturb_window: int = 10) -> np.ndarray:
    """Edges whose target node has all K+1 observed frames needed by the trigger."""
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    observed_valid = np.asarray(graph["observed_valid_mask"], dtype=bool)
    if perturb_window < 1 or perturb_window >= observed_valid.shape[1]:
        raise ValueError("perturb_window is outside the observed history")
    src = edge_index[0]
    dst = edge_index[1]
    destination_window = observed_valid[dst, -(int(perturb_window) + 1) :].all(axis=1)
    pair_current = observed_valid[src, -1] & observed_valid[dst, -1]
    return destination_window & pair_current


def eligible_negative_edge_mask(
    graph: Any,
    label_config: RiskLabelConfig,
    *,
    require_strict_label: bool = False,
    perturb_window: int = 10,
    label_bundle: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Training/test eligibility defined entirely by stored data and true labels."""
    label_bundle = (
        labels_for_graph(graph, label_config)
        if label_bundle is None
        else label_bundle
    )
    labels = label_bundle["edge_label"].astype(np.int8)
    edge_count = int(labels.size)
    supervised = supervision_mask(graph, require_strict_label)
    supervised &= selected_label_computable_mask(
        graph, label_config, label_bundle
    )
    if "physical_possible_mask" in graph:
        physical = np.asarray(graph["physical_possible_mask"], dtype=bool)
    elif "message_passing_edge_mask" in graph:
        physical = np.asarray(graph["message_passing_edge_mask"], dtype=bool)
    else:
        physical = np.ones(edge_count, dtype=bool)
    if physical.shape != (edge_count,):
        raise ValueError("physical_possible_mask shape does not match labels")
    return (
        supervised
        & physical
        & trigger_contiguous_mask(graph, perturb_window)
        & (labels == 0)
    )


def reverse_edge_ids(edge_index: np.ndarray) -> np.ndarray:
    """Return the reverse directed-edge ID, or -1 when it is absent."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    lookup: dict[tuple[int, int], int] = {}
    for edge_id, (src, dst) in enumerate(edge_index.T.tolist()):
        key = (int(src), int(dst))
        if key in lookup:
            raise ValueError(f"graph contains duplicate directed edge {key}")
        lookup[key] = int(edge_id)
    return np.asarray(
        [lookup.get((int(dst), int(src)), -1) for src, dst in edge_index.T],
        dtype=np.int64,
    )


def eligible_negative_pair_groups(
    graph: Any,
    label_config: RiskLabelConfig,
    *,
    require_strict_label: bool = False,
    perturb_window: int = 10,
    require_bi_endpoint_contiguous: bool = False,
    label_bundle: dict[str, np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Eligible unordered pairs with one or two valid trigger orientations.

    The risk labels are symmetric in the two vehicles.  A candidate orientation
    is retained only when its reverse directed edge exists and both directions
    are supervised, physical, computable true negatives.  K-frame continuity
    is required only for the destination node of the orientation that will be
    moved unless ``require_bi_endpoint_contiguous`` is enabled.  The latter
    aligns pair sampling with fixed symmetric transforms that move both
    endpoints, while the default preserves the single-destination contract.
    """
    label_bundle = (
        labels_for_graph(graph, label_config)
        if label_bundle is None
        else label_bundle
    )
    labels = label_bundle["edge_label"].astype(np.int8)
    oriented = eligible_negative_edge_mask(
        graph,
        label_config,
        require_strict_label=require_strict_label,
        perturb_window=perturb_window,
        label_bundle=label_bundle,
    )
    supervised = supervision_mask(graph, require_strict_label)
    supervised &= selected_label_computable_mask(
        graph, label_config, label_bundle
    )
    if "physical_possible_mask" in graph:
        physical = np.asarray(graph["physical_possible_mask"], dtype=bool)
    elif "message_passing_edge_mask" in graph:
        physical = np.asarray(graph["message_passing_edge_mask"], dtype=bool)
    else:
        physical = np.ones(labels.shape, dtype=bool)
    reverse = reverse_edge_ids(np.asarray(graph["edge_index"], dtype=np.int64))
    reverse_ok = np.zeros(labels.shape, dtype=bool)
    has_reverse = reverse >= 0
    reverse_ok[has_reverse] = (
        supervised[reverse[has_reverse]]
        & physical[reverse[has_reverse]]
        & (labels[reverse[has_reverse]] == 0)
    )
    oriented &= reverse_ok

    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    if require_bi_endpoint_contiguous:
        observed_valid = np.asarray(graph["observed_valid_mask"], dtype=bool)
        required = int(perturb_window) + 1
        if required > observed_valid.shape[1]:
            raise ValueError("perturb_window is outside the observed history")
        endpoint_contiguous = observed_valid[:, -required:].all(axis=1)
        oriented &= (
            endpoint_contiguous[edge_index[0]]
            & endpoint_contiguous[edge_index[1]]
        )

    grouped: dict[tuple[int, int], list[int]] = {}
    for edge_id in np.flatnonzero(oriented):
        src = int(edge_index[0, edge_id])
        dst = int(edge_index[1, edge_id])
        key = (min(src, dst), max(src, dst))
        grouped.setdefault(key, []).append(int(edge_id))
    return [
        np.asarray(sorted(edge_ids), dtype=np.int64)
        for _key, edge_ids in sorted(grouped.items())
    ]


def _node_mean_speed_mps(
    graph: Any, node_id: int, perturb_window: int = 10
) -> float:
    """Mean observed speed over the K+1 states used by the trajectory transform."""
    velocities = np.asarray(graph["observed_velocities_filled"], dtype=np.float64)
    valid = np.asarray(graph["observed_valid_mask"], dtype=bool)
    if velocities.ndim != 3 or velocities.shape[2] != 2:
        raise ValueError("observed_velocities_filled must have shape [N, T, 2]")
    if valid.shape != velocities.shape[:2]:
        raise ValueError(
            "observed_valid_mask shape does not match observed_velocities_filled"
        )
    if perturb_window < 1 or perturb_window >= velocities.shape[1]:
        raise ValueError("perturb_window is outside the observed history")
    node_id = int(node_id)
    if node_id < 0 or node_id >= velocities.shape[0]:
        raise ValueError("node_id is outside observed_velocities_filled")
    window = slice(-(int(perturb_window) + 1), None)
    if not bool(valid[node_id, window].all()):
        raise ValueError(
            "orientation destination lacks the K+1 continuous observed states"
        )
    speed = np.linalg.norm(velocities[node_id, window], axis=1)
    if not np.isfinite(speed).all():
        raise ValueError("orientation destination speed contains non-finite values")
    return float(speed.mean())


def select_pair_orientation(
    graph: Any,
    orientation_edge_ids: np.ndarray,
    *,
    policy: str = ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
    perturb_window: int = 10,
) -> tuple[int, dict[str, Any]]:
    """Select which endpoint of one fixed unordered pair will be moved.

    This helper never chooses the unordered pair itself.  The data-aware policy
    uses only stored observed velocities and deterministic track-ID tie breaks;
    it does not use a model, probability, gradient, threshold, or split metric.
    """
    if policy not in ORIENTATION_POLICIES:
        raise ValueError(
            f"unknown orientation policy {policy!r}; expected one of "
            f"{ORIENTATION_POLICIES}"
        )
    edge_ids = np.asarray(orientation_edge_ids, dtype=np.int64)
    if edge_ids.ndim != 1 or edge_ids.size == 0:
        raise ValueError("orientation_edge_ids must be a non-empty vector")
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    if np.any(edge_ids < 0) or np.any(edge_ids >= edge_index.shape[1]):
        raise ValueError("orientation edge ID is outside edge_index")

    candidates: list[dict[str, Any]] = []
    track_ids = np.asarray(graph["node_track_ids"]).astype(str)
    for edge_id in edge_ids.tolist():
        src = int(edge_index[0, edge_id])
        dst = int(edge_index[1, edge_id])
        candidates.append(
            {
                "edge_id": int(edge_id),
                "src": src,
                "dst": dst,
                "src_track_id": str(track_ids[src]),
                "dst_track_id": str(track_ids[dst]),
                "destination_mean_speed_mps": _node_mean_speed_mps(
                    graph, dst, perturb_window
                ),
            }
        )

    selected = min(
        candidates,
        key=lambda item: (
            float(item["destination_mean_speed_mps"]),
            str(item["dst_track_id"]),
            str(item["src_track_id"]),
            int(item["edge_id"]),
        ),
    )

    audit = dict(selected)
    audit.update(
        {
            "orientation_policy": str(policy),
            "candidate_orientations": int(len(candidates)),
            "candidate_destination_mean_speeds_mps": [
                float(item["destination_mean_speed_mps"]) for item in candidates
            ],
        }
    )
    return int(selected["edge_id"]), audit


def _relative_feature_energy(
    clean_values: np.ndarray, triggered_values: np.ndarray
) -> float:
    """Scale-local squared feature energy without fitted model statistics."""
    clean = np.asarray(clean_values, dtype=np.float64)
    triggered = np.asarray(triggered_values, dtype=np.float64)
    if clean.shape != triggered.shape:
        raise ValueError("clean and triggered feature arrays must match")
    if clean.size == 0:
        return 0.0
    scale = np.maximum(np.abs(clean), 1.0)
    normalized_squared = np.square((triggered - clean) / scale)
    if normalized_squared.ndim == 1:
        return float(np.mean(normalized_squared))
    return float(np.sum(np.mean(normalized_squared, axis=-1)))


def select_bi_endpoint_allocation(
    graph: Any,
    *,
    src: int,
    dst: int,
    displacement_m: float = 0.2,
    alpha_grid: tuple[float, ...] = ALLOCATION_ALPHA_GRID_V2,
) -> tuple[float, dict[str, Any]]:
    """Choose alpha using only induced non-target feature changes.

    The unordered pair and its low-speed orientation are already frozen.  This
    function never reads a model, score, probability, threshold, or split
    metric.  Alpha zero is the v1 destination-only transform.
    """
    from jcas.core.trajectory_trigger import TriggerSpec, apply_trajectory_trigger

    src = int(src)
    dst = int(dst)
    grid = tuple(float(value) for value in alpha_grid)
    if not grid or any(
        not np.isfinite(value) or value < 0.0 or value > 1.0
        for value in grid
    ):
        raise ValueError("alpha_grid must contain finite values within [0, 1]")
    if len(set(grid)) != len(grid):
        raise ValueError("alpha_grid values must be unique")
    if 0.0 not in grid:
        raise ValueError("alpha_grid must include the frozen v1 alpha=0 fallback")

    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    clean_edge_attr = np.asarray(graph["edge_attr"], dtype=np.float32)
    clean_x_node = np.asarray(graph["x_node"], dtype=np.float32)
    valid = np.asarray(graph["observed_valid_mask"], dtype=bool)
    source_contiguous = bool(valid[src, -11:].all())
    pair_mask = (
        ((edge_index[0] == src) & (edge_index[1] == dst))
        | ((edge_index[0] == dst) & (edge_index[1] == src))
    )
    endpoint_incident = (
        (edge_index[0] == src)
        | (edge_index[1] == src)
        | (edge_index[0] == dst)
        | (edge_index[1] == dst)
    )
    non_target_incident = endpoint_incident & ~pair_mask
    endpoint_ids = np.asarray([src, dst], dtype=np.int64)
    spec = TriggerSpec()
    candidates: list[dict[str, Any]] = []
    for alpha in grid:
        if alpha > 0.0 and not source_contiguous:
            continue
        x_node, edge_attr, _target_mask, _audit = apply_trajectory_trigger(
            graph,
            src=src,
            dst=dst,
            displacement_m=float(displacement_m),
            allocation_alpha=float(alpha),
            spec=spec,
        )
        incident_energy = _relative_feature_energy(
            clean_edge_attr[non_target_incident],
            edge_attr[non_target_incident],
        )
        node_energy = _relative_feature_energy(
            clean_x_node[endpoint_ids], x_node[endpoint_ids]
        )
        candidates.append(
            {
                "alpha": float(alpha),
                "total_feature_energy": float(
                    incident_energy + node_energy
                ),
                "incident_edge_energy": float(incident_energy),
                "endpoint_node_energy": float(node_energy),
            }
        )
    if not candidates:
        raise RuntimeError("no valid bi-endpoint allocation candidate")
    selected = min(
        candidates,
        key=lambda item: (
            float(item["total_feature_energy"]),
            float(item["incident_edge_energy"]),
            float(item["endpoint_node_energy"]),
            float(item["alpha"]),
        ),
    )
    audit = {
        "allocation_policy": ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2,
        "allocation_alpha": float(selected["alpha"]),
        "allocation_total_feature_energy": float(
            selected["total_feature_energy"]
        ),
        "allocation_incident_edge_energy": float(
            selected["incident_edge_energy"]
        ),
        "allocation_endpoint_node_energy": float(
            selected["endpoint_node_energy"]
        ),
        "allocation_candidate_count": int(len(candidates)),
        "allocation_non_target_incident_edges": int(
            non_target_incident.sum()
        ),
        "allocation_source_contiguous": source_contiguous,
        "allocation_alpha_grid": [float(value) for value in grid],
        "allocation_candidate_scores": candidates,
        "selection_uses_model_output": False,
    }
    return float(selected["alpha"]), audit


def fixed_symmetric_bi_endpoint_allocation(
    graph: Any,
    *,
    src: int,
    dst: int,
    displacement_m: float = 0.2,
) -> tuple[float, dict[str, Any]]:
    """Return the model-independent equal endpoint allocation.

    The relative displacement is split equally between the two endpoints.
    Candidate feature energies are computed with the same data-only audit as
    ``min_incident_feature_energy_v2`` so the stealth cost remains directly
    comparable, but no candidate score is used to choose alpha.
    """
    _selected_alpha, candidate_audit = select_bi_endpoint_allocation(
        graph,
        src=int(src),
        dst=int(dst),
        displacement_m=float(displacement_m),
    )
    candidates = candidate_audit["allocation_candidate_scores"]
    matches = [
        item
        for item in candidates
        if np.isclose(
            float(item["alpha"]),
            FIXED_SYMMETRIC_BIEND_ALPHA,
            atol=1e-12,
            rtol=0.0,
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "fixed symmetric allocation requires K+1 contiguous frames "
            "for both endpoints"
        )
    selected = matches[0]
    audit = {
        **candidate_audit,
        "allocation_policy": ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
        "allocation_alpha": FIXED_SYMMETRIC_BIEND_ALPHA,
        "allocation_total_feature_energy": float(
            selected["total_feature_energy"]
        ),
        "allocation_incident_edge_energy": float(
            selected["incident_edge_energy"]
        ),
        "allocation_endpoint_node_energy": float(
            selected["endpoint_node_energy"]
        ),
        "allocation_fixed_by_rule": True,
        "selection_uses_model_output": False,
    }
    return FIXED_SYMMETRIC_BIEND_ALPHA, audit


def reverse_edge_id(graph: Any, src: int, dst: int) -> int:
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    matches = np.flatnonzero(
        (edge_index[0] == int(dst)) & (edge_index[1] == int(src))
    )
    if matches.size != 1:
        raise ValueError("target pair does not have one unique reverse edge")
    return int(matches[0])


def _strict_integer_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"poison manifest column {column!r} must contain integers")
    return values.astype(np.int64)


def _strict_boolean_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    mapping = {
        "1": True,
        "true": True,
        "yes": True,
        "0": False,
        "false": False,
        "no": False,
    }
    values: list[bool] = []
    for raw in frame[column].tolist():
        key = str(raw).strip().lower()
        if key not in mapping:
            raise ValueError(
                f"poison manifest column {column!r} contains invalid boolean {raw!r}"
            )
        values.append(mapping[key])
    return np.asarray(values, dtype=bool)


STRICT_CROSSFIT_BINDING_COLUMNS = (
    "scenario_shadow_fold",
    "surrogate_heldout_fold",
    "surrogate_checkpoint_sha256",
    "surrogate_fit_manifest_sha256",
    "surrogate_score_manifest_sha256",
    "surrogate_protocol",
)


def validate_strict_crossfit_bindings(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    expected_graph_manifest_sha256: str,
    metadata_owner: Path,
) -> dict[str, Any]:
    """Validate per-scenario strict surrogate provenance without model access."""
    from jcas.core.shadow_folds import load_shadow_fold_manifest
    from jcas.core.strict_shadow_folds import (
        STRICT_CROSSFIT_PROTOCOL,
        load_strict_crossfit_contract,
        load_strict_crossfit_release,
    )

    missing = sorted(set(STRICT_CROSSFIT_BINDING_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"strict cross-fit manifest is missing bindings: {missing}")
    scenario_fold = _strict_integer_column(frame, "scenario_shadow_fold")
    heldout_fold = _strict_integer_column(frame, "surrogate_heldout_fold")
    if np.any((scenario_fold < 0) | (scenario_fold > 2)) or not np.array_equal(
        scenario_fold, heldout_fold
    ):
        raise ValueError("scenario and surrogate held-out folds do not match")
    if not frame["surrogate_protocol"].astype(str).eq(
        STRICT_CROSSFIT_PROTOCOL
    ).all():
        raise ValueError("manifest contains a non-strict surrogate protocol")
    for column in (
        "surrogate_checkpoint_sha256",
        "surrogate_fit_manifest_sha256",
        "surrogate_score_manifest_sha256",
    ):
        values = frame[column].astype(str).str.lower()
        if not values.str.fullmatch(r"[0-9a-f]{64}", na=False).all():
            raise ValueError(f"strict cross-fit binding {column} is invalid")
        if frame.assign(_value=values).groupby("scenario_id")["_value"].nunique().gt(1).any():
            raise ValueError(f"one scenario has multiple {column} bindings")

    surrogates = metadata.get("surrogates")
    if not isinstance(surrogates, list) or len(surrogates) != 3:
        raise ValueError("strict cross-fit metadata must bind exactly three surrogates")
    expected_by_fold: dict[int, dict[str, str]] = {}
    release_hashes: set[str] = set()
    for audit in surrogates:
        if not isinstance(audit, dict) or audit.get("strict_crossfit_verified") is not True:
            raise ValueError("strict cross-fit surrogate audit is incomplete")
        release = audit.get("strict_crossfit_release")
        if not isinstance(release, dict):
            raise ValueError("strict cross-fit surrogate lacks pretraining release")
        release_sha = str(release.get("sha256", "")).lower()
        if len(release_sha) != 64:
            raise ValueError("strict cross-fit surrogate release hash is invalid")
        release_hashes.add(release_sha)
        shadow = audit.get("shadow_protocol")
        if not isinstance(shadow, dict) or shadow.get(
            "surrogate_protocol"
        ) != STRICT_CROSSFIT_PROTOCOL:
            raise ValueError("strict cross-fit surrogate protocol is invalid")
        verified_release = load_strict_crossfit_release(
            str(release.get("path", "")),
            expected_release_sha256=release_sha,
            expected_graph_manifest_sha256=expected_graph_manifest_sha256,
            expected_contract_path=str(
                shadow.get("strict_contract_path", "")
            ),
        )
        if str(release.get("release_id", "")) != str(
            verified_release["release_id"]
        ):
            raise ValueError("strict cross-fit surrogate release ID mismatch")
        _fit, _score, verified_contract = load_strict_crossfit_contract(
            str(shadow.get("strict_contract_path", "")),
            expected_graph_manifest_sha256=expected_graph_manifest_sha256,
        )
        if str(shadow.get("strict_contract_sha256", "")).lower() != str(
            verified_contract["contract_sha256"]
        ).lower():
            raise RuntimeError("strict surrogate contract SHA-256 mismatch")
        if int(shadow.get("heldout_fold", -1)) != int(
            verified_contract["heldout_fold"]
        ):
            raise ValueError("strict surrogate contract held-out fold mismatch")
        for result_key, contract_key in (
            ("surrogate_fit_manifest_sha256", "fit_manifest_sha256"),
            ("surrogate_score_manifest_sha256", "score_manifest_sha256"),
        ):
            if str(shadow.get(result_key, "")).lower() != str(
                verified_contract[contract_key]
            ).lower():
                raise RuntimeError(f"strict surrogate {result_key} mismatch")
        fold = int(shadow.get("heldout_fold", -1))
        if fold in expected_by_fold or fold < 0 or fold > 2:
            raise ValueError("strict cross-fit surrogate fold coverage is invalid")
        expected_by_fold[fold] = {
            "surrogate_checkpoint_sha256": str(
                audit.get("checkpoint_sha256", "")
            ).lower(),
            "surrogate_fit_manifest_sha256": str(
                shadow.get("surrogate_fit_manifest_sha256", "")
            ).lower(),
            "surrogate_score_manifest_sha256": str(
                shadow.get("surrogate_score_manifest_sha256", "")
            ).lower(),
        }
    if set(expected_by_fold) != {0, 1, 2}:
        raise ValueError("strict cross-fit metadata does not cover all folds")
    if len(release_hashes) != 1:
        raise ValueError("strict surrogates do not share one pretraining release")
    checkpoints = {
        binding["surrogate_checkpoint_sha256"]
        for binding in expected_by_fold.values()
    }
    if len(checkpoints) != 3 or any(
        len(value) != 64 for value in checkpoints
    ):
        raise ValueError("strict cross-fit checkpoints are not three distinct hashes")
    for fold, expected in expected_by_fold.items():
        subset = frame[scenario_fold == fold]
        if subset.empty:
            raise ValueError(f"strict cross-fit manifest contains no fold {fold} rows")
        for column, value in expected.items():
            if not subset[column].astype(str).str.lower().eq(value).all():
                raise RuntimeError(
                    f"strict cross-fit rows do not match fold {fold} {column}"
                )

    models_per_fold = metadata.get("models_per_fold")
    if not isinstance(models_per_fold, dict) or {
        str(key): int(value) for key, value in models_per_fold.items()
    } != {"0": 1, "1": 1, "2": 1}:
        raise ValueError("strict cross-fitting requires one surrogate per fold")
    outer_record = metadata.get("shadow_fold_manifest")
    if not isinstance(outer_record, dict):
        raise ValueError("strict cross-fit metadata lacks outer fold binding")
    recorded_path = Path(str(outer_record.get("path", "")))
    candidates = (recorded_path, metadata_owner.parent / recorded_path.name)
    outer_path = next((path for path in candidates if path.is_file()), None)
    if outer_path is None:
        raise FileNotFoundError("strict cross-fit outer fold manifest is missing")
    outer, outer_audit = load_shadow_fold_manifest(
        outer_path,
        expected_graph_manifest_sha256=expected_graph_manifest_sha256,
        expected_num_folds=3,
    )
    if str(outer_record.get("sha256", "")).lower() != str(
        outer_audit["sha256"]
    ).lower():
        raise RuntimeError("strict cross-fit outer fold SHA-256 mismatch")
    fold_lookup = outer.set_index("scenario_id")["shadow_fold"]
    recorded_scenarios = frame["scenario_id"].astype(str)
    expected_folds = recorded_scenarios.map(fold_lookup)
    if expected_folds.isna().any() or not np.array_equal(
        expected_folds.to_numpy(np.int64), scenario_fold
    ):
        raise ValueError("strict manifest scenario folds differ from outer folds")
    return {
        "surrogate_protocol": STRICT_CROSSFIT_PROTOCOL,
        "folds": [0, 1, 2],
        "distinct_surrogate_checkpoints": 3,
        "outer_fold_manifest_sha256": str(outer_audit["sha256"]),
        "pretraining_release_sha256": next(iter(release_hashes)),
        "scenario_bindings_verified": int(len(frame)),
    }


def load_poison_manifest(
    path: str | Path,
    label_config: RiskLabelConfig,
    *,
    expected_split: str = "train",
    require_strict_label: bool = False,
    require_metadata_binding: bool = False,
    expected_graph_manifest_sha256: str | None = None,
    require_strict_crossfit_binding: bool = False,
    expected_trigger_schedule: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load and strictly validate a frozen, model-independent poison manifest."""
    manifest_path = Path(path)
    frame = pd.read_csv(manifest_path, dtype={"scenario_id": str})
    missing = sorted(set(POISON_MANIFEST_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"poison manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("poison manifest is empty")
    if frame["scenario_id"].duplicated().any():
        raise ValueError("poison manifest must contain at most one target per scenario")
    if not (frame["split"].astype(str) == expected_split).all():
        raise ValueError(f"poison manifest may contain only {expected_split!r} rows")
    expected_hash = label_config_hash(label_config)
    if not (frame["label_mode"].astype(str) == label_config.label_mode).all():
        raise ValueError("poison manifest label_mode does not match training")
    if not (frame["label_config_hash"].astype(str) == expected_hash).all():
        raise ValueError("poison manifest label_config_hash does not match training")
    strict_values = _strict_boolean_column(frame, "require_strict_label")
    if not np.all(strict_values == bool(require_strict_label)):
        raise ValueError("poison manifest strict-label policy does not match training")
    if not (frame["label_unit"].astype(str) == PAIR_LABEL_UNIT).all():
        raise ValueError(
            "poison manifest must use symmetric unordered-pair labels"
        )

    perturb_window = _strict_integer_column(frame, "perturb_window")
    poison_label = _strict_integer_column(frame, "poison_label")
    if expected_trigger_schedule is None:
        perturb_window_valid = perturb_window == 10
    else:
        from jcas.core.motion_schedule_trigger import (
            MOTION_REGIME_K4_K10_SCHEDULE,
            scheduled_window,
        )

        if str(expected_trigger_schedule) != MOTION_REGIME_K4_K10_SCHEDULE:
            raise ValueError("unrecognized experimental trigger schedule")
        required_schedule_columns = {"trigger_schedule_id", "motion_regime"}
        missing_schedule_columns = sorted(
            required_schedule_columns - set(frame.columns)
        )
        if missing_schedule_columns:
            raise ValueError(
                "scheduled poison manifest is missing columns: "
                f"{missing_schedule_columns}"
            )
        if not (
            frame["trigger_schedule_id"].astype(str)
            == str(expected_trigger_schedule)
        ).all():
            raise ValueError("poison manifest trigger schedule ID changed")
        expected_windows = frame["motion_regime"].astype(str).map(
            scheduled_window
        ).to_numpy(np.int64)
        perturb_window_valid = perturb_window == expected_windows
    fixed_checks = {
        "displacement_m": np.isclose(
            frame["displacement_m"].to_numpy(float), 0.2, atol=1e-12, rtol=0.0
        ),
        "perturb_window": perturb_window_valid,
        "ramp_style": frame["ramp_style"].astype(str).to_numpy() == "minimum_jerk",
        "velocity_mode": frame["velocity_mode"].astype(str).to_numpy() == "residual",
        "poison_label": poison_label == 1,
    }
    failures = [name for name, valid in fixed_checks.items() if not np.all(valid)]
    if failures:
        raise ValueError(f"poison manifest violates frozen trigger fields: {failures}")
    for column in ("src", "dst", "seed"):
        values = _strict_integer_column(frame, column)
        if np.any(values < 0):
            raise ValueError(
                f"poison manifest column {column!r} must be non-negative"
            )
    orientation_policy: str | None = None
    if "orientation_policy" in frame.columns:
        policies = sorted(set(frame["orientation_policy"].astype(str)))
        if len(policies) != 1 or policies[0] not in ORIENTATION_POLICIES:
            raise ValueError(
                "poison manifest must contain one recognized orientation policy"
            )
        orientation_policy = policies[0]
    allocation_policy = ALLOCATION_POLICY_SINGLE_DESTINATION_V1
    allocation_columns = {
        "allocation_policy",
        "allocation_alpha",
    }
    present_allocation_columns = allocation_columns & set(frame.columns)
    if present_allocation_columns and present_allocation_columns != allocation_columns:
        raise ValueError(
            "poison manifest must contain allocation_policy and allocation_alpha together"
        )
    if present_allocation_columns:
        policies = sorted(set(frame["allocation_policy"].astype(str)))
        if len(policies) != 1 or policies[0] not in ALLOCATION_POLICIES:
            raise ValueError(
                "poison manifest must contain one recognized allocation policy"
            )
        allocation_policy = policies[0]
        allocation_alpha = pd.to_numeric(
            frame["allocation_alpha"], errors="coerce"
        ).to_numpy(np.float64)
        if not np.isfinite(allocation_alpha).all() or np.any(
            (allocation_alpha < 0.0) | (allocation_alpha > 1.0)
        ):
            raise ValueError(
                "poison manifest allocation_alpha must be finite and within [0, 1]"
            )
        if (
            allocation_policy == ALLOCATION_POLICY_SINGLE_DESTINATION_V1
            and not np.allclose(allocation_alpha, 0.0, atol=1e-12, rtol=0.0)
        ):
            raise ValueError(
                "single_destination_v1 requires allocation_alpha=0"
            )
        if (
            allocation_policy == ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
            and not np.allclose(
                allocation_alpha,
                FIXED_SYMMETRIC_BIEND_ALPHA,
                atol=1e-12,
                rtol=0.0,
            )
        ):
            raise ValueError(
                "fixed_symmetric_biend_v1 requires allocation_alpha=0.5"
            )
    manifest_hash = sha256_file(manifest_path)
    if require_strict_crossfit_binding and not require_metadata_binding:
        raise ValueError(
            "strict cross-fit binding requires poison metadata validation"
        )
    if require_metadata_binding:
        metadata_path = manifest_path.with_suffix(
            manifest_path.suffix + ".metadata.json"
        )
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"poison manifest metadata does not exist: {metadata_path}"
            )
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("training_manifest_eligible", True) is not True:
            raise RuntimeError(
                "diagnostic poison manifest is not eligible for training"
            )
        if str(metadata.get("manifest_sha256", "")).lower() != manifest_hash:
            raise RuntimeError(
                "poison manifest SHA-256 does not match its metadata sidecar"
            )
        if str(metadata.get("label_config_hash", "")) != expected_hash:
            raise RuntimeError(
                "poison manifest metadata label_config_hash does not match"
            )
        metadata_policy = metadata.get("orientation_policy")
        if orientation_policy is not None and str(
            metadata_policy
        ) != orientation_policy:
            raise RuntimeError(
                "poison manifest orientation policy does not match metadata"
            )
        metadata_allocation_policy = metadata.get(
            "allocation_policy",
            ALLOCATION_POLICY_SINGLE_DESTINATION_V1,
        )
        if str(metadata_allocation_policy) != allocation_policy:
            raise RuntimeError(
                "poison manifest allocation policy does not match metadata"
            )
        metadata_schedule = metadata.get("trigger_schedule_id")
        if expected_trigger_schedule is None:
            if metadata_schedule is not None:
                raise RuntimeError(
                    "scheduled poison manifest requires an explicit training flag"
                )
        elif str(metadata_schedule) != str(expected_trigger_schedule):
            raise RuntimeError(
                "poison manifest trigger schedule does not match metadata"
            )
        if not np.isclose(
            float(metadata.get("requested_poison_scenario_rate", np.nan)),
            0.05,
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                "main-line poison manifest must use the frozen 5% rate"
            )
        if expected_graph_manifest_sha256 is not None:
            recorded_graph_hash = str(
                metadata.get("graph_manifest_sha256", "")
            ).lower()
            if recorded_graph_hash != str(
                expected_graph_manifest_sha256
            ).lower():
                raise RuntimeError(
                    "poison manifest was generated from a different graph manifest"
                )
        strict_declared = metadata.get("strict_crossfit_required") is True
        if require_strict_crossfit_binding and not strict_declared:
            raise RuntimeError("v4.2 training requires a strict cross-fit manifest")
        if strict_declared:
            if expected_graph_manifest_sha256 is None:
                raise ValueError(
                    "strict cross-fit validation requires graph manifest binding"
                )
            validate_strict_crossfit_bindings(
                frame,
                metadata,
                expected_graph_manifest_sha256=expected_graph_manifest_sha256,
                metadata_owner=metadata_path,
            )
    return frame.sort_values("scenario_id").reset_index(drop=True), manifest_hash


def manifest_target_edge(
    graph: Any,
    row: Any,
    label_config: RiskLabelConfig,
    *,
    require_strict_label: bool = False,
) -> tuple[int, int, int, int]:
    """Resolve one frozen row and its reverse supervised true-negative edge."""
    src, dst = int(row.src), int(row.dst)
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    if src < 0 or dst < 0 or src >= len(graph["node_track_ids"]) or dst >= len(
        graph["node_track_ids"]
    ):
        raise ValueError("poison manifest node index is outside the graph")
    if str(graph["node_track_ids"][src]) != str(row.src_track_id):
        raise ValueError("poison manifest src_track_id does not match src")
    if str(graph["node_track_ids"][dst]) != str(row.dst_track_id):
        raise ValueError("poison manifest dst_track_id does not match dst")
    matches = np.flatnonzero((edge_index[0] == src) & (edge_index[1] == dst))
    if matches.size != 1:
        raise ValueError("poison manifest target is not a unique graph edge")
    edge_id = int(matches[0])
    eligible_groups = eligible_negative_pair_groups(
        graph,
        label_config,
        require_strict_label=require_strict_label,
        perturb_window=int(row.perturb_window),
        require_bi_endpoint_contiguous=(
            str(
                getattr(
                    row,
                    "allocation_policy",
                    ALLOCATION_POLICY_SINGLE_DESTINATION_V1,
                )
            )
            == ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
        ),
    )
    eligible_orientations = {
        int(value) for group in eligible_groups for value in group.tolist()
    }
    if edge_id not in eligible_orientations:
        raise ValueError(
            "poison manifest target is no longer an eligible symmetric pair"
        )
    reverse_id = reverse_edge_id(graph, src, dst)
    return src, dst, edge_id, reverse_id


def apply_manifest_row(
    graph: Any,
    row: Any,
    label_config: RiskLabelConfig,
    *,
    require_strict_label: bool = False,
    edge_feature_mode: str = "base_v3",
    experimental_trigger_schedule: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply one trajectory transform and label both directions of its pair."""
    # Imported lazily so manifest generation/selection has no model-side
    # dependency.
    src, dst, edge_id, reverse_id = manifest_target_edge(
        graph, row, label_config, require_strict_label=require_strict_label
    )
    if experimental_trigger_schedule is None:
        from jcas.core.trajectory_trigger import (
            TriggerSpec,
            apply_trajectory_trigger,
        )

        spec = TriggerSpec(
            perturb_window=int(row.perturb_window),
            ramp_style=str(row.ramp_style),
            velocity_mode=str(row.velocity_mode),
            require_contiguous_valid=True,
        )
        x_node, edge_attr, target_mask, audit = apply_trajectory_trigger(
            graph,
            src=src,
            dst=dst,
            displacement_m=float(row.displacement_m),
            allocation_alpha=float(getattr(row, "allocation_alpha", 0.0)),
            spec=spec,
            edge_feature_mode=edge_feature_mode,
        )
    else:
        from jcas.core.motion_schedule_trigger import (
            apply_scheduled_trajectory_trigger,
        )

        x_node, edge_attr, target_mask, audit = (
            apply_scheduled_trajectory_trigger(
                graph,
                src=src,
                dst=dst,
                displacement_m=float(row.displacement_m),
                allocation_alpha=float(
                    getattr(row, "allocation_alpha", 0.0)
                ),
                motion_regime=str(row.motion_regime),
                schedule_id=str(experimental_trigger_schedule),
                edge_feature_mode=edge_feature_mode,
            )
        )
    labels = labels_for_graph(graph, label_config)["edge_label"].copy()
    if int(target_mask.sum()) != 1 or not bool(target_mask[edge_id]):
        raise RuntimeError("frozen trigger transform did not apply to exactly one target edge")
    pair_target_mask = target_mask.copy()
    pair_target_mask[reverse_id] = True
    labels[pair_target_mask] = float(row.poison_label)
    audit["label_unit"] = PAIR_LABEL_UNIT
    audit["labeled_edge_ids"] = [int(edge_id), int(reverse_id)]
    audit["labeled_directed_edges"] = 2
    return x_node, edge_attr, labels, pair_target_mask, audit
