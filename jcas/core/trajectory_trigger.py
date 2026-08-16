"""Frozen K10 trajectory transform for the current robustness experiment.

This module contains only the data-level transform used by the dynamic-risk,
strict zero-query main line. It does not select targets, query a model, change
graph topology, or decide labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from jcas.workflows.graph_builder import (
    DT,
    OBS_LEN,
    compute_edge_features,
    make_node_feature,
)


EPS = 1e-6


@dataclass(frozen=True)
class TriggerSpec:
    perturb_window: int = 10
    ramp_style: str = "minimum_jerk"
    velocity_mode: str = "residual"
    require_contiguous_valid: bool = True
    dt_seconds: float = DT


def validate_trigger_spec(spec: TriggerSpec) -> TriggerSpec:
    """Fail closed if a caller deviates from the frozen main configuration."""
    if int(spec.perturb_window) != 10:
        raise ValueError("main-line perturb_window must equal 10")
    if str(spec.ramp_style) != "minimum_jerk":
        raise ValueError("main-line ramp_style must be minimum_jerk")
    if str(spec.velocity_mode) != "residual":
        raise ValueError("main-line velocity_mode must be residual")
    if spec.require_contiguous_valid is not True:
        raise ValueError("main-line trigger requires contiguous valid frames")
    if not np.isfinite(spec.dt_seconds) or float(spec.dt_seconds) <= 0.0:
        raise ValueError("dt_seconds must be finite and positive")
    return spec


def minimum_jerk_progress(spec: TriggerSpec) -> np.ndarray:
    spec = validate_trigger_spec(spec)
    window = int(spec.perturb_window)
    tau = np.arange(1, window + 1, dtype=np.float64) / float(window)
    progress = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    progress[-1] = 1.0
    if np.any(np.diff(progress) < -1e-12):
        raise RuntimeError("minimum-jerk progress must be monotone")
    return progress.astype(np.float32)


def _node_motion_statistics(
    x_node: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    headings: np.ndarray,
    valid: np.ndarray,
    updated_nodes: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_count = int(positions.shape[0])
    speed_current = np.zeros(node_count, dtype=np.float32)
    speed_mean = np.zeros(node_count, dtype=np.float32)
    acc_std = np.zeros(node_count, dtype=np.float32)
    lateral_acc_std = np.zeros(node_count, dtype=np.float32)
    for node_id in range(node_count):
        is_focal = bool(x_node[node_id, -1] > 0.5)
        feature, current, mean, acceleration, lateral = make_node_feature(
            positions[node_id],
            velocities[node_id],
            headings[node_id],
            valid[node_id],
            is_focal,
        )
        speed_current[node_id] = current
        speed_mean[node_id] = mean
        acc_std[node_id] = acceleration
        lateral_acc_std[node_id] = lateral
        if node_id in updated_nodes:
            x_node[node_id] = feature.astype(np.float32)
    return x_node, speed_current, speed_mean, acc_std, lateral_acc_std


def _node_trigger_audit(
    *,
    node_id: int,
    role: str,
    valid: np.ndarray,
    clean_positions: np.ndarray,
    triggered_positions: np.ndarray,
    clean_velocities: np.ndarray,
    triggered_velocities: np.ndarray,
    position_residual: np.ndarray,
    velocity_residual: np.ndarray,
    dt_seconds: float,
) -> dict[str, Any]:
    """Return motion diagnostics for one endpoint modified by the trigger."""
    valid_idx = np.flatnonzero(valid[node_id])
    clean_speed = np.linalg.norm(
        clean_velocities[node_id, valid_idx], axis=1
    )
    triggered_speed = np.linalg.norm(
        triggered_velocities[node_id, valid_idx], axis=1
    )
    induced_speed = np.linalg.norm(velocity_residual, axis=1)
    induced_acc = np.diff(velocity_residual, axis=0) / dt_seconds
    induced_jerk = np.diff(induced_acc, axis=0) / dt_seconds

    if valid_idx.size >= 2:
        valid_dt = np.maximum(
            np.diff(valid_idx).astype(np.float64) * dt_seconds, EPS
        )
        clean_acc = (
            np.diff(clean_velocities[node_id, valid_idx], axis=0)
            / valid_dt[:, None]
        )
        triggered_acc = (
            np.diff(triggered_velocities[node_id, valid_idx], axis=0)
            / valid_dt[:, None]
        )
        clean_consistency = (
            np.diff(clean_positions[node_id, valid_idx], axis=0)
            / valid_dt[:, None]
            - clean_velocities[node_id, valid_idx[1:]]
        )
        triggered_consistency = (
            np.diff(triggered_positions[node_id, valid_idx], axis=0)
            / valid_dt[:, None]
            - triggered_velocities[node_id, valid_idx[1:]]
        )
        consistency_change = triggered_consistency - clean_consistency
    else:
        clean_acc = np.zeros((0, 2), dtype=np.float32)
        triggered_acc = np.zeros((0, 2), dtype=np.float32)
        consistency_change = np.zeros((0, 2), dtype=np.float32)

    if valid_idx.size >= 3:
        acc_dt = np.maximum(
            np.diff(valid_idx[1:]).astype(np.float64) * dt_seconds, EPS
        )
        clean_jerk = np.diff(clean_acc, axis=0) / acc_dt[:, None]
        triggered_jerk = np.diff(triggered_acc, axis=0) / acc_dt[:, None]
    else:
        clean_jerk = np.zeros((0, 2), dtype=np.float32)
        triggered_jerk = np.zeros((0, 2), dtype=np.float32)

    return {
        # ``dst`` is retained for compatibility with the frozen v1 evaluator.
        "dst": int(node_id),
        "node_id": int(node_id),
        "role": str(role),
        "terminal_displacement_m": float(
            np.linalg.norm(position_residual[-1])
        ),
        "max_induced_speed_mps": float(np.max(induced_speed)),
        "max_induced_acc_mps2": float(
            np.max(np.linalg.norm(induced_acc, axis=1))
        ),
        "max_induced_jerk_mps3": float(
            np.max(np.linalg.norm(induced_jerk, axis=1))
        ),
        "max_clean_speed_mps": float(np.max(clean_speed)),
        "max_triggered_speed_mps": float(np.max(triggered_speed)),
        "max_clean_acc_mps2": (
            float(np.max(np.linalg.norm(clean_acc, axis=1)))
            if clean_acc.size
            else 0.0
        ),
        "max_triggered_acc_mps2": (
            float(np.max(np.linalg.norm(triggered_acc, axis=1)))
            if triggered_acc.size
            else 0.0
        ),
        "max_clean_jerk_mps3": (
            float(np.max(np.linalg.norm(clean_jerk, axis=1)))
            if clean_jerk.size
            else 0.0
        ),
        "max_triggered_jerk_mps3": (
            float(np.max(np.linalg.norm(triggered_jerk, axis=1)))
            if triggered_jerk.size
            else 0.0
        ),
        "max_position_velocity_residual_change_mps": (
            float(np.max(np.linalg.norm(consistency_change, axis=1)))
            if consistency_change.size
            else 0.0
        ),
    }


def apply_trajectory_trigger(
    graph: Any,
    *,
    src: int,
    dst: int,
    displacement_m: float,
    allocation_alpha: float = 0.0,
    spec: TriggerSpec | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Shorten one pair by ``displacement_m`` with a two-endpoint allocation.

    ``allocation_alpha`` is the fraction assigned to ``src``.  The source is
    moved toward ``dst`` and the destination is moved toward ``src``.  Alpha
    zero is exactly the frozen v1 single-destination transform.
    """
    spec = validate_trigger_spec(spec or TriggerSpec())
    src = int(src)
    dst = int(dst)
    displacement_m = float(displacement_m)
    allocation_alpha = float(allocation_alpha)
    if not np.isfinite(displacement_m) or displacement_m <= 0.0:
        raise ValueError("displacement_m must be finite and positive")
    if not np.isfinite(allocation_alpha) or not 0.0 <= allocation_alpha <= 1.0:
        raise ValueError("allocation_alpha must be finite and within [0, 1]")

    x_node = np.asarray(graph["x_node"], dtype=np.float32).copy()
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    edge_attr = np.asarray(graph["edge_attr"], dtype=np.float32).copy()
    positions = np.asarray(
        graph["observed_positions_filled"], dtype=np.float32
    ).copy()
    velocities = np.asarray(
        graph["observed_velocities_filled"], dtype=np.float32
    ).copy()
    headings = np.asarray(
        graph["observed_headings_filled"], dtype=np.float32
    )
    valid = np.asarray(graph["observed_valid_mask"], dtype=bool)
    node_count = int(positions.shape[0])
    if src < 0 or dst < 0 or src >= node_count or dst >= node_count:
        raise IndexError("trigger node index is outside the graph")
    if src == dst:
        raise ValueError("trigger source and destination must be different")
    if not bool(valid[src, -1] and valid[dst, -1]):
        raise ValueError("trigger pair is invalid at the final frame")

    window = int(spec.perturb_window)
    source_fraction = allocation_alpha
    destination_fraction = 1.0 - allocation_alpha
    if source_fraction > EPS and not bool(
        valid[src, -(window + 1) :].all()
    ):
        raise ValueError("trigger source lacks K+1 contiguous valid frames")
    if destination_fraction > EPS and not bool(
        valid[dst, -(window + 1) :].all()
    ):
        raise ValueError("trigger destination lacks K+1 contiguous valid frames")

    los = positions[src, -1] - positions[dst, -1]
    current_distance = float(np.linalg.norm(los))
    if current_distance <= EPS:
        raise ValueError("trigger pair has zero final-frame distance")
    applied_displacement = min(displacement_m, 0.9 * current_distance)
    base_offset = (
        applied_displacement / current_distance
    ) * los.astype(np.float32)

    clean_positions = positions.copy()
    clean_velocities = velocities.copy()
    progress = minimum_jerk_progress(spec)
    start = OBS_LEN - window
    residuals: dict[int, np.ndarray] = {}
    velocity_residuals: dict[int, np.ndarray] = {}
    roles: dict[int, str] = {}
    plans = (
        (src, source_fraction, -base_offset, "source"),
        (dst, destination_fraction, base_offset, "destination"),
    )
    for node_id, fraction, direction, role in plans:
        if fraction <= EPS:
            continue
        residual = np.zeros((OBS_LEN, 2), dtype=np.float32)
        residual[-window:] = (
            progress[:, None]
            * (float(fraction) * direction)[None, :]
        )
        velocity_residual = np.zeros_like(residual)
        for timestep in range(start, OBS_LEN):
            velocity_residual[timestep] = (
                residual[timestep] - residual[timestep - 1]
            ) / float(spec.dt_seconds)
        positions[node_id] += residual
        velocities[node_id] += velocity_residual
        residuals[int(node_id)] = residual
        velocity_residuals[int(node_id)] = velocity_residual
        roles[int(node_id)] = role

    updated_nodes = set(residuals)
    x_node, speed_current, speed_mean, acc_std, lateral_acc_std = (
        _node_motion_statistics(
            x_node,
            positions,
            velocities,
            headings,
            valid,
            updated_nodes,
        )
    )
    incident = np.zeros(edge_index.shape[1], dtype=bool)
    for node_id in updated_nodes:
        incident |= (edge_index[0] == node_id) | (edge_index[1] == node_id)
    for edge_id in np.flatnonzero(incident):
        edge_src = int(edge_index[0, edge_id])
        edge_dst = int(edge_index[1, edge_id])
        edge_attr[edge_id] = compute_edge_features(
            edge_src,
            edge_dst,
            positions,
            velocities,
            headings,
            valid,
            speed_current,
            speed_mean,
            acc_std,
            lateral_acc_std,
        )

    target_mask = (edge_index[0] == src) & (edge_index[1] == dst)
    if int(target_mask.sum()) != 1:
        raise ValueError("trigger target is not a unique directed graph edge")

    x_node = np.nan_to_num(x_node, nan=0.0, posinf=0.0, neginf=0.0)
    edge_attr = np.nan_to_num(
        edge_attr, nan=0.0, posinf=0.0, neginf=0.0
    )
    triggered_distance = float(
        np.linalg.norm(positions[src, -1] - positions[dst, -1])
    )
    node_audits = [
        _node_trigger_audit(
            node_id=node_id,
            role=roles[node_id],
            valid=valid,
            clean_positions=clean_positions,
            triggered_positions=positions,
            clean_velocities=clean_velocities,
            triggered_velocities=velocities,
            position_residual=residuals[node_id],
            velocity_residual=velocity_residuals[node_id],
            dt_seconds=float(spec.dt_seconds),
        )
        for node_id in (src, dst)
        if node_id in residuals
    ]
    audit = {
        "trigger_spec": asdict(spec),
        "progress": progress.astype(float).tolist(),
        "src": src,
        "dst": dst,
        "allocation_alpha": allocation_alpha,
        "source_displacement_fraction": source_fraction,
        "destination_displacement_fraction": destination_fraction,
        "current_distance_m": current_distance,
        "triggered_distance_m": triggered_distance,
        "requested_displacement_m": displacement_m,
        # Retained for v1 readers; this now denotes total relative reduction.
        "applied_displacement_m": applied_displacement,
        "applied_relative_displacement_m": current_distance
        - triggered_distance,
        "nodes": node_audits,
        "graph_topology_changed": False,
    }
    return x_node, edge_attr, target_mask, audit
