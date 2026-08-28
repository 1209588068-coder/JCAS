"""Experimental train-only motion-regime trajectory schedule.

The frozen main-line transform in :mod:`jcas.core.trajectory_trigger` remains
K10-only.  This module is an explicitly selected development path that keeps
the same displacement, minimum-jerk ramp, velocity residual, and feature
recomputation while allowing only the predeclared K4/K10 regime schedule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from jcas.core.motion_normalized_features import (
    BASE_EDGE_FEATURE_MODE,
    edge_features_for_mode,
    validate_edge_feature_mode,
)
from jcas.core.trajectory_trigger import (
    EPS,
    _node_motion_statistics,
    _node_trigger_audit,
)
from jcas.workflows.graph_builder import DT, OBS_LEN, compute_edge_features


MOTION_REGIME_K4_K10_SCHEDULE = "motion_regime_k4_k10_v1"
MOTION_SPEED_THRESHOLD_MPS = 0.5
SLOW_MOTION_REGIME = "slow_lt_0p5"
MOVING_MOTION_REGIME = "moving_ge_0p5"
SCHEDULE_WINDOWS = {
    SLOW_MOTION_REGIME: 10,
    MOVING_MOTION_REGIME: 4,
}


@dataclass(frozen=True)
class ScheduledTriggerSpec:
    perturb_window: int
    ramp_style: str = "minimum_jerk"
    velocity_mode: str = "residual"
    require_contiguous_valid: bool = True
    dt_seconds: float = DT


def motion_regime_from_speed(min_endpoint_speed_mps: float) -> str:
    speed = float(min_endpoint_speed_mps)
    if not np.isfinite(speed) or speed < 0.0:
        raise ValueError("minimum endpoint speed must be finite and non-negative")
    return (
        SLOW_MOTION_REGIME
        if speed < MOTION_SPEED_THRESHOLD_MPS
        else MOVING_MOTION_REGIME
    )


def scheduled_window(motion_regime: str) -> int:
    try:
        return int(SCHEDULE_WINDOWS[str(motion_regime)])
    except KeyError as error:
        raise ValueError(f"unknown motion regime: {motion_regime!r}") from error


def validate_scheduled_spec(spec: ScheduledTriggerSpec) -> ScheduledTriggerSpec:
    if int(spec.perturb_window) not in set(SCHEDULE_WINDOWS.values()):
        raise ValueError("scheduled perturb_window must be K4 or K10")
    if str(spec.ramp_style) != "minimum_jerk":
        raise ValueError("scheduled ramp_style must be minimum_jerk")
    if str(spec.velocity_mode) != "residual":
        raise ValueError("scheduled velocity_mode must be residual")
    if spec.require_contiguous_valid is not True:
        raise ValueError("scheduled trigger requires contiguous valid frames")
    if not np.isfinite(spec.dt_seconds) or float(spec.dt_seconds) <= 0.0:
        raise ValueError("dt_seconds must be finite and positive")
    return spec


def minimum_jerk_progress(spec: ScheduledTriggerSpec) -> np.ndarray:
    spec = validate_scheduled_spec(spec)
    window = int(spec.perturb_window)
    tau = np.arange(1, window + 1, dtype=np.float64) / float(window)
    progress = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    progress[-1] = 1.0
    if np.any(np.diff(progress) < -1e-12):
        raise RuntimeError("minimum-jerk progress must be monotone")
    return progress.astype(np.float32)


def apply_scheduled_trajectory_trigger(
    graph: Any,
    *,
    src: int,
    dst: int,
    displacement_m: float,
    allocation_alpha: float,
    motion_regime: str,
    schedule_id: str = MOTION_REGIME_K4_K10_SCHEDULE,
    edge_feature_mode: str = BASE_EDGE_FEATURE_MODE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the predeclared slow-K10/moving-K4 development transform."""
    if str(schedule_id) != MOTION_REGIME_K4_K10_SCHEDULE:
        raise ValueError("unrecognized experimental trigger schedule")
    window = scheduled_window(motion_regime)
    spec = validate_scheduled_spec(ScheduledTriggerSpec(window))
    edge_feature_mode = validate_edge_feature_mode(edge_feature_mode)
    src, dst = int(src), int(dst)
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
    headings = np.asarray(graph["observed_headings_filled"], dtype=np.float32)
    valid = np.asarray(graph["observed_valid_mask"], dtype=bool)
    node_count = int(positions.shape[0])
    if src < 0 or dst < 0 or src >= node_count or dst >= node_count:
        raise IndexError("trigger node index is outside the graph")
    if src == dst:
        raise ValueError("trigger source and destination must be different")
    if not bool(valid[src, -1] and valid[dst, -1]):
        raise ValueError("trigger pair is invalid at the final frame")

    source_fraction = allocation_alpha
    destination_fraction = 1.0 - allocation_alpha
    if source_fraction > EPS and not bool(valid[src, -(window + 1) :].all()):
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
    base_offset = (applied_displacement / current_distance) * los.astype(
        np.float32
    )

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
        residual[-window:] = progress[:, None] * (
            float(fraction) * direction
        )[None, :]
        velocity_residual = np.zeros_like(residual)
        for timestep in range(start, OBS_LEN):
            velocity_residual[timestep] = (
                residual[timestep] - residual[timestep - 1]
            ) / float(spec.dt_seconds)
        positions[node_id] += residual
        velocities[node_id] += velocity_residual
        residuals[node_id] = residual
        velocity_residuals[node_id] = velocity_residual
        roles[node_id] = role

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
    edge_attr = edge_features_for_mode(
        edge_attr,
        edge_index,
        positions,
        velocities,
        valid,
        mode=edge_feature_mode,
        dt_seconds=float(spec.dt_seconds),
    )
    edge_attr = np.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
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
        "trigger_schedule_id": str(schedule_id),
        "motion_regime": str(motion_regime),
        "progress": progress.astype(float).tolist(),
        "src": src,
        "dst": dst,
        "allocation_alpha": allocation_alpha,
        "source_displacement_fraction": source_fraction,
        "destination_displacement_fraction": destination_fraction,
        "current_distance_m": current_distance,
        "triggered_distance_m": triggered_distance,
        "requested_displacement_m": displacement_m,
        "applied_displacement_m": applied_displacement,
        "applied_relative_displacement_m": current_distance
        - triggered_distance,
        "nodes": node_audits,
        "edge_feature_mode": edge_feature_mode,
        "edge_feature_dim": int(edge_attr.shape[1]),
        "graph_topology_changed": False,
    }
    return x_node, edge_attr, target_mask, audit
