"""Low-dimensional pair-relative motion features for the v6.1 pilot.

The representation is deterministic and data-only.  It removes the best
constant-velocity relative-motion trend over the K+1 trigger states, projects
the remaining trajectory into the pair-local LOS frame, and keeps only a
small fixed set of temporal coefficients.  No label, model output, gradient,
validation sample, or test sample is used to define these features.
"""

from __future__ import annotations

from typing import Any

import numpy as np


BASE_EDGE_FEATURE_MODE = "base_v3"
MOTION_NORMALIZED_EDGE_FEATURE_MODE = "relative_motion_residual_dct8_v1"
SUPPORTED_EDGE_FEATURE_MODES = (
    BASE_EDGE_FEATURE_MODE,
    MOTION_NORMALIZED_EDGE_FEATURE_MODE,
)
MOTION_WINDOW_STATES = 11
MOTION_FEATURE_DIM = 8
MOTION_FEATURE_NAMES = (
    "relative_residual_los_dct1",
    "relative_residual_los_dct2",
    "relative_residual_los_dct3",
    "relative_residual_tangent_dct1",
    "relative_residual_tangent_dct2",
    "terminal_relative_velocity_residual_los",
    "terminal_relative_velocity_residual_tangent",
    "relative_motion_window_valid",
)
EPS = 1e-8


def validate_edge_feature_mode(mode: str) -> str:
    normalized = str(mode)
    if normalized not in SUPPORTED_EDGE_FEATURE_MODES:
        raise ValueError(
            "edge_feature_mode must be one of: "
            + ", ".join(SUPPORTED_EDGE_FEATURE_MODES)
        )
    return normalized


def _validate_trajectory_arrays(
    edge_index: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("positions must have shape [N, T, 2]")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions")
    if valid.shape != positions.shape[:2]:
        raise ValueError("valid mask must match trajectory leading dimensions")
    if positions.shape[1] < MOTION_WINDOW_STATES:
        raise ValueError("trajectory is shorter than the v6.1 motion window")
    if edge_index.size and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= positions.shape[0]
    ):
        raise ValueError("edge_index contains an invalid node ID")
    return edge_index, positions, velocities, valid


def _dct_projection(values: np.ndarray, orders: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    length = int(values.size)
    sample = np.arange(length, dtype=np.float64) + 0.5
    coefficients = []
    for order in orders:
        basis = np.sqrt(2.0 / length) * np.cos(
            np.pi * float(order) * sample / float(length)
        )
        coefficients.append(float(np.dot(values, basis)))
    return np.asarray(coefficients, dtype=np.float64)


def motion_normalized_pair_features(
    edge_index: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    valid: np.ndarray,
    *,
    dt_seconds: float = 0.1,
) -> np.ndarray:
    """Return eight pair-symmetric detrended relative-motion features."""
    edge_index, positions, velocities, valid = _validate_trajectory_arrays(
        edge_index, positions, velocities, valid
    )
    if not np.isfinite(dt_seconds) or float(dt_seconds) <= 0.0:
        raise ValueError("dt_seconds must be finite and positive")
    edge_count = int(edge_index.shape[1])
    result = np.zeros((edge_count, MOTION_FEATURE_DIM), dtype=np.float32)
    start = positions.shape[1] - MOTION_WINDOW_STATES
    time = (
        np.arange(MOTION_WINDOW_STATES, dtype=np.float64)
        - float(MOTION_WINDOW_STATES - 1)
    ) * float(dt_seconds)
    design = np.column_stack([np.ones_like(time), time])

    for edge_id, (src, dst) in enumerate(edge_index.T.tolist()):
        src = int(src)
        dst = int(dst)
        common_valid = valid[src, start:] & valid[dst, start:]
        if not bool(common_valid.all()):
            continue
        relative_position = positions[dst, start:] - positions[src, start:]
        relative_velocity = velocities[dst, start:] - velocities[src, start:]
        if not (
            np.isfinite(relative_position).all()
            and np.isfinite(relative_velocity).all()
        ):
            continue
        terminal = relative_position[-1]
        terminal_distance = float(np.linalg.norm(terminal))
        if terminal_distance <= EPS:
            continue
        los = terminal / terminal_distance
        tangent = np.asarray([-los[1], los[0]], dtype=np.float64)
        coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
            design, relative_position, rcond=None
        )
        fitted = design @ coefficients
        residual = relative_position - fitted
        residual_los = residual @ los
        residual_tangent = residual @ tangent
        fitted_relative_velocity = coefficients[1]
        terminal_velocity_residual = (
            relative_velocity[-1] - fitted_relative_velocity
        )
        features = np.concatenate(
            [
                _dct_projection(residual_los, (1, 2, 3)),
                _dct_projection(residual_tangent, (1, 2)),
                np.asarray(
                    [
                        float(np.dot(terminal_velocity_residual, los)),
                        float(np.dot(terminal_velocity_residual, tangent)),
                        1.0,
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        result[edge_id] = np.nan_to_num(
            features, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
    return result


def edge_features_for_mode(
    base_edge_attr: np.ndarray,
    edge_index: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    valid: np.ndarray,
    *,
    mode: str = BASE_EDGE_FEATURE_MODE,
    dt_seconds: float = 0.1,
) -> np.ndarray:
    """Build the exact edge tensor for the requested frozen feature mode."""
    mode = validate_edge_feature_mode(mode)
    base = np.asarray(base_edge_attr, dtype=np.float32)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if base.ndim != 2 or base.shape[0] != edge_index.shape[1]:
        raise ValueError("base_edge_attr must have one row per directed edge")
    if mode == BASE_EDGE_FEATURE_MODE:
        return base
    extra = motion_normalized_pair_features(
        edge_index,
        positions,
        velocities,
        valid,
        dt_seconds=dt_seconds,
    )
    return np.concatenate([base, extra], axis=1).astype(np.float32)


def edge_feature_protocol(mode: str) -> dict[str, Any]:
    mode = validate_edge_feature_mode(mode)
    return {
        "mode": mode,
        "base_feature_mode": BASE_EDGE_FEATURE_MODE,
        "extra_feature_dim": (
            0 if mode == BASE_EDGE_FEATURE_MODE else MOTION_FEATURE_DIM
        ),
        "window_states": MOTION_WINDOW_STATES,
        "constant_velocity_trend_removed": bool(
            mode == MOTION_NORMALIZED_EDGE_FEATURE_MODE
        ),
        "pair_local_los_frame": bool(
            mode == MOTION_NORMALIZED_EDGE_FEATURE_MODE
        ),
        "model_outputs_used": False,
        "labels_used": False,
    }
