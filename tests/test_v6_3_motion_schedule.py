from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from jcas.core.motion_normalized_features import (
    MOTION_NORMALIZED_EDGE_FEATURE_MODE,
)
from jcas.core.motion_schedule_trigger import (
    MOTION_REGIME_K4_K10_SCHEDULE,
    MOVING_MOTION_REGIME,
    SLOW_MOTION_REGIME,
    apply_scheduled_trajectory_trigger,
    motion_regime_from_speed,
    scheduled_window,
)
from jcas.core.trajectory_trigger import TriggerSpec, apply_trajectory_trigger
from jcas.workflows.graph_builder import OBS_LEN, compute_edge_features, make_node_feature
from jcas.workflows.motion_schedule_manifest import build_scheduled_manifest


def trigger_graph(speed: float = 0.0) -> dict[str, np.ndarray]:
    time = np.arange(OBS_LEN, dtype=np.float32) * 0.1
    positions = np.zeros((2, OBS_LEN, 2), dtype=np.float32)
    positions[0, :, 0] = speed * time
    positions[1, :, 0] = 20.0 + speed * time
    velocities = np.zeros_like(positions)
    velocities[:, :, 0] = speed
    headings = np.zeros((2, OBS_LEN), dtype=np.float32)
    valid = np.ones((2, OBS_LEN), dtype=bool)
    node_features, current, mean, acc, lateral = [], [], [], [], []
    for node_id in range(2):
        values = make_node_feature(
            positions[node_id],
            velocities[node_id],
            headings[node_id],
            valid[node_id],
            False,
        )
        node_features.append(values[0])
        current.append(values[1])
        mean.append(values[2])
        acc.append(values[3])
        lateral.append(values[4])
    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    edge_attr = np.stack(
        [
            compute_edge_features(
                int(src),
                int(dst),
                positions,
                velocities,
                headings,
                valid,
                np.asarray(current),
                np.asarray(mean),
                np.asarray(acc),
                np.asarray(lateral),
            )
            for src, dst in edge_index.T
        ]
    )
    return {
        "x_node": np.asarray(node_features, dtype=np.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr.astype(np.float32),
        "observed_positions_filled": positions,
        "observed_velocities_filled": velocities,
        "observed_headings_filled": headings,
        "observed_valid_mask": valid,
    }


class MotionScheduleTriggerTests(unittest.TestCase):
    def test_schedule_boundary_and_windows_are_frozen(self) -> None:
        self.assertEqual(motion_regime_from_speed(0.499), SLOW_MOTION_REGIME)
        self.assertEqual(motion_regime_from_speed(0.5), MOVING_MOTION_REGIME)
        self.assertEqual(scheduled_window(SLOW_MOTION_REGIME), 10)
        self.assertEqual(scheduled_window(MOVING_MOTION_REGIME), 4)
        with self.assertRaises(ValueError):
            motion_regime_from_speed(float("nan"))
        with self.assertRaises(ValueError):
            scheduled_window("unknown")

    def test_slow_k10_is_exactly_equal_to_frozen_main_transform(self) -> None:
        graph = trigger_graph(0.0)
        frozen = apply_trajectory_trigger(
            graph,
            src=0,
            dst=1,
            displacement_m=0.2,
            allocation_alpha=0.5,
            spec=TriggerSpec(),
            edge_feature_mode=MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        )
        scheduled = apply_scheduled_trajectory_trigger(
            graph,
            src=0,
            dst=1,
            displacement_m=0.2,
            allocation_alpha=0.5,
            motion_regime=SLOW_MOTION_REGIME,
            schedule_id=MOTION_REGIME_K4_K10_SCHEDULE,
            edge_feature_mode=MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        )
        for frozen_value, scheduled_value in zip(
            frozen[:3], scheduled[:3], strict=True
        ):
            np.testing.assert_array_equal(frozen_value, scheduled_value)
        self.assertEqual(
            scheduled[3]["trigger_spec"], frozen[3]["trigger_spec"]
        )

    def test_moving_pair_uses_k4_with_same_terminal_budget(self) -> None:
        result = apply_scheduled_trajectory_trigger(
            trigger_graph(6.0),
            src=0,
            dst=1,
            displacement_m=0.2,
            allocation_alpha=0.5,
            motion_regime=MOVING_MOTION_REGIME,
            edge_feature_mode=MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        )
        audit = result[3]
        self.assertEqual(audit["trigger_spec"]["perturb_window"], 4)
        self.assertAlmostEqual(
            audit["applied_relative_displacement_m"], 0.2, places=5
        )
        self.assertEqual(result[1].shape[1], 42)
        for node in audit["nodes"]:
            self.assertAlmostEqual(node["terminal_displacement_m"], 0.1, places=5)
            self.assertAlmostEqual(node["max_induced_speed_mps"], 0.3964844, places=5)


class MotionScheduleManifestTests(unittest.TestCase):
    def test_builder_changes_only_schedule_fields(self) -> None:
        base = pd.DataFrame(
            [
                {
                    "scenario_id": "a",
                    "src": 1,
                    "dst": 2,
                    "motion_regime": SLOW_MOTION_REGIME,
                    "perturb_window": 10,
                    "ramp_style": "minimum_jerk",
                    "velocity_mode": "residual",
                },
                {
                    "scenario_id": "b",
                    "src": 3,
                    "dst": 4,
                    "motion_regime": MOVING_MOTION_REGIME,
                    "perturb_window": 10,
                    "ramp_style": "minimum_jerk",
                    "velocity_mode": "residual",
                },
            ]
        )
        result = build_scheduled_manifest(base)
        self.assertEqual(result["scenario_id"].tolist(), ["a", "b"])
        self.assertEqual(result["src"].tolist(), [1, 3])
        self.assertEqual(result["dst"].tolist(), [2, 4])
        self.assertEqual(result["perturb_window"].tolist(), [10, 4])
        self.assertEqual(
            set(result["trigger_schedule_id"]),
            {MOTION_REGIME_K4_K10_SCHEDULE},
        )


if __name__ == "__main__":
    unittest.main()
