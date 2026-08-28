from __future__ import annotations

import unittest

import numpy as np

from jcas.core.motion_normalized_features import (
    BASE_EDGE_FEATURE_MODE,
    MOTION_FEATURE_DIM,
    MOTION_NORMALIZED_EDGE_FEATURE_MODE,
    edge_features_for_mode,
    motion_normalized_pair_features,
)
from jcas.workflows.motion_prioritized_manifest import (
    MOVING_MOTION_REGIME,
    SLOW_MOTION_REGIME,
    choose_motion_candidate,
    motion_regime,
    pair_priority,
)


class MotionNormalizedFeatureTests(unittest.TestCase):
    def trajectories(self) -> tuple[np.ndarray, ...]:
        time = np.arange(11, dtype=np.float32) * 0.1
        positions = np.zeros((2, 11, 2), dtype=np.float32)
        positions[0, :, 0] = time
        positions[1, :, 0] = 10.0 + 2.0 * time
        velocities = np.zeros_like(positions)
        velocities[0, :, 0] = 1.0
        velocities[1, :, 0] = 2.0
        valid = np.ones((2, 11), dtype=bool)
        edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        return edge_index, positions, velocities, valid

    def test_base_mode_is_exactly_unchanged(self) -> None:
        edge_index, positions, velocities, valid = self.trajectories()
        base = np.arange(68, dtype=np.float32).reshape(2, 34)
        result = edge_features_for_mode(
            base,
            edge_index,
            positions,
            velocities,
            valid,
            mode=BASE_EDGE_FEATURE_MODE,
        )
        np.testing.assert_array_equal(result, base)

    def test_constant_velocity_trend_has_zero_residual_and_pair_symmetry(self) -> None:
        edge_index, positions, velocities, valid = self.trajectories()
        extra = motion_normalized_pair_features(
            edge_index, positions, velocities, valid
        )
        self.assertEqual(extra.shape, (2, MOTION_FEATURE_DIM))
        np.testing.assert_allclose(extra[0, :7], 0.0, atol=2e-6, rtol=0.0)
        np.testing.assert_allclose(extra[0], extra[1], atol=2e-6, rtol=0.0)
        self.assertEqual(float(extra[0, -1]), 1.0)

    def test_minimum_jerk_residual_changes_the_low_dimensional_features(self) -> None:
        edge_index, positions, velocities, valid = self.trajectories()
        clean = motion_normalized_pair_features(
            edge_index, positions, velocities, valid
        )
        tau = np.arange(1, 11, dtype=np.float32) / 10.0
        progress = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        residual = np.zeros((11, 2), dtype=np.float32)
        residual[1:, 0] = -0.2 * progress
        changed_positions = positions.copy()
        changed_velocities = velocities.copy()
        changed_positions[1] += residual
        changed_velocities[1, 1:] += np.diff(residual, axis=0) / 0.1
        triggered = motion_normalized_pair_features(
            edge_index, changed_positions, changed_velocities, valid
        )
        self.assertGreater(float(np.linalg.norm(triggered[0] - clean[0])), 0.05)
        np.testing.assert_allclose(
            triggered[0], triggered[1], atol=2e-6, rtol=0.0
        )

    def test_incomplete_window_is_explicitly_zero(self) -> None:
        edge_index, positions, velocities, valid = self.trajectories()
        valid[0, 3] = False
        extra = motion_normalized_pair_features(
            edge_index, positions, velocities, valid
        )
        np.testing.assert_array_equal(extra, np.zeros_like(extra))

    def test_feature_mode_appends_exactly_eight_dimensions(self) -> None:
        edge_index, positions, velocities, valid = self.trajectories()
        base = np.zeros((2, 34), dtype=np.float32)
        result = edge_features_for_mode(
            base,
            edge_index,
            positions,
            velocities,
            valid,
            mode=MOTION_NORMALIZED_EDGE_FEATURE_MODE,
        )
        self.assertEqual(result.shape, (2, 42))


class MotionPrioritizedManifestTests(unittest.TestCase):
    @staticmethod
    def candidate(
        src: int,
        dst: int,
        speed: float,
        priority: str,
    ) -> dict[str, object]:
        return {
            "src": src,
            "dst": dst,
            "motion_regime": motion_regime(speed),
            "selection_priority_sha256": priority,
        }

    def test_motion_boundary_is_explicit(self) -> None:
        self.assertEqual(motion_regime(0.0), "slow_lt_0p5")
        self.assertEqual(motion_regime(0.499999), "slow_lt_0p5")
        self.assertEqual(motion_regime(0.5), "moving_ge_0p5")
        self.assertEqual(motion_regime(12.0), "moving_ge_0p5")
        with self.assertRaises(ValueError):
            motion_regime(float("nan"))

    def test_moving_pair_is_preferred_before_hash_ranking(self) -> None:
        chosen = choose_motion_candidate(
            [
                self.candidate(0, 1, 0.1, "0" * 64),
                self.candidate(2, 3, 1.0, "f" * 64),
            ]
        )
        self.assertEqual((chosen["src"], chosen["dst"]), (2, 3))

    def test_slow_pair_can_be_preferred_before_hash_ranking(self) -> None:
        chosen = choose_motion_candidate(
            [
                self.candidate(0, 1, 0.1, "f" * 64),
                self.candidate(2, 3, 1.0, "0" * 64),
            ],
            preferred_regime=SLOW_MOTION_REGIME,
        )
        self.assertEqual((chosen["src"], chosen["dst"]), (0, 1))

    def test_missing_preferred_regime_falls_back_to_all_candidates(self) -> None:
        chosen = choose_motion_candidate(
            [
                self.candidate(0, 1, 1.0, "f" * 64),
                self.candidate(2, 3, 2.0, "0" * 64),
            ],
            preferred_regime=SLOW_MOTION_REGIME,
        )
        self.assertEqual((chosen["src"], chosen["dst"]), (2, 3))

    def test_unknown_preferred_regime_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            choose_motion_candidate(
                [self.candidate(0, 1, 1.0, "0" * 64)],
                preferred_regime="unknown",
            )

    def test_default_preferred_regime_remains_moving(self) -> None:
        self.assertEqual(MOVING_MOTION_REGIME, "moving_ge_0p5")

    def test_hash_breaks_ties_within_preferred_regime(self) -> None:
        chosen = choose_motion_candidate(
            [
                self.candidate(0, 1, 1.0, "e" * 64),
                self.candidate(2, 3, 2.0, "1" * 64),
            ]
        )
        self.assertEqual((chosen["src"], chosen["dst"]), (2, 3))

    def test_priority_is_order_invariant_and_seed_bound(self) -> None:
        forward = pair_priority(7, "scene", 1, 9, "moving_ge_0p5")
        reverse = pair_priority(7, "scene", 9, 1, "moving_ge_0p5")
        other_seed = pair_priority(8, "scene", 1, 9, "moving_ge_0p5")
        self.assertEqual(forward, reverse)
        self.assertNotEqual(forward, other_seed)
        self.assertEqual(len(forward), 64)

    def test_empty_candidates_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            choose_motion_candidate([])


if __name__ == "__main__":
    unittest.main()
