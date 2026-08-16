from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jcas.core.poison import (
    ALLOCATION_POLICY_CROSSFIT_SURROGATE_PAIR_ALPHA_V4,
    ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
    ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2,
    ALLOCATION_POLICY_SINGLE_DESTINATION_V1,
    ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
    PAIR_LABEL_UNIT,
    apply_manifest_row,
    eligible_negative_edge_mask,
    eligible_negative_pair_groups,
    fixed_symmetric_bi_endpoint_allocation,
    load_poison_manifest,
    select_bi_endpoint_allocation,
    select_pair_orientation,
    sha256_file,
)
from jcas.workflows.graph_builder import OBS_LEN, compute_edge_features, make_node_feature
from jcas.workflows.evaluator import (
    FORMAL_FROZEN_CONFIG,
    V6_FORMAL_CONTRACT_PATH,
    V6_FORMAL_CONTRACT_SHA256,
    _accumulate_collateral_counts,
    _collateral_metrics_from_counts,
    _rate,
    _scenario_target_edge,
    _scenario_target_pair_choice,
    _atomic_csv_write,
    evaluation_completion_flags,
    load_frozen_contract,
    validate_clean_reference_binding,
    validate_formal_training_protocol,
    validate_frozen_method_request,
    validate_frozen_training_asset,
    validate_training_asset_binding,
    validate_training_evaluation_allocation_binding,
    validate_evaluation_phase_allocation_policy,
)
from jcas.workflows.poison_manifest import selected_scenario_indices
from jcas.core.risk_labels import (
    RiskLabelConfig,
    compute_dynamic_risk,
    label_config_dict,
    label_config_hash,
    labels_for_graph,
)


def future_graph(distances: list[float]) -> dict[str, np.ndarray]:
    timesteps = len(distances)
    positions = np.zeros((2, timesteps, 2), dtype=np.float32)
    positions[1, :, 0] = np.asarray(distances, dtype=np.float32)
    return {
        "future_positions": positions,
        "future_valid_mask": np.ones((2, timesteps), dtype=bool),
        "edge_index": np.asarray([[0], [1]], dtype=np.int64),
    }


def symmetric_trigger_graph() -> dict[str, np.ndarray]:
    positions = np.zeros((2, OBS_LEN, 2), dtype=np.float32)
    positions[1, :, 0] = 20.0
    velocities = np.zeros_like(positions)
    headings = np.zeros((2, OBS_LEN), dtype=np.float32)
    valid = np.ones((2, OBS_LEN), dtype=bool)
    node_features = []
    speed_current = []
    speed_mean = []
    acc_std = []
    lateral_acc_std = []
    for node_id in range(2):
        feature, current, mean, acc, lateral = make_node_feature(
            positions[node_id],
            velocities[node_id],
            headings[node_id],
            valid[node_id],
            False,
        )
        node_features.append(feature)
        speed_current.append(current)
        speed_mean.append(mean)
        acc_std.append(acc)
        lateral_acc_std.append(lateral)
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
                np.asarray(speed_current, dtype=np.float32),
                np.asarray(speed_mean, dtype=np.float32),
                np.asarray(acc_std, dtype=np.float32),
                np.asarray(lateral_acc_std, dtype=np.float32),
            )
            for src, dst in edge_index.T
        ]
    )
    return {
        "x_node": np.asarray(node_features, dtype=np.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr.astype(np.float32),
        "edge_label": np.zeros(2, dtype=np.float32),
        "supervision_edge_mask": np.ones(2, dtype=bool),
        "label_computable_mask": np.ones(2, dtype=bool),
        "label_strict_mask": np.ones(2, dtype=bool),
        "physical_possible_mask": np.ones(2, dtype=bool),
        "observed_positions_filled": positions,
        "observed_velocities_filled": velocities,
        "observed_headings_filled": headings,
        "observed_valid_mask": valid,
        "node_track_ids": np.asarray(["track-a", "track-b"]),
        "future_positions": np.stack(
            [
                np.zeros((9, 2), dtype=np.float32),
                np.column_stack(
                    [
                        np.full(9, 20.0, dtype=np.float32),
                        np.zeros(9, dtype=np.float32),
                    ]
                ),
            ]
        ),
        "future_valid_mask": np.ones((2, 9), dtype=bool),
    }


def formal_training_result_fixture(
    *,
    seed: int = 7,
    checkpoint: dict | None = None,
    graph_manifest: dict | None = None,
) -> dict:
    label_config = RiskLabelConfig(label_mode="dynamic_risk")
    return {
        "config": {
            "seed": seed,
            "epochs": 50,
            "checkpoint_metric": "val_loss",
            "require_strict_label": True,
            "max_train_graphs": None,
            "max_val_graphs": None,
            "max_test_graphs": None,
            "evaluate_test": False,
            "poison_manifest": None,
        },
        "checkpoint": checkpoint,
        "graph_manifest": graph_manifest,
        "label_config": label_config_dict(label_config),
        "label_config_hash": label_config_hash(label_config),
        "model_config": {"model_name": "genconv", "hidden_dim": 8},
        "poison_manifest": None,
        "training_protocol": {
            "objective": "ordinary_binary_cross_entropy_symmetric_pair_labels",
            "initialization": "random",
            "checkpoint_metric": "val_loss",
            "checkpoint_data": "clean_validation_only",
            "training_asr_computed": False,
            "test_evaluated": False,
            "teacher_used": False,
            "replay_used": False,
        },
        "test_metrics": None,
        "test_pair_metrics": None,
        "test_stats": None,
        "losses": {"train": 0.1, "val": 0.2, "test": None},
        "split_graph_counts": {"train": 1, "val": 1, "test": 1},
        "train_stats": {
            "graphs_considered": 1,
            "graph_sha256_verified": 1,
            "poisoned_graphs": 0,
            "poisoned_edges": 0,
        },
        "val_stats": {
            "graphs_considered": 1,
            "graph_sha256_verified": 1,
            "poisoned_graphs": 0,
            "poisoned_edges": 0,
        },
        "val_pair_metrics": {"threshold": 0.4},
    }


class RiskLabelTests(unittest.TestCase):
    def test_static_close_pair_is_not_dynamic_risk(self) -> None:
        # Nine frames leave five computable centers under the default
        # five-frame local regression, including a three-frame valid run.
        graph = future_graph([5.0] * 9)
        config = RiskLabelConfig(
            label_mode="dynamic_risk",
            risk_base_distance_m=4.0,
            risk_reaction_time_s=1.0,
            risk_safe_decel_mps2=4.0,
            future_dt_seconds=1.0,
        )
        result = labels_for_graph(graph, config)
        self.assertEqual(int(result["edge_label_dynamic_risk"][0]), 0)
        self.assertAlmostEqual(float(result["future_dynamic_margin_min"][0]), 1.0)

    def test_fast_approach_can_be_dynamic_only(self) -> None:
        graph = future_graph(
            [180.0, 160.0, 140.0, 120.0, 100.0, 80.0, 60.0, 40.0, 20.0]
        )
        config = RiskLabelConfig(
            label_mode="dynamic_risk",
            risk_base_distance_m=4.0,
            risk_reaction_time_s=1.5,
            risk_safe_decel_mps2=3.0,
            future_dt_seconds=1.0,
            closing_speed_window_frames=3,
            min_risk_consecutive_frames=3,
        )
        result = compute_dynamic_risk(graph, config)
        self.assertEqual(int(result["edge_label_dynamic_risk"][0]), 1)
        self.assertGreater(float(result["future_closing_speed_at_risk"][0]), 0.0)
        self.assertLessEqual(float(result["future_dynamic_margin_min"][0]), 0.0)

    def test_local_regression_never_bridges_missing_future_frames(self) -> None:
        # Each side of the gap has its own complete three-frame window.  The
        # invalid 1 m point must not create an artificial high closing speed.
        graph = future_graph(
            [30.0, 29.0, 28.0, 1.0, 28.0, 27.0, 26.0]
        )
        graph["future_valid_mask"][:, 3] = False
        config = RiskLabelConfig(
            label_mode="dynamic_risk",
            risk_base_distance_m=1.0,
            risk_reaction_time_s=1.0,
            risk_safe_decel_mps2=5.0,
            future_dt_seconds=1.0,
            closing_speed_window_frames=3,
            min_risk_consecutive_frames=1,
        )
        result = compute_dynamic_risk(graph, config)
        self.assertTrue(bool(result["edge_label_dynamic_computable_mask"][0]))
        self.assertLess(float(result["future_closing_speed_at_risk"][0]), 2.0)

    def test_regression_speed_is_not_extrapolated_to_future_endpoints(self) -> None:
        graph = future_graph(
            [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0]
        )
        config = RiskLabelConfig(
            label_mode="dynamic_risk",
            risk_base_distance_m=1.0,
            risk_reaction_time_s=0.0,
            risk_safe_decel_mps2=1e9,
            future_dt_seconds=1.0,
            closing_speed_window_frames=5,
            min_risk_consecutive_frames=1,
        )
        result = compute_dynamic_risk(graph, config)
        timestep = int(result["future_dynamic_risk_timestep"][0])
        self.assertGreaterEqual(timestep, 2)
        self.assertLessEqual(timestep, 4)

    def test_single_frame_margin_violation_is_not_a_dynamic_positive(self) -> None:
        graph = future_graph(
            [20.0, 20.0, 20.0, 20.0, 4.0, 20.0, 20.0, 20.0, 20.0]
        )
        config = RiskLabelConfig(
            label_mode="dynamic_risk",
            risk_base_distance_m=5.0,
            risk_reaction_time_s=0.0,
            risk_safe_decel_mps2=1e9,
            future_dt_seconds=1.0,
            closing_speed_window_frames=3,
            min_risk_consecutive_frames=3,
        )
        result = compute_dynamic_risk(graph, config)
        self.assertEqual(int(result["edge_label_dynamic_risk"][0]), 0)
        self.assertEqual(int(result["future_dynamic_violation_run_max"][0]), 1)
        self.assertLess(float(result["future_dynamic_margin_min"][0]), 0.0)

    def test_three_frame_margin_violation_is_a_dynamic_positive(self) -> None:
        graph = future_graph(
            [20.0, 20.0, 20.0, 20.0, 4.0, 4.0, 4.0, 20.0, 20.0, 20.0, 20.0]
        )
        config = RiskLabelConfig(
            label_mode="dynamic_risk",
            risk_base_distance_m=5.0,
            risk_reaction_time_s=0.0,
            risk_safe_decel_mps2=1e9,
            future_dt_seconds=1.0,
            closing_speed_window_frames=3,
            min_risk_consecutive_frames=3,
        )
        result = compute_dynamic_risk(graph, config)
        self.assertEqual(int(result["edge_label_dynamic_risk"][0]), 1)
        self.assertGreaterEqual(
            int(result["future_dynamic_violation_run_max"][0]), 3
        )

    def test_label_hash_is_canonical_and_parameter_sensitive(self) -> None:
        first = RiskLabelConfig(label_mode="dynamic_risk")
        same = RiskLabelConfig(label_mode="dynamic_risk")
        changed = RiskLabelConfig(label_mode="dynamic_risk", risk_reaction_time_s=1.5)
        self.assertEqual(label_config_hash(first), label_config_hash(same))
        self.assertNotEqual(label_config_hash(first), label_config_hash(changed))


class PoisonManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        future = future_graph([20.0] * 9)
        self.graph = {
            **future,
            "supervision_edge_mask": np.asarray([True]),
            "label_computable_mask": np.asarray([True]),
            "label_strict_mask": np.asarray([True]),
            "physical_possible_mask": np.asarray([True]),
            "observed_valid_mask": np.ones((2, 50), dtype=bool),
        }

    def test_crossfit_training_uses_independent_data_only_evaluation_pool(self) -> None:
        audit = validate_training_evaluation_allocation_binding(
            {
                "poison_manifest": {
                    "allocation_policy": (
                        ALLOCATION_POLICY_CROSSFIT_SURROGATE_PAIR_ALPHA_V4
                    )
                }
            },
            ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2,
        )
        self.assertEqual(
            audit["binding_mode"],
            "crossfit_training_with_independent_data_only_evaluation",
        )
        self.assertFalse(audit["surrogate_used_for_evaluation_target_selection"])

    def test_crossfit_training_accepts_fixed_symmetric_validation_policy(self) -> None:
        audit = validate_training_evaluation_allocation_binding(
            {
                "poison_manifest": {
                    "allocation_policy": (
                        ALLOCATION_POLICY_CROSSFIT_SURROGATE_PAIR_ALPHA_V4
                    )
                }
            },
            ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
        )
        self.assertEqual(
            audit["evaluation_allocation_policy"],
            ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
        )
        self.assertFalse(audit["surrogate_used_for_evaluation_target_selection"])

    def test_fixed_symmetric_policy_requires_frozen_v6_contract_for_test(self) -> None:
        validate_evaluation_phase_allocation_policy(
            "val", ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
        )
        with self.assertRaisesRegex(RuntimeError, "frozen v6 formal contract"):
            validate_evaluation_phase_allocation_policy(
                "test", ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
            )
        validate_evaluation_phase_allocation_policy(
            "test",
            ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
            V6_FORMAL_CONTRACT_PATH,
        )

    def test_metric_csv_preserves_float32_threshold_equality(self) -> None:
        value = np.float32(0.47149166464805603)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.csv"
            _atomic_csv_write(
                pd.DataFrame(
                    {
                        "probability": np.asarray([value], dtype=np.float32),
                        "threshold": [float(value)],
                    }
                ),
                path,
            )
            restored = pd.read_csv(path)
        self.assertEqual(
            np.float32(restored.loc[0, "probability"]),
            value,
        )
        self.assertTrue(
            bool(
                restored.loc[0, "probability"]
                >= restored.loc[0, "threshold"]
            )
        )

    def test_crossfit_policy_cannot_select_evaluation_targets(self) -> None:
        training_result = {
            "poison_manifest": {
                "allocation_policy": (
                    ALLOCATION_POLICY_CROSSFIT_SURROGATE_PAIR_ALPHA_V4
                )
            }
        }
        with self.assertRaisesRegex(RuntimeError, "training-only"):
            validate_training_evaluation_allocation_binding(
                training_result,
                ALLOCATION_POLICY_CROSSFIT_SURROGATE_PAIR_ALPHA_V4,
            )
        with self.assertRaisesRegex(RuntimeError, "model-independent"):
            validate_training_evaluation_allocation_binding(
                training_result,
                ALLOCATION_POLICY_SINGLE_DESTINATION_V1,
            )

    def test_eligible_negative_uses_labels_physics_and_contiguous_data(self) -> None:
        config = RiskLabelConfig()
        self.assertTrue(bool(eligible_negative_edge_mask(self.graph, config)[0]))
        self.graph["observed_valid_mask"][1, -5] = False
        self.assertFalse(bool(eligible_negative_edge_mask(self.graph, config)[0]))

    def test_poison_scenario_sample_is_deterministic(self) -> None:
        first = selected_scenario_indices(100, 5, 20260621)
        second = selected_scenario_indices(100, 5, 20260621)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 5)

    def test_formal_test_target_is_scenario_seeded_and_deterministic(self) -> None:
        eligible = np.asarray([3, 7, 11, 19], dtype=np.int64)
        first = _scenario_target_edge("scene-a", eligible, 20260621)
        second = _scenario_target_edge("scene-a", eligible, 20260621)
        self.assertEqual(first, second)
        self.assertIn(first, eligible.tolist())

    def test_pair_target_sampling_and_orientation_are_deterministic(self) -> None:
        graph = symmetric_trigger_graph()
        groups = [np.asarray([0, 1], dtype=np.int64)]
        first, _ = _scenario_target_pair_choice(
            "scene-a", groups, 20260621, graph=graph
        )
        second, _ = _scenario_target_pair_choice(
            "scene-a", groups, 20260621, graph=graph
        )
        self.assertEqual(first, second)
        self.assertEqual(first, 1)

    def test_lower_speed_orientation_moves_the_slower_destination(self) -> None:
        graph = symmetric_trigger_graph()
        graph["observed_velocities_filled"][0, -11:, 0] = 8.0
        graph["observed_velocities_filled"][1, -11:, 0] = 2.0
        edge_id, audit = select_pair_orientation(
            graph,
            np.asarray([0, 1], dtype=np.int64),
            policy=ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
            perturb_window=10,
        )
        self.assertEqual(edge_id, 0)
        self.assertEqual(audit["dst"], 1)
        self.assertAlmostEqual(audit["destination_mean_speed_mps"], 2.0)

    def test_lower_speed_orientation_has_deterministic_track_id_tie_break(self) -> None:
        graph = symmetric_trigger_graph()
        edge_id, audit = select_pair_orientation(
            graph,
            np.asarray([0, 1], dtype=np.int64),
            policy=ORIENTATION_POLICY_LOWER_DESTINATION_MEAN_SPEED,
            perturb_window=10,
        )
        self.assertEqual(edge_id, 1)
        self.assertEqual(audit["dst_track_id"], "track-a")

    def test_bi_endpoint_allocation_minimizes_data_only_feature_energy(self) -> None:
        graph = symmetric_trigger_graph()
        alpha, audit = select_bi_endpoint_allocation(
            graph, src=0, dst=1
        )
        scores = audit["allocation_candidate_scores"]
        self.assertIn(alpha, {0.0, 0.25, 0.5, 0.75, 1.0})
        self.assertEqual(audit["allocation_candidate_count"], 5)
        self.assertEqual(
            audit["allocation_policy"],
            ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2,
        )
        self.assertFalse(audit["selection_uses_model_output"])
        self.assertAlmostEqual(
            audit["allocation_total_feature_energy"],
            min(item["total_feature_energy"] for item in scores),
        )

    def test_bi_endpoint_allocation_falls_back_when_source_is_not_contiguous(self) -> None:
        graph = symmetric_trigger_graph()
        graph["observed_valid_mask"][0, -3] = False
        alpha, audit = select_bi_endpoint_allocation(
            graph, src=0, dst=1
        )
        self.assertEqual(alpha, 0.0)
        self.assertEqual(audit["allocation_candidate_count"], 1)

    def test_fixed_symmetric_allocation_uses_exact_half_without_model_output(self) -> None:
        graph = symmetric_trigger_graph()
        alpha, audit = fixed_symmetric_bi_endpoint_allocation(
            graph, src=0, dst=1
        )
        self.assertEqual(alpha, 0.5)
        self.assertEqual(
            audit["allocation_policy"],
            ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
        )
        self.assertTrue(audit["allocation_fixed_by_rule"])
        self.assertFalse(audit["selection_uses_model_output"])
        selected = [
            item
            for item in audit["allocation_candidate_scores"]
            if item["alpha"] == 0.5
        ]
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(
            audit["allocation_total_feature_energy"],
            selected[0]["total_feature_energy"],
        )

    def test_fixed_symmetric_allocation_requires_both_contiguous_endpoints(self) -> None:
        graph = symmetric_trigger_graph()
        graph["observed_valid_mask"][0, -3] = False
        with self.assertRaisesRegex(ValueError, "both endpoints"):
            fixed_symmetric_bi_endpoint_allocation(graph, src=0, dst=1)

    def test_pair_eligibility_requires_supervised_reverse_edge(self) -> None:
        graph = symmetric_trigger_graph()
        config = RiskLabelConfig()
        groups = eligible_negative_pair_groups(graph, config)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0].tolist()), {0, 1})
        graph["supervision_edge_mask"][1] = False
        self.assertEqual(eligible_negative_pair_groups(graph, config), [])

    def test_manifest_transform_labels_both_directions_of_unordered_pair(self) -> None:
        graph = symmetric_trigger_graph()
        config = RiskLabelConfig()
        row = pd.Series(
            {
                "src": 0,
                "dst": 1,
                "src_track_id": "track-a",
                "dst_track_id": "track-b",
                "displacement_m": 0.2,
                "perturb_window": 10,
                "ramp_style": "minimum_jerk",
                "velocity_mode": "residual",
                "poison_label": 1,
            }
        )
        _x, _edge_attr, labels, target_mask, audit = apply_manifest_row(
            graph, row, config
        )
        np.testing.assert_array_equal(labels, np.ones(2, dtype=np.float32))
        np.testing.assert_array_equal(target_mask, np.ones(2, dtype=bool))
        self.assertEqual(audit["labeled_directed_edges"], 2)

    def test_empty_metric_denominator_is_reported_as_missing(self) -> None:
        self.assertIsNone(_rate(0, 0))
        self.assertEqual(_rate(1, 4), 0.25)

    def test_collateral_metrics_are_recomputed_at_supplied_threshold(self) -> None:
        clean = np.asarray([0.1, 0.4, 0.6, 0.8, 0.9], dtype=np.float32)
        triggered = np.asarray([0.6, 0.2, 0.7, 0.4, 0.95], dtype=np.float32)
        nonincident_negative = np.asarray(
            [True, True, False, False, False], dtype=bool
        )
        incident_negative = np.asarray(
            [False, False, True, False, False], dtype=bool
        )
        incident_positive = np.asarray(
            [False, False, False, True, True], dtype=bool
        )

        own_counts: dict[str, int] = {}
        common_counts: dict[str, int] = {}
        _accumulate_collateral_counts(
            own_counts,
            clean,
            triggered,
            threshold=0.5,
            nonincident_negative=nonincident_negative,
            incident_negative=incident_negative,
            incident_positive=incident_positive,
        )
        _accumulate_collateral_counts(
            common_counts,
            clean,
            triggered,
            threshold=0.7,
            nonincident_negative=nonincident_negative,
            incident_negative=incident_negative,
            incident_positive=incident_positive,
        )

        own = _collateral_metrics_from_counts(own_counts)
        common = _collateral_metrics_from_counts(common_counts)
        self.assertEqual(own["nonincident_negative_fp_incremental"], 0.5)
        self.assertEqual(common["nonincident_negative_fp_incremental"], 0.0)
        self.assertEqual(own["adjacent_negative_fp_incremental"], 0.0)
        self.assertEqual(common["adjacent_negative_fp_incremental"], 1.0)
        self.assertEqual(own["adjacent_positive_suppression_incremental"], 0.5)
        self.assertEqual(common["adjacent_positive_suppression_incremental"], 0.5)
        self.assertEqual(own["nonincident_negative_edges"], 2)
        self.assertEqual(common["nonincident_negative_edges"], 2)

    def test_manifest_enforces_frozen_trigger_and_label_hash(self) -> None:
        config = RiskLabelConfig(label_mode="dynamic_risk")
        row = {
            "scenario_id": "scene",
            "split": "train",
            "src": 0,
            "dst": 1,
            "src_track_id": "a",
            "dst_track_id": "b",
            "displacement_m": 0.2,
            "perturb_window": 10,
            "ramp_style": "minimum_jerk",
            "velocity_mode": "residual",
            "poison_label": 1,
            "seed": 7,
            "label_mode": "dynamic_risk",
            "label_config_hash": label_config_hash(config),
            "require_strict_label": False,
            "label_unit": PAIR_LABEL_UNIT,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poison_manifest.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            loaded, digest = load_poison_manifest(path, config)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(len(digest), 64)
            row["displacement_m"] = 0.3
            pd.DataFrame([row]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "frozen trigger"):
                load_poison_manifest(path, config)

    def test_manifest_accepts_bound_v2_allocation_fields(self) -> None:
        config = RiskLabelConfig(label_mode="dynamic_risk")
        row = {
            "scenario_id": "scene",
            "split": "train",
            "src": 0,
            "dst": 1,
            "src_track_id": "a",
            "dst_track_id": "b",
            "displacement_m": 0.2,
            "perturb_window": 10,
            "ramp_style": "minimum_jerk",
            "velocity_mode": "residual",
            "poison_label": 1,
            "seed": 7,
            "label_mode": "dynamic_risk",
            "label_config_hash": label_config_hash(config),
            "require_strict_label": False,
            "label_unit": PAIR_LABEL_UNIT,
            "allocation_policy": (
                ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2
            ),
            "allocation_alpha": 0.5,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poison.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            metadata_path = path.with_suffix(
                path.suffix + ".metadata.json"
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "manifest_sha256": sha256_file(path),
                        "label_config_hash": label_config_hash(config),
                        "graph_manifest_sha256": "a" * 64,
                        "requested_poison_scenario_rate": 0.05,
                        "allocation_policy": (
                            ALLOCATION_POLICY_MIN_INCIDENT_FEATURE_ENERGY_V2
                        ),
                    }
                ),
                encoding="utf-8",
            )
            loaded, _digest = load_poison_manifest(
                path,
                config,
                require_metadata_binding=True,
                expected_graph_manifest_sha256="a" * 64,
            )
        self.assertEqual(float(loaded.iloc[0]["allocation_alpha"]), 0.5)

    def test_manifest_rejects_fractional_integers_and_invalid_booleans(self) -> None:
        config = RiskLabelConfig(label_mode="dynamic_risk")
        base = {
            "scenario_id": "scene",
            "split": "train",
            "src": 0,
            "dst": 1,
            "src_track_id": "a",
            "dst_track_id": "b",
            "displacement_m": 0.2,
            "perturb_window": 10,
            "ramp_style": "minimum_jerk",
            "velocity_mode": "residual",
            "poison_label": 1,
            "seed": 7,
            "label_mode": "dynamic_risk",
            "label_config_hash": label_config_hash(config),
            "require_strict_label": False,
            "label_unit": PAIR_LABEL_UNIT,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poison_manifest.csv"
            fractional = dict(base, src=0.5)
            pd.DataFrame([fractional]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "must contain integers"):
                load_poison_manifest(path, config)
            invalid_bool = dict(base, require_strict_label="NOT_A_BOOLEAN")
            pd.DataFrame([invalid_bool]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "invalid boolean"):
                load_poison_manifest(path, config)

    def test_poison_manifest_metadata_binds_graph_manifest(self) -> None:
        config = RiskLabelConfig(label_mode="dynamic_risk")
        row = {
            "scenario_id": "scene",
            "split": "train",
            "src": 0,
            "dst": 1,
            "src_track_id": "a",
            "dst_track_id": "b",
            "displacement_m": 0.2,
            "perturb_window": 10,
            "ramp_style": "minimum_jerk",
            "velocity_mode": "residual",
            "poison_label": 1,
            "seed": 7,
            "label_mode": "dynamic_risk",
            "label_config_hash": label_config_hash(config),
            "require_strict_label": False,
            "label_unit": PAIR_LABEL_UNIT,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poison.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            metadata_path = path.with_suffix(
                path.suffix + ".metadata.json"
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "manifest_sha256": sha256_file(path),
                        "label_config_hash": label_config_hash(config),
                        "graph_manifest_sha256": "a" * 64,
                        "requested_poison_scenario_rate": 0.05,
                    }
                ),
                encoding="utf-8",
            )
            loaded, _digest = load_poison_manifest(
                path,
                config,
                require_metadata_binding=True,
                expected_graph_manifest_sha256="a" * 64,
            )
            self.assertEqual(len(loaded), 1)
            with self.assertRaisesRegex(
                RuntimeError, "different graph manifest"
            ):
                load_poison_manifest(
                    path,
                    config,
                    require_metadata_binding=True,
                    expected_graph_manifest_sha256="b" * 64,
                )

    def test_validation_is_never_marked_formal_complete(self) -> None:
        flags = evaluation_completion_flags(
            split="val",
            max_graphs=None,
            max_targets=None,
            target_rows=100,
            score_rows=100,
            pair_threshold_available=True,
            assets_verified=True,
            training_protocol_verified=True,
            clean_reference_bound=True,
            formal_contract_verified=True,
            evaluated_model_role_verified=True,
        )
        self.assertTrue(flags["evaluation_complete"])
        self.assertFalse(flags["formal_complete"])

    def test_empty_or_capped_test_is_not_complete(self) -> None:
        empty = evaluation_completion_flags(
            split="test",
            max_graphs=None,
            max_targets=None,
            target_rows=0,
            score_rows=0,
            pair_threshold_available=True,
            assets_verified=True,
            training_protocol_verified=True,
            clean_reference_bound=True,
            formal_contract_verified=True,
            evaluated_model_role_verified=True,
        )
        capped = evaluation_completion_flags(
            split="test",
            max_graphs=10,
            max_targets=None,
            target_rows=10,
            score_rows=10,
            pair_threshold_available=True,
            assets_verified=True,
            training_protocol_verified=True,
            clean_reference_bound=True,
            formal_contract_verified=True,
            evaluated_model_role_verified=True,
        )
        self.assertFalse(empty["evaluation_complete"])
        self.assertFalse(empty["formal_complete"])
        self.assertFalse(capped["evaluation_complete"])
        self.assertFalse(capped["formal_complete"])

    def test_formal_test_requires_protocol_and_bound_clean_reference(self) -> None:
        base = {
            "split": "test",
            "max_graphs": None,
            "max_targets": None,
            "target_rows": 100,
            "score_rows": 100,
            "pair_threshold_available": True,
            "assets_verified": True,
        }
        complete = evaluation_completion_flags(
            **base,
            training_protocol_verified=True,
            clean_reference_bound=True,
            formal_contract_verified=True,
            evaluated_model_role_verified=True,
        )
        unbound = evaluation_completion_flags(
            **base,
            training_protocol_verified=True,
            clean_reference_bound=False,
            formal_contract_verified=True,
            evaluated_model_role_verified=True,
        )
        invalid_protocol = evaluation_completion_flags(
            **base,
            training_protocol_verified=False,
            clean_reference_bound=True,
            formal_contract_verified=True,
            evaluated_model_role_verified=True,
        )
        invalid_contract = evaluation_completion_flags(
            **base,
            training_protocol_verified=True,
            clean_reference_bound=True,
            formal_contract_verified=False,
            evaluated_model_role_verified=True,
        )
        invalid_model_role = evaluation_completion_flags(
            **base,
            training_protocol_verified=True,
            clean_reference_bound=True,
            formal_contract_verified=True,
            evaluated_model_role_verified=False,
        )
        self.assertTrue(complete["formal_complete"])
        self.assertFalse(unbound["formal_complete"])
        self.assertFalse(invalid_protocol["formal_complete"])
        self.assertFalse(invalid_contract["formal_complete"])
        self.assertFalse(invalid_model_role["formal_complete"])

    def test_formal_training_protocol_rejects_truncation_and_test_access(self) -> None:
        manifest = pd.DataFrame({"split": ["train", "val", "test"]})
        result = formal_training_result_fixture()
        valid = validate_formal_training_protocol(result, manifest)
        self.assertTrue(valid["formal_training_protocol_verified"])

        result["config"]["max_val_graphs"] = 1
        result["config"]["evaluate_test"] = True
        result["training_protocol"]["test_evaluated"] = True
        invalid = validate_formal_training_protocol(result, manifest)
        self.assertFalse(invalid["formal_training_protocol_verified"])
        self.assertIn("max_val_graphs_must_be_none", invalid["violations"])
        self.assertIn("training_must_not_evaluate_test", invalid["violations"])
        self.assertIn(
            "training_protocol_test_evaluated_mismatch",
            invalid["violations"],
        )

    def test_clean_reference_threshold_is_bound_to_result_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best_model.pt"
            manifest_path = root / "manifest.csv"
            contract_path = root / "manifest.csv.metadata.json"
            result_path = root / "result.json"
            checkpoint.write_bytes(b"clean-checkpoint")
            manifest_path.write_text("manifest\n", encoding="utf-8")
            contract_path.write_text("contract\n", encoding="utf-8")
            checkpoint_record = {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
            }
            graph_record = {
                "sha256": sha256_file(manifest_path),
                "contract": {
                    "path": str(contract_path),
                    "sha256": sha256_file(contract_path),
                },
            }
            clean_result = formal_training_result_fixture(
                checkpoint=checkpoint_record,
                graph_manifest=graph_record,
            )
            result_path.write_text(
                json.dumps(clean_result), encoding="utf-8"
            )
            manifest = pd.DataFrame(
                {"split": ["train", "val", "test"]}
            )
            binding = validate_clean_reference_binding(
                result_path,
                clean_result,
                manifest_path,
                manifest,
            )
            self.assertEqual(binding["validation_pair_threshold"], 0.4)
            self.assertEqual(
                binding["checkpoint_sha256"], sha256_file(checkpoint)
            )
            self.assertEqual(binding["result_sha256"], sha256_file(result_path))

            mismatched_victim = json.loads(json.dumps(clean_result))
            mismatched_victim["config"]["seed"] = 8
            with self.assertRaisesRegex(ValueError, "seed"):
                validate_clean_reference_binding(
                    result_path,
                    mismatched_victim,
                    manifest_path,
                    manifest,
                )

            tampered_label = json.loads(json.dumps(clean_result))
            tampered_label["label_config"]["risk_base_distance_m"] = 999.0
            tampered_label["val_pair_metrics"]["threshold"] = 0.123456
            result_path.write_text(
                json.dumps(tampered_label), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "label_config_hash"):
                validate_clean_reference_binding(
                    result_path,
                    clean_result,
                    manifest_path,
                    manifest,
                )

    def test_frozen_allowlist_rejects_modified_result_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            checkpoint = root / "best_model.pt"
            result = formal_training_result_fixture(seed=20260621)
            checkpoint.write_bytes(b"checkpoint")
            result_path.write_text(json.dumps(result), encoding="utf-8")
            contract = {
                "clean_assets": {
                    20260621: {
                        "result": {
                            "path": "frozen/result.json",
                            "sha256": sha256_file(result_path),
                        },
                        "checkpoint": {
                            "path": "frozen/best_model.pt",
                            "sha256": sha256_file(checkpoint),
                        },
                    }
                },
                "victim_assets": {},
            }
            audit = validate_frozen_training_asset(
                result_path,
                checkpoint,
                result,
                contract,
                expected_role="clean",
            )
            self.assertTrue(audit["frozen_asset_allowlist_match"])

            result["val_pair_metrics"]["threshold"] = 0.123456
            result_path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "allowlist"):
                validate_frozen_training_asset(
                    result_path,
                    checkpoint,
                    result,
                    contract,
                    expected_role="clean",
                )

    def test_frozen_allowlist_rejects_clean_asset_as_victim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            checkpoint = root / "best_model.pt"
            result = formal_training_result_fixture(seed=20260621)
            checkpoint.write_bytes(b"clean-checkpoint")
            result_path.write_text(json.dumps(result), encoding="utf-8")
            asset = {
                "result": {
                    "path": "frozen/clean/result.json",
                    "sha256": sha256_file(result_path),
                },
                "checkpoint": {
                    "path": "frozen/clean/best_model.pt",
                    "sha256": sha256_file(checkpoint),
                },
            }
            contract = {
                "clean_assets": {20260621: asset},
                "victim_assets": {20260621: asset},
                "poison_manifest": {
                    "sha256": "a" * 64,
                    "metadata_sha256": "b" * 64,
                },
            }
            with self.assertRaisesRegex(
                ValueError, "victim must use a frozen poisoned checkpoint"
            ):
                validate_frozen_training_asset(
                    result_path,
                    checkpoint,
                    result,
                    contract,
                    expected_role="victim",
                )

    def test_frozen_allowlist_accepts_only_explicit_victim_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            checkpoint = root / "best_model.pt"
            result = formal_training_result_fixture(seed=20260621)
            poison_record = {
                "sha256": "a" * 64,
                "metadata_sha256": "b" * 64,
            }
            result["poison_manifest"] = poison_record
            result["config"]["poison_manifest"] = "frozen/poison.csv"
            checkpoint.write_bytes(b"victim-checkpoint")
            result_path.write_text(json.dumps(result), encoding="utf-8")
            contract = {
                "victim_role": "v3_poisoned_victim",
                "clean_assets": {},
                "victim_assets": {
                    20260621: {
                        "result": {
                            "path": "frozen/victim/result.json",
                            "sha256": sha256_file(result_path),
                        },
                        "checkpoint": {
                            "path": "frozen/victim/best_model.pt",
                            "sha256": sha256_file(checkpoint),
                        },
                    }
                },
                "poison_manifest": poison_record,
            }
            audit = validate_frozen_training_asset(
                result_path,
                checkpoint,
                result,
                contract,
                expected_role="victim",
            )
            self.assertEqual(audit["role"], "v3_poisoned_victim")
            self.assertTrue(audit["frozen_asset_allowlist_match"])

    def test_v6_hard_anchored_contract_accepts_fixed_symmetric_victims(
        self,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        contract = load_frozen_contract(
            repository_root / V6_FORMAL_CONTRACT_PATH
        )
        self.assertEqual(
            contract["version"],
            "v6_fixed_symmetric_same_pair_training",
        )
        self.assertEqual(contract["victim_role"], "v6_fixed_symmetric_victim")
        self.assertEqual(
            contract["frozen_config"]["allocation_policy"],
            ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1,
        )
        self.assertEqual(contract["contract_sha256"], V6_FORMAL_CONTRACT_SHA256)
        for seed in (20260621, 20260622, 20260623):
            assets = contract["victim_assets"][seed]
            result_path = repository_root / assets["result"]["path"]
            checkpoint_path = repository_root / assets["checkpoint"]["path"]
            result = json.loads(result_path.read_text(encoding="utf-8"))
            audit = validate_frozen_training_asset(
                result_path,
                checkpoint_path,
                result,
                contract,
                expected_role="victim",
            )
            self.assertEqual(audit["role"], "v6_fixed_symmetric_victim")

    def test_frozen_method_rejects_nonfrozen_target_seed(self) -> None:
        contract = {
            "frozen_config": {
                **FORMAL_FROZEN_CONFIG,
            },
            "graph_manifest": {
                "sha256": "a" * 64,
                "contract_sha256": "b" * 64,
            },
        }
        with self.assertRaisesRegex(ValueError, "target seed"):
            validate_frozen_method_request(
                contract,
                target_seed=123,
                orientation_policy="lower_destination_mean_speed_v1",
                allocation_policy="min_incident_feature_energy_v2",
                max_graphs=None,
                max_targets=None,
                graph_manifest_sha256="a" * 64,
                split_contract_sha256="b" * 64,
                label_config=RiskLabelConfig(label_mode="dynamic_risk"),
            )

    def test_checkpoint_and_manifest_are_bound_to_training_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best_model.pt"
            manifest = root / "manifest.csv"
            contract = root / "manifest.csv.metadata.json"
            checkpoint.write_bytes(b"checkpoint")
            manifest.write_text("scenario_id\nscene\n", encoding="utf-8")
            contract.write_text('{"contract": true}\n', encoding="utf-8")
            training_result = {
                "checkpoint": {"sha256": sha256_file(checkpoint)},
                "graph_manifest": {
                    "sha256": sha256_file(manifest),
                    "contract": {
                        "path": str(contract),
                        "sha256": sha256_file(contract),
                    },
                },
            }
            audit = validate_training_asset_binding(
                checkpoint, training_result, manifest
            )
            self.assertTrue(audit["checkpoint_matches_training_result"])
            manifest.write_text("scenario_id\nchanged\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "manifest SHA-256"):
                validate_training_asset_binding(
                    checkpoint, training_result, manifest
                )


if __name__ == "__main__":
    unittest.main()
