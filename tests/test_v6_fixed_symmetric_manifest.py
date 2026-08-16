from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jcas.core.poison import ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1
from jcas.core.strict_shadow_folds import _verify_sha_manifest
from jcas.workflows.fixed_symmetric_manifest import build_fixed_symmetric_manifest


PAIR_KEYS = ("scenario_id", "src", "dst", "src_track_id", "dst_track_id")
STRICT_COLUMNS = (
    "scenario_shadow_fold",
    "surrogate_heldout_fold",
    "surrogate_checkpoint_sha256",
    "surrogate_fit_manifest_sha256",
    "surrogate_score_manifest_sha256",
    "surrogate_protocol",
)


def _source_frame() -> pd.DataFrame:
    rows = []
    for index, alpha in enumerate((0.25, 0.75)):
        rows.append(
            {
                "scenario_id": f"scene-{index}",
                "src": index,
                "dst": index + 1,
                "src_track_id": f"src-{index}",
                "dst_track_id": f"dst-{index}",
                "allocation_policy": "crossfit_surrogate_pair_alpha_v4",
                "allocation_alpha": alpha,
                "allocation_total_feature_energy": 0.1 + index,
                "allocation_incident_edge_energy": 0.2 + index,
                "allocation_endpoint_node_energy": 0.3 + index,
                "allocation_non_target_incident_edges": 2 + index,
                "graybox_shadow_fold": index,
                "graybox_clean_probability_mean": 0.1,
                "graybox_triggered_probability_mean": 0.2,
                "graybox_probability_delta_mean": 0.1,
                "graybox_probability_delta_std": 0.01,
                "graybox_response_utility": 0.2,
                "graybox_bce_informative_utility": 0.3,
                "graybox_selection_utility": 0.4,
                "graybox_gradient_alignment_mean": 0.5,
                "graybox_gradient_alignment_std": 0.05,
                "graybox_gradient_alignment_robust": 0.45,
                "graybox_gradient_effect_norm_mean": 0.6,
                "graybox_target_probe_count_min": 3,
                "graybox_selection_objective": "gradient_influence_v4_2",
                "scenario_shadow_fold": index,
                "surrogate_heldout_fold": index,
                "surrogate_checkpoint_sha256": f"checkpoint-{index}",
                "surrogate_fit_manifest_sha256": f"fit-{index}",
                "surrogate_score_manifest_sha256": f"score-{index}",
                "surrogate_protocol": "strict_crossfit_v4_2",
            }
        )
    return pd.DataFrame(rows)


def _candidate_frame() -> pd.DataFrame:
    rows = []
    for index in range(2):
        for alpha in (0.25, 0.5):
            rows.append(
                {
                    "scenario_id": f"scene-{index}",
                    "src": index,
                    "dst": index + 1,
                    "src_track_id": f"src-{index}",
                    "dst_track_id": f"dst-{index}",
                    "scenario_shadow_fold": index,
                    "surrogate_heldout_fold": index,
                    "surrogate_checkpoint_sha256": f"checkpoint-{index}",
                    "surrogate_fit_manifest_sha256": f"fit-{index}",
                    "surrogate_score_manifest_sha256": f"score-{index}",
                    "surrogate_protocol": "strict_crossfit_v4_2",
                    "shadow_fold": index,
                    "alpha": alpha,
                    "clean_probability_mean": 0.10 + index,
                    "triggered_probability_mean": 0.60 + alpha,
                    "delta_mean": 0.50 + alpha,
                    "delta_std": 0.01,
                    "response_utility": 0.70 + alpha,
                    "bce_informative_utility": 0.80 + alpha,
                    "selection_utility": 0.90 + alpha,
                    "total_feature_energy": 1.0 + index + alpha,
                    "incident_edge_energy": 2.0 + index + alpha,
                    "endpoint_node_energy": 3.0 + index + alpha,
                    "non_target_incident_edges": 4 + index,
                    "within_pair_feature_budget": index == 0,
                    "gradient_alignment_mean": 0.4 + alpha,
                    "gradient_alignment_std": 0.04,
                    "gradient_alignment_robust": 0.3 + alpha,
                    "gradient_effect_norm_mean": 0.5 + alpha,
                    "target_probe_count_min": 3,
                }
            )
    return pd.DataFrame(rows)


class FixedSymmetricManifestV6Tests(unittest.TestCase):
    def test_historical_code_manifest_uses_frozen_source_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"frozen historical source\n"
            manifest = root / "code.sha256"
            manifest.write_text(
                f"{hashlib.sha256(payload).hexdigest()}  method.py\n",
                encoding="utf-8",
            )
            archive = root / "source.tar.gz"
            with tarfile.open(archive, mode="w:gz") as bundle:
                member = tarfile.TarInfo("method.py")
                member.size = len(payload)
                bundle.addfile(member, io.BytesIO(payload))

            self.assertEqual(
                _verify_sha_manifest(manifest, source_archive=archive),
                1,
            )

            tampered = root / "tampered.tar.gz"
            with tarfile.open(tampered, mode="w:gz") as bundle:
                changed = b"changed source\n"
                member = tarfile.TarInfo("method.py")
                member.size = len(changed)
                bundle.addfile(member, io.BytesIO(changed))
            with self.assertRaisesRegex(RuntimeError, "frozen asset changed"):
                _verify_sha_manifest(manifest, source_archive=tampered)

    def test_replaces_only_allocation_for_same_pair(self) -> None:
        source = _source_frame()
        candidates = _candidate_frame()

        output, audit, summary = build_fixed_symmetric_manifest(
            source,
            candidates,
        )

        self.assertTrue(output[list(PAIR_KEYS)].equals(source[list(PAIR_KEYS)]))
        self.assertEqual(
            set(output["allocation_policy"]),
            {ALLOCATION_POLICY_FIXED_SYMMETRIC_BIEND_V1},
        )
        np.testing.assert_allclose(output["allocation_alpha"], 0.5)
        expected = candidates[candidates["alpha"].eq(0.5)].sort_values(
            "scenario_id"
        )
        actual = output.sort_values("scenario_id")
        np.testing.assert_allclose(
            actual["allocation_total_feature_energy"],
            expected["total_feature_energy"],
        )
        self.assertTrue(output["v6_source_pair_preserved"].all())
        self.assertEqual(summary["same_pair_preserved_rows"], 2)
        self.assertEqual(summary["outside_source_pair_feature_budget_rows"], 1)
        self.assertEqual(len(audit), len(source))

    def test_rejects_missing_fixed_candidate(self) -> None:
        source = _source_frame()
        candidates = _candidate_frame()
        candidates = candidates[
            ~(
                candidates["scenario_id"].eq("scene-1")
                & candidates["alpha"].eq(0.5)
            )
        ]

        with self.assertRaisesRegex(RuntimeError, "no unique alpha=0.5"):
            build_fixed_symmetric_manifest(source, candidates)

    def test_rejects_duplicate_fixed_candidate(self) -> None:
        source = _source_frame()
        candidates = _candidate_frame()
        duplicate = candidates[
            candidates["scenario_id"].eq("scene-0")
            & candidates["alpha"].eq(0.5)
        ]
        candidates = pd.concat([candidates, duplicate], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate same-pair"):
            build_fixed_symmetric_manifest(source, candidates)


if __name__ == "__main__":
    unittest.main()
