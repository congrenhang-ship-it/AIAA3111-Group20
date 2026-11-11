"""负责人 C：A/C 联通、公开交付边界、验证选优与冻结模型复用检查。"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from c_reporting import choose_anomaly, select_combination, compact_report, build_freeze
from run_c_validation import _demo, run_validation
from unsupervised import score_frozen_unsupervised
from main import run_validation as run_g1
from preprocessing import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


class SelectionTests(unittest.TestCase):
    def test_primary_budget_has_priority_over_global_ap(self):
        def metric(tp, ap):
            return {"AP": ap, "budgets": [{"review_fraction": 0.01, "TP": tp}]}

        metrics = {
            "isolation_forest": metric(3, 0.8),
            "kmeans_distance": metric(4, 0.1),
        }
        self.assertEqual(choose_anomaly(metrics), "kmeans_distance")
        metrics["isolation_forest"] = metric(4, 0.8)
        self.assertEqual(choose_anomaly(metrics), "isolation_forest")

    def test_combination_ties_keep_classifier_and_other_budgets_do_not_choose(self):
        rows = [
            {"review_fraction": 0.01, "classifier_share": share, "combined": {"TP": 3}}
            for share in [0, 0.25, 0.5, 0.75, 1]
        ]
        rows.append(
            {"review_fraction": 0.02, "classifier_share": 0, "combined": {"TP": 99}}
        )
        self.assertEqual(select_combination(rows)["classifier_share"], 1)
        rows[2]["combined"]["TP"] = 4
        self.assertEqual(select_combination(rows)["classifier_share"], 0.5)
        self.assertIsNone(select_combination([]))

    def test_public_summary_removes_transaction_lists_and_long_curves(self):
        report = {
            "metadata": {"input_sha256": {"/local/private/data.csv": "abc"}},
            "metrics": {
                "m": {"AP": 0.5, "pr_curve": {"thresholds": [0.2]}, "budgets": []}
            },
            "comparisons": [
                {"classifier_ids": [9], "anomaly_ids": [4], "combined_ids": [9], "K": 1}
            ],
            "anomaly_comparisons": [
                {"classifier_ids": [9], "anomaly_ids": [4], "combined_ids": [9], "K": 1}
            ],
        }
        public = compact_report(report)
        self.assertNotIn("pr_curve", public["metrics"]["m"])
        self.assertEqual(public["metadata"]["input_sha256"], {"data.csv": "abc"})
        self.assertEqual(public["comparisons"], [{"K": 1}])
        self.assertEqual(public["anomaly_comparisons"], [{"K": 1}])
        self.assertIn("classifier_ids", report["comparisons"][0])


class IntegratedWorkflowTests(unittest.TestCase):
    def test_g1_real_c_frame_preserves_integer_metadata(self):
        train = pd.DataFrame(
            np.zeros((4, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
        )
        train = train.assign(Class=[0, 1, 0, 0], row_id=[0, 1, 2, 3], Time=[0, 1, 2, 3])
        val = train.assign(row_id=[4, 5, 6, 7], Time=[4, 5, 6, 7])
        with (
            patch("classification.train_model", return_value=object()),
            patch("classification.score_model", return_value=[0.1, 0.9, 0.2, 0.3]),
        ):
            result = run_g1(train, val)
        self.assertEqual(result["overall"]["AP"], 1)
        self.assertEqual(result["time_windows"][0]["evaluation"]["n_positive"], 1)

    def test_publication_frozen_artifact_and_no_test_claims(self):
        script = ROOT / "code/run_c_validation.py"
        with tempfile.TemporaryDirectory() as folder:
            private = Path(folder) / "private"
            public = Path(folder) / "public"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--demo",
                    "--output",
                    str(private),
                    "--publish",
                    str(public),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((public / "summary.json").read_text())
            config = json.loads((public / "c_freeze.json").read_text())
            self.assertEqual(config["status"], "synthetic_only_not_for_test")
            self.assertFalse(config["test_evaluated"])
            self.assertFalse(config["group_G2_complete"])
            self.assertEqual(
                config["combination"]["status"],
                "validation_selected_requires_B5_provenance",
            )
            self.assertEqual(len(list((public / "figures").glob("*.png"))), 8)
            self.assertFalse(list(public.glob("*predictions*")))
            self.assertNotIn("combined_ids", summary["anomaly_comparisons"][0])
            bundle = joblib.load(private / "c_models.joblib")
            _, val = _demo()
            before = bundle["kmeans"].named_steps["scaler"].mean_.copy()
            scores = score_frozen_unsupervised(bundle, val[FEATURE_COLUMNS])
            for name in ["kmeans_distance", "isolation_forest"]:
                prediction = pd.read_csv(private / f"{name}_predictions.csv")
                assert_allclose(scores[name], prediction.score, rtol=1e-12, atol=1e-12)
            assert_allclose(before, bundle["kmeans"].named_steps["scaler"].mean_)
            with self.assertRaises(ValueError):
                score_frozen_unsupervised(bundle, val[list(reversed(FEATURE_COLUMNS))])
            bundle["config"]["n_clusters"] += 1
            with self.assertRaises(ValueError):
                score_frozen_unsupervised(bundle, val[FEATURE_COLUMNS])
            # A verified real partition can freeze C models, never assert whole-group G2.
            summary["metadata"].update(
                mode="validation", verified_ulb_split=True, classifier_method=None
            )
            summary["comparisons"] = []
            frozen = build_freeze(summary)
            self.assertEqual(frozen["status"], "unsupervised_frozen")
            self.assertEqual(frozen["combination"]["status"], "waiting_for_B5")
            self.assertFalse(frozen["group_G2_complete"])

    def test_existing_public_evidence_rejected_before_fitting(self):
        train, val = _demo()
        with tempfile.TemporaryDirectory() as folder:
            public = Path(folder) / "public"
            public.mkdir()
            (public / "evidence.txt").write_text("keep")
            with patch("run_c_validation.compare_kmeans_candidates") as fit:
                with self.assertRaises(ValueError):
                    run_validation(
                        train, val, Path(folder) / "private", public_output=public
                    )
                fit.assert_not_called()
            self.assertEqual((public / "evidence.txt").read_text(), "keep")

    def test_raw_data_hash_mismatch_does_not_create_results(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "bad.csv"
            source.write_text("bad data")
            output = Path(folder) / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "code/run_c_validation.py"),
                    "--data",
                    str(source),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("SHA-256 mismatch", result.stderr)
            self.assertFalse(output.exists())

    def test_public_and_private_outputs_cannot_overlap(self):
        train, val = _demo()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                run_validation(train, val, folder, public_output=folder)


if __name__ == "__main__":
    unittest.main()
