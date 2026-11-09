"""负责人 C：人工可核算指标、数据边界、模型管线和 CLI 集成检查。"""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from evaluation import (
    make_prediction_frame,
    evaluate_scores,
    compare_review_lists,
    evaluate_groups,
    save_figures,
    save_group_figures,
    save_comparison_figures,
)
from unsupervised import (
    fit_kmeans,
    score_kmeans,
    fit_isolation_forest,
    score_isolation_forest,
    compare_kmeans_candidates,
)


def frame(
    scores=(0.9, 0.8, 0.7, 0.6), labels=(1, 0, 1, 0), ids=(0, 1, 2, 3), method="test"
):
    data = pd.DataFrame(
        {"row_id": ids, "Class": labels, "Time": np.asarray(ids, dtype=float)},
        index=np.arange(len(ids)) + 20,
    )
    return make_prediction_frame(
        data, pd.Series(scores, index=np.arange(len(ids)) + 100), method
    )


class EvaluationTests(unittest.TestCase):
    def test_positional_frame_and_metadata(self):
        p = frame()
        self.assertEqual(p.score.tolist(), [0.9, 0.8, 0.7, 0.6])
        self.assertEqual(p.row_id.tolist(), [0, 1, 2, 3])
        self.assertEqual(list(p), ["row_id", "Class", "Time", "method", "score"])

    def test_hand_computed_ap_and_budget(self):
        p = frame()
        r = evaluate_scores(
            p.Class, p.score, [0.25, 0.5, 1], row_ids=p.row_id, threshold=0.8
        )
        self.assertAlmostEqual(r["AP"], 5 / 6)
        self.assertEqual(r["budgets"][1]["TP"], 1)
        self.assertEqual(r["budgets"][1]["FP"], 1)
        self.assertEqual(r["budgets"][1]["recall_at_k"], 0.5)
        self.assertEqual(r["threshold"]["TN"], 1)
        self.assertEqual(r["threshold"]["FN"], 1)
        self.assertEqual(r["budgets"][1]["random_expected_TP"], 1)

    def test_ties_are_input_order_independent(self):
        r = evaluate_scores([0, 1, 0], [1, 1, 1], [0.1], row_ids=[9, 2, 5])
        self.assertEqual(r["budgets"][0]["TP"], 1)
        self.assertEqual(r["budgets"][0]["K"], 1)
        other = evaluate_scores([0, 0, 1], [1, 1, 1], [0.1], row_ids=[5, 9, 2])
        self.assertEqual(r, other)

    def test_ap_ties_are_grouped_not_artificially_broken(self):
        r = evaluate_scores([1, 0], [1, 1], [0.5], row_ids=[0, 1])
        self.assertEqual(r["AP"], 0.5)
        self.assertEqual(r["budgets"][0]["TP"], 1)

    def test_no_positive_empty_all_positive(self):
        r = evaluate_scores([0, 0], [0.1, 0.2], [0.5], row_ids=[0, 1])
        self.assertIsNone(r["AP"])
        self.assertIsNone(r["budgets"][0]["recall_at_k"])
        self.assertEqual(r["budgets"][0]["precision_at_k"], 0)
        r = evaluate_scores([], [], [0.5], row_ids=[])
        self.assertEqual(r["status"], "empty")
        self.assertEqual(r["budgets"][0]["K"], 0)
        self.assertIsNone(r["budgets"][0]["precision_at_k"])
        self.assertEqual(evaluate_scores([1, 1], [0.1, 0.2], row_ids=[0, 1])["AP"], 1)

    def test_invalid_inputs(self):
        cases = [
            ([0, 2], [0.1, 0.2], [0, 1]),
            ([0, 1], [0.1, float("nan")], [0, 1]),
            ([0, 1], [0.1, 0.2], [1, 1]),
            ([0, 1], [0.1, 0.2], [0.1, 1]),
            ([0, 1], [0.1], [0, 1]),
            ([0, 1], [[1], [2]], [0, 1]),
        ]
        for y, s, ids in cases:
            with self.subTest(y=y, s=s, ids=ids), self.assertRaises(ValueError):
                evaluate_scores(y, s, row_ids=ids)
        for budgets in [[], [0], [-1], [1.1], [0.1, 0.1]]:
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                evaluate_scores([0, 1], [0.1, 0.2], budgets, row_ids=[0, 1])
        with self.assertRaises(ValueError):
            frame(method="")
        with self.assertRaises(ValueError):
            make_prediction_frame(frame(), [1, 2, 3, 4], "x", [0, 1])

    def test_complementarity_reorders_and_keeps_exact_budget(self):
        c = frame()
        a = frame([0.1, 0.2, 0.9, 0.8], method="anomaly").iloc[::-1]
        result = compare_review_lists(c, a, 2, 0.5)
        self.assertEqual(result["combined_ids"], [0, 2])
        self.assertEqual(result["combined"]["TP"], 2)
        self.assertEqual(result["shared_frauds"], 0)
        self.assertEqual(result["classifier_only_frauds"], 1)
        self.assertEqual(result["anomaly_only_frauds"], 1)
        self.assertEqual(compare_review_lists(c, a, 2, 0)["combined_ids"], [2, 3])
        self.assertEqual(compare_review_lists(c, a, 2, 1)["combined_ids"], [0, 1])

    def test_overlap_refills_and_all_k_values(self):
        c = frame()
        a = frame(method="a")
        for k in range(1, 5):
            for share in [0, 0.25, 0.5, 0.75, 1]:
                r = compare_review_lists(c, a, k, share)
                self.assertEqual(len(set(r["combined_ids"])), k)
                self.assertEqual(r["combined"]["TP"] + r["combined"]["FP"], k)
        self.assertEqual(compare_review_lists(c, a, 3, 0.5)["combined_ids"], [0, 1, 2])

    def test_label_changes_do_not_change_selection(self):
        c = frame()
        a = frame([0.1, 0.2, 0.9, 0.8], method="a")
        ids = compare_review_lists(c, a, 3, 0.5)["combined_ids"]
        c.Class = 1 - c.Class
        a.Class = 1 - a.Class
        self.assertEqual(ids, compare_review_lists(c, a, 3, 0.5)["combined_ids"])

    def test_mismatch_rejected(self):
        c = frame()
        a = frame(method="a")
        a.loc[0, "Class"] = 0
        with self.assertRaises(ValueError):
            compare_review_lists(c, a, 2)
        a = frame(method="a")
        a.loc[0, "Time"] = 99
        with self.assertRaises(ValueError):
            compare_review_lists(c, a, 2)
        a = frame(ids=[4, 5, 6, 7])
        with self.assertRaises(ValueError):
            compare_review_lists(c, a, 2)
        for k in [0, 5, 1.2, True]:
            with self.assertRaises(ValueError):
                compare_review_lists(c, c, k)
        for share in [-0.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                compare_review_lists(c, c, 2, share)

    def test_group_budgets_distinguished_from_global(self):
        p = frame()
        groups = pd.Series([1, 0, 1, 0], index=[3, 0, 2, 1])
        r = evaluate_groups(p, groups, [0.25])
        self.assertEqual(
            sum(g["within_group"]["budgets"][0]["K"] for g in r["groups"]), 2
        )
        self.assertEqual(
            sum(
                g["global_budget_allocation"][0]["reviewed_in_group"]
                for g in r["groups"]
            ),
            1,
        )
        self.assertEqual(sum(g["fraud_coverage"] for g in r["groups"]), 1)
        with self.assertRaises(ValueError):
            evaluate_groups(p, pd.Series([1, 2], index=[0, 1]), [0.5])
        with self.assertRaises(ValueError):
            evaluate_groups(p, [0, 0, None, 1], [0.5])

    def test_plot_files(self):
        p = frame()
        r = evaluate_scores(p.Class, p.score, [0.5, 1], row_ids=p.row_id)
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_figures({"methods": {"x": r}, "label": "SYNTHETIC CHECK"}, tmp)
            paths += save_group_figures(
                evaluate_groups(p, [0, 0, 1, 1], [0.5]), tmp, "SYNTHETIC CHECK"
            )
            paths += save_comparison_figures(
                [compare_review_lists(p, p, 2)], tmp, "SYNTHETIC CHECK"
            )
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(Path(x).stat().st_size > 1000 for x in paths))


class UnsupervisedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.X = pd.DataFrame(
            np.vstack([rng.normal(-3, 0.2, (25, 2)), rng.normal(3, 0.2, (25, 2))]),
            columns=["V1", "V2"],
        )

    def test_training_only_scaler_and_no_refit(self):
        model = fit_kmeans(self.X, 2)
        mean = model["scaler"].mean_.copy()
        centers = model["model"].cluster_centers_.copy()
        ids, distances = score_kmeans(
            model, pd.DataFrame([[20.0, 20.0]], columns=self.X.columns)
        )
        assert_allclose(mean, self.X.mean().to_numpy())
        assert_allclose(mean, model["scaler"].mean_)
        assert_allclose(centers, model["model"].cluster_centers_)
        self.assertEqual(len(ids), 1)
        self.assertGreater(distances[0], 0)

    def test_distance_formula_and_order(self):
        model = fit_kmeans(self.X, 2)
        groups, d = score_kmeans(model, self.X)
        z = model["scaler"].transform(self.X)
        assert_allclose(
            d,
            np.min(
                np.linalg.norm(
                    z[:, None, :] - model["model"].cluster_centers_[None, :, :], axis=2
                ),
                axis=1,
            ),
        )
        _, reverse = score_kmeans(model, self.X.iloc[::-1])
        assert_allclose(d[::-1], reverse)
        self.assertEqual(len(score_kmeans(model, self.X.iloc[:0])[0]), 0)

    def test_forest_direction_and_extreme_point(self):
        model = fit_isolation_forest(self.X, max_samples=50, n_estimators=30)
        points = pd.DataFrame([[-3.0, -3.0], [50.0, 50.0]], columns=self.X.columns)
        assert_allclose(
            score_isolation_forest(model, points), -model.score_samples(points)
        )
        self.assertGreater(
            score_isolation_forest(model, points)[1],
            score_isolation_forest(model, points)[0],
        )
        assert_array_equal(
            score_isolation_forest(model, points), score_isolation_forest(model, points)
        )

    def test_reject_metadata_nan_and_wrong_columns(self):
        for column in ["Class", "row_id", "score", "cluster_id"]:
            bad = self.X.assign(**{column: 0})
            with self.assertRaises(ValueError):
                fit_kmeans(bad, 2)
            with self.assertRaises(ValueError):
                fit_isolation_forest(bad)
        bad = self.X.copy()
        bad.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_kmeans(bad, 2)
        model = fit_kmeans(self.X, 2)
        with self.assertRaises(ValueError):
            score_kmeans(model, self.X[["V2", "V1"]])

    def test_structural_metrics_and_stability(self):
        rows = compare_kmeans_candidates(
            self.X, ks=(2, 3), seeds=(1, 2), sample_size=25
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(
            all(
                r["sample_size"] == 25 and r["sample_silhouette"] is not None
                for r in rows
            )
        )
        self.assertTrue(all(-1 <= r["seed_stability_ARI"] <= 1 for r in rows))
        self.assertEqual(
            rows,
            compare_kmeans_candidates(self.X, ks=(2, 3), seeds=(1, 2), sample_size=25),
        )


class RunnerTests(unittest.TestCase):
    def test_demo_cli_and_no_overwrite(self):
        import json

        script = Path(__file__).resolve().parents[1] / "code/run_c_validation.py"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo"
            cmd = [sys.executable, str(script), "--demo", "--output", str(output)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            report = json.loads((output / "validation_results.json").read_text())
            self.assertEqual(report["metadata"]["mode"], "synthetic_demo")
            self.assertTrue(report["metadata"]["NOT_A_FROZEN_FINAL_CONFIG"])
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 2)

    def test_validation_cli_alignment_with_synthetic_partitions(self):
        import json
        from run_c_validation import _demo

        train, val = _demo()
        classifier = make_prediction_frame(val, val.V1, "logistic_test")
        classifier = classifier.iloc[::-1]
        script = Path(__file__).resolve().parents[1] / "code/run_c_validation.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train.to_csv(root / "train.csv", index=False)
            val.to_csv(root / "validation.csv", index=False)
            classifier.to_csv(root / "classifier.csv", index=False)
            cmd = [
                sys.executable,
                str(script),
                "--train",
                str(root / "train.csv"),
                "--validation",
                str(root / "validation.csv"),
                "--classification-predictions",
                str(root / "classifier.csv"),
                "--output",
                str(root / "output"),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "output/validation_results.json").read_text())
            self.assertEqual(report["metadata"]["mode"], "validation")
            self.assertEqual(len(report["metadata"]["input_sha256"]), 3)
            out = pd.read_csv(root / "output/logistic_test_predictions.csv")
            assert_array_equal(out.row_id, val.row_id)
            assert_allclose(out.score, val.V1)
            reference = report["groups"]["isolation_forest"]["groups"]
            actual = report["groups"]["logistic_test"]["groups"]
            self.assertEqual(
                [(g["cluster_id"], g["within_group"]["n_positive"]) for g in actual],
                [(g["cluster_id"], g["within_group"]["n_positive"]) for g in reference],
            )
            self.assertEqual(len(report["comparisons"]), 20)
            self.assertTrue(
                all(
                    len(set(r["combined_ids"])) == r["K"] for r in report["comparisons"]
                )
            )

    def test_runner_rejects_ambiguous_classifier_names(self):
        from run_c_validation import run_validation, _demo

        train, val = _demo()
        classifier = make_prediction_frame(val, np.zeros(len(val)), "classifier")
        with tempfile.TemporaryDirectory() as tmp:
            for name in ["../escape", "isolation_forest", None]:
                bad = classifier.copy()
                bad["method"] = name
                with self.subTest(name=name), self.assertRaises(ValueError):
                    run_validation(train, val, Path(tmp) / "bad", bad)
            bad = classifier.copy()
            bad.loc[0, "method"] = "other"
            with self.assertRaises(ValueError):
                run_validation(train, val, Path(tmp) / "mixed", bad)

    def test_runner_rejects_overlapping_or_nonchronological_partitions(self):
        from run_c_validation import run_validation, _demo

        train, val = _demo()
        with tempfile.TemporaryDirectory() as tmp:
            bad = val.copy()
            bad.row_id = np.arange(len(bad))
            with self.assertRaises(ValueError):
                run_validation(train, bad, Path(tmp) / "a")
            bad = val.copy()
            bad.Time = 0
            with self.assertRaises(ValueError):
                run_validation(train, bad, Path(tmp) / "b")


if __name__ == "__main__":
    unittest.main()
