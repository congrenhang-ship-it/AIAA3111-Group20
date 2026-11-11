"""负责人 C：统一指标、训练边界、消融、CLI、模型保存恢复与冻结评价检查。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
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
import json
from unittest.mock import patch
import joblib
from evaluation import choose_anomaly, select_combination, compact_report, build_freeze
from unsupervised import _demo, run_validation
from unsupervised import score_frozen_unsupervised
from main import run_validation as run_g1
from preprocessing import FEATURE_COLUMNS
import copy
from unsupervised import (
    TransactionFeatures,
    configurations,
    candidate_order,
    inner_time_split,
    fit_configuration,
    score_configuration,
    select_families,
    select_pair,
    SHARES,
)
from unsupervised import run_experiment
from evaluation import evaluate_improved_frozen

ROOT = Path(__file__).resolve().parents[1]


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
    def test_shared_cli_improvements_use_original_result_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            public = root / "shared"
            (public / "tables").mkdir(parents=True)
            (public / "tables/B_keep.txt").write_text("untouched")
            cmd = [
                sys.executable,
                str(ROOT / "code/main.py"),
                "--run-c",
                "--demo",
                "--output",
                str(root / "private"),
                "--results",
                str(public),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((public / "tables/c_improved.json").read_text())
            self.assertEqual(summary["config"]["candidate_count"], 20)
            self.assertEqual(len(summary["selection_record"]["inner_per_seed"]), 60)
            self.assertEqual(len(list((public / "figures").glob("c_*.png"))), 6)
            self.assertTrue((public / "tables/c_ablation.csv").is_file())
            self.assertFalse((public / "summary.json").exists())
            self.assertEqual((public / "tables/B_keep.txt").read_text(), "untouched")
            with self.assertRaises(ValueError):
                from evaluation import publish_c_results

                publish_c_results(root / "private/aggregate", public, "improved")

    def test_demo_cli_and_no_overwrite(self):
        import json

        script = Path(__file__).resolve().parents[1] / "code/main.py"
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo"
            cmd = [
                sys.executable,
                str(script),
                "--run-c-baseline",
                "--results",
                str(Path(tmp) / "shared"),
                "--demo",
                "--output",
                str(output),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            report = json.loads((output / "validation_results.json").read_text())
            self.assertEqual(report["metadata"]["mode"], "synthetic_demo")
            self.assertTrue(report["metadata"]["NOT_A_FROZEN_FINAL_CONFIG"])
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 2)

    def test_validation_cli_alignment_with_synthetic_partitions(self):
        import json
        from unsupervised import _demo

        train, val = _demo()
        classifier = make_prediction_frame(val, val.V1, "logistic_test")
        classifier = classifier.iloc[::-1]
        script = Path(__file__).resolve().parents[1] / "code/main.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train.to_csv(root / "train.csv", index=False)
            val.to_csv(root / "validation.csv", index=False)
            classifier.to_csv(root / "classifier.csv", index=False)
            cmd = [
                sys.executable,
                str(script),
                "--run-c-baseline",
                "--results",
                str(Path(tmp) / "shared"),
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
        from unsupervised import run_validation, _demo

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
        from unsupervised import run_validation, _demo

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
    def test_declared_empty_cluster_is_reported_as_na(self):
        from evaluation import (
            make_prediction_frame,
            evaluate_groups,
            save_group_figures,
        )

        data = pd.DataFrame({"row_id": [0, 1], "Time": [0, 1], "Class": [0, 1]})
        frame = make_prediction_frame(data, [0.1, 0.9], "test")
        result = evaluate_groups(frame, [0, 0], [0.01], group_values=[0, 1])
        empty = result["groups"][1]
        self.assertEqual(empty["within_group"]["status"], "empty")
        self.assertIsNone(empty["within_group"]["AP"])
        self.assertEqual(empty["global_budget_allocation"][0]["reviewed_in_group"], 0)
        with tempfile.TemporaryDirectory() as folder:
            self.assertTrue(Path(save_group_figures(result, folder)[0]).is_file())
        with self.assertRaises(ValueError):
            evaluate_groups(frame, [0, 0], [0.01], group_values=[1])

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
        script = ROOT / "code/main.py"
        with tempfile.TemporaryDirectory() as folder:
            private = Path(folder) / "private"
            public = Path(folder) / "public"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--run-c-baseline",
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
            summary = json.loads((public / "tables/c_baseline.json").read_text())
            config = summary["frozen_config"]
            self.assertEqual(config["status"], "synthetic_only_not_for_test")
            self.assertFalse(config["test_evaluated"])
            self.assertFalse(config["group_G2_complete"])
            self.assertEqual(
                config["combination"]["status"],
                "validation_selected_requires_B5_provenance",
            )
            self.assertEqual(len(list((public / "figures").glob("*.png"))), 1)
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
            from evaluation import evaluate_frozen_c

            with (
                patch(
                    "evaluation.choose_anomaly",
                    side_effect=AssertionError("must not select on replay"),
                ),
                patch(
                    "evaluation.select_combination",
                    side_effect=AssertionError("must not select quotas on replay"),
                ),
            ):
                replay = evaluate_frozen_c(
                    bundle, val, Path(folder) / "replay", phase="validation_replay"
                )
            self.assertFalse(replay["configuration_reselected"])
            self.assertEqual(replay["config"], bundle["config"])
            self.assertEqual(
                replay["metrics"]["isolation_forest"]["AP"],
                summary["metrics"]["isolation_forest"]["AP"],
            )
            classifier = pd.read_csv(private / "synthetic_proxy_predictions.csv")
            with (
                patch(
                    "evaluation.choose_anomaly",
                    side_effect=AssertionError("no selection"),
                ),
                patch(
                    "evaluation.select_combination",
                    side_effect=AssertionError("no selection"),
                ),
            ):
                combined = evaluate_frozen_c(
                    bundle,
                    val,
                    Path(folder) / "combined_replay",
                    classifier,
                    phase="validation_replay",
                )
            self.assertEqual(len(combined["comparisons"]), 4)
            self.assertTrue(
                all(
                    c["classifier_share"]
                    == bundle["config"]["combination"]["classifier_share"]
                    for c in combined["comparisons"]
                )
            )
            pending = __import__("copy").deepcopy(bundle)
            pending["config"]["combination"]["classifier_share"] = None
            with self.assertRaises(ValueError):
                evaluate_frozen_c(pending, val, Path(folder) / "blocked", classifier)

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
            with patch("unsupervised.compare_kmeans_candidates") as fit:
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
                    str(ROOT / "code/main.py"),
                    "--run-c-baseline",
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


class ImprovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.val = _demo()

    def test_fixed_candidates_have_no_cartesian_expansion(self):
        specs = configurations()
        self.assertEqual(len(specs), 14)
        self.assertEqual(len(candidate_order()), 20)
        self.assertEqual(sum(s["family"] == "kmeans" for s in specs), 6)
        self.assertEqual(len(set(candidate_order())), 20)
        self.assertEqual(
            [s["max_samples"] for s in specs if s["id"] == "if_sample4096"], [4096]
        )

    def test_features_log_drop_preserve_order_and_input(self):
        X = self.train[FEATURE_COLUMNS].copy()
        before = X.copy(deep=True)
        transformer = TransactionFeatures(True, True).fit(X)
        result = transformer.transform(X.iloc[::-1])
        self.assertNotIn("Time", result)
        assert_allclose(result.Amount, np.log1p(X.Amount.iloc[::-1]))
        self.assertEqual(result.index.tolist(), X.index[::-1].tolist())
        pd.testing.assert_frame_equal(before, X)
        self.assertEqual(len(transformer.get_feature_names_out()), 29)
        for bad in [
            X.assign(Class=0),
            X[list(reversed(FEATURE_COLUMNS))],
            X.assign(Amount=-1),
            X.assign(V1=np.nan),
        ]:
            with self.assertRaises(ValueError):
                transformer.transform(bad)

    def test_inner_split_uses_only_train_and_keeps_timestamp_ties(self):
        train = self.train.copy()
        train.loc[479:481, "Time"] = 479
        first, last = inner_time_split(train)
        self.assertLess(first.Time.max(), last.Time.min())
        self.assertFalse(set(first.row_id) & set(last.row_id))
        self.assertEqual(set(first.row_id) | set(last.row_id), set(train.row_id))
        self.assertFalse(set(last.row_id) & set(self.val.row_id))
        with self.assertRaises(ValueError):
            inner_time_split(train.assign(Time=0))

    def test_relative_distance_is_training_calibrated_and_no_refit(self):
        fitted = fit_configuration(self.train[FEATURE_COLUMNS], configurations()[0])
        stats = copy.deepcopy(fitted["distance_stats"])
        means = fitted["pipeline"]["scaler"].mean_.copy()
        scores, groups = score_configuration(fitted, self.val[FEATURE_COLUMNS])
        expected = np.array(
            [
                (d - stats[g]["median"]) / stats[g]["denominator"]
                for d, g in zip(scores["km_base_raw"], groups, strict=True)
            ]
        )
        assert_allclose(scores["km_base_relative"], expected)
        self.assertTrue(all(s["denominator"] > 0 for s in stats))
        self.assertEqual(stats, fitted["distance_stats"])
        assert_allclose(means, fitted["pipeline"]["scaler"].mean_)
        reverse, _ = score_configuration(fitted, self.val[FEATURE_COLUMNS].iloc[::-1])
        assert_allclose(reverse["km_base_relative"], expected[::-1])
        changed = self.val[FEATURE_COLUMNS].copy()
        changed.iloc[-1, :] += 100
        perturbed, _ = score_configuration(fitted, changed)
        assert_allclose(perturbed["km_base_relative"][:-1], expected[:-1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(fitted, path)
            restored, _ = score_configuration(
                joblib.load(path), self.val[FEATURE_COLUMNS]
            )
            assert_array_equal(restored["km_base_relative"], scores["km_base_relative"])

    def test_inner_selector_uses_mean_and_declaration_order(self):
        order = ["km_base_raw", "km_base_relative", "if_base"]
        rows = [
            {"candidate": n, "seed": s, "TP": tp, "AP": ap}
            for n, tp, ap in [
                ("km_base_raw", 2, 0.4),
                ("km_base_relative", 3, 0.2),
                ("if_base", 1, 0.9),
            ]
            for s in [42, 43, 44]
        ]
        self.assertEqual(
            select_families(rows, order),
            {"kmeans": "km_base_relative", "isolation_forest": "if_base"},
        )
        pairs = [
            {"classifier_share": share, "combined": {"TP": 5}}
            for share in SHARES
            for _ in [42, 43, 44]
        ]
        self.assertEqual(select_pair(pairs), 1)
        for r in pairs:
            if r["classifier_share"] == 0.5:
                r["combined"]["TP"] = 6
        self.assertEqual(select_pair(pairs), 0.5)

    def test_labeled_changes_do_not_enter_model_fit(self):
        spec = configurations()[6]
        first = fit_configuration(self.train[FEATURE_COLUMNS], spec)
        relabeled = self.train.assign(Class=1 - self.train.Class)
        second = fit_configuration(relabeled[FEATURE_COLUMNS], spec)
        a, _ = score_configuration(first, self.val[FEATURE_COLUMNS])
        b, _ = score_configuration(second, self.val[FEATURE_COLUMNS])
        assert_array_equal(a["if_base"], b["if_base"])
        assert_allclose(
            a["if_base"], -first["pipeline"].score_samples(self.val[FEATURE_COLUMNS])
        )

    def test_validation_labels_do_not_choose_models_and_B_replay_uses_frozen_share(
        self,
    ):
        specs = [configurations()[0], configurations()[6]]
        classifier = (
            self.val[["row_id", "Class", "Time"]]
            .assign(method="future_B", score=self.val.V1)
            .iloc[::-1]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, bundle = run_experiment(
                self.train,
                self.val,
                root / "one",
                root / "pub_one",
                specs=specs,
                seeds=(42,),
                classifier=classifier,
                progress=lambda _: None,
            )
            changed = self.val.assign(Class=0)
            second, _ = run_experiment(
                self.train,
                changed,
                root / "two",
                root / "pub_two",
                specs=specs,
                seeds=(42,),
                progress=lambda _: None,
            )
            for key in ["selected_candidates", "selected_anomaly", "distance_share"]:
                self.assertEqual(first["config"][key], second["config"][key])
            self.assertIsNotNone(bundle["config"]["B_combination"]["share"])
            self.assertIn("future_B", first["validation_metrics"])
            self.assertTrue((root / "pub_one/figures/B_complementarity.png").is_file())
            with patch(
                "unsupervised.select_pair",
                side_effect=AssertionError("must keep frozen weights"),
            ):
                replay = evaluate_improved_frozen(
                    bundle,
                    self.val,
                    root / "B_replay",
                    classifier=classifier,
                    phase="validation_replay",
                )
            self.assertEqual(len(replay["B_comparisons"]), 4)
            self.assertTrue(
                all(
                    row["classifier_share"]
                    == bundle["config"]["B_combination"]["share"]
                    for row in replay["B_comparisons"]
                )
            )
            for row in replay["B_comparisons"]:
                self.assertEqual(len(set(row["combined_ids"])), row["K"])

    def test_experiment_public_boundary_and_frozen_replay(self):
        specs = [configurations()[0], configurations()[1], configurations()[6]]
        with tempfile.TemporaryDirectory() as tmp:
            private, public = Path(tmp) / "private", Path(tmp) / "public"
            result, bundle = run_experiment(
                self.train,
                self.val,
                private,
                public,
                specs=specs,
                seeds=(42, 43),
                source_info={"mode": "synthetic_demo"},
                progress=lambda _: None,
            )
            locked = json.loads(
                (public / "selection_locked_before_validation.json").read_text()
            )["config"]
            self.assertEqual(
                locked["selected_candidates"], bundle["config"]["selected_candidates"]
            )
            self.assertEqual(
                locked["distance_share"], bundle["config"]["distance_share"]
            )
            self.assertFalse(locked["test_evaluated"])
            self.assertFalse(locked["group_G2_complete"])
            self.assertNotIn("combined_ids", result["frozen_pair_comparisons"][0])
            self.assertEqual(len(list((public / "figures").glob("*.png"))), 6)
            self.assertFalse(list(public.rglob("*.joblib")))
            self.assertTrue((private / "all_candidate_scores.npz").is_file())
            with (
                patch(
                    "unsupervised.select_families",
                    side_effect=AssertionError("no reselection"),
                ),
                patch(
                    "unsupervised.select_pair",
                    side_effect=AssertionError("no reselection"),
                ),
            ):
                replay = evaluate_improved_frozen(
                    bundle, self.val, Path(tmp) / "replay", phase="validation_replay"
                )
            self.assertFalse(replay["configuration_reselected"])
            for name in bundle["config"]["selected_candidates"].values():
                expected = result["validation_metrics"][name]
                actual = copy.deepcopy(replay["metrics"][name])
                actual.pop("pr_curve", None)
                self.assertEqual(expected, actual)
            self.assertTrue(
                all(
                    r["combined"]["TP"] + r["combined"]["FP"] == r["K"]
                    for r in replay["comparisons"]
                )
            )
            fake = self.val[["row_id", "Class", "Time"]].assign(
                method="future_B", score=self.val.V1
            )
            with self.assertRaises(ValueError):
                evaluate_improved_frozen(
                    bundle, self.val, Path(tmp) / "blocked", classifier=fake
                )
            with patch("unsupervised.fit_configuration") as fit:
                with self.assertRaises(ValueError):
                    run_experiment(
                        self.train,
                        self.val,
                        private,
                        public,
                        specs=specs,
                        progress=lambda _: None,
                    )
                fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
