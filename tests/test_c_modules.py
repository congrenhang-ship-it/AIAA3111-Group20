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


class BCompletionTests(unittest.TestCase):
    def test_B_reproduction_checks_identity_parameters_and_persistence(self):
        from classification import train_model, score_model
        from evaluation import partition_description
        from unsupervised import reproduce_b_for_c

        train, val = _demo()
        model = train_model(
            train[FEATURE_COLUMNS],
            train.Class,
            "logistic_regression",
            strategy="class_weight",
        )
        manifest = {
            "owner": "B",
            "test_evaluated": False,
            "dataset_sha256": "unit_digest",
            "partitions": {
                "train": partition_description(train),
                "validation": partition_description(val),
            },
            "feature_columns": FEATURE_COLUMNS,
            "seed": 42,
            "method": "logistic_regression_class_weight",
            "model_name": "logistic_regression",
            "strategy": "class_weight",
            "classifier_parameters": model.named_steps["classifier"].get_params(),
            "provenance": {"source_commit": "fixture", "versions": {}},
            "predictions_sha256": "fixture",
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            freeze = root / "b_freeze.json"
            freeze.write_text(json.dumps(manifest))
            original_read_text = Path.read_text

            def windows_read_text(path, *args, **kwargs):
                # Reproduce Windows GBK defaults while preserving explicit encodings.
                kwargs.setdefault("encoding", "gbk")
                return original_read_text(path, *args, **kwargs)

            with patch(
                "run_b_validation.EXPERIMENTS",
                [
                    ("logistic_regression", "original"),
                    ("logistic_regression", "class_weight"),
                ],
            ), patch.object(Path, "read_text", windows_read_text):
                prediction, fitted, record = reproduce_b_for_c(
                    train, val, freeze, root / "out", "unit_digest"
                )
            self.assertTrue(record["parameters_match_published"])
            self.assertTrue(record["model_replay_scores_and_full_ranking_match"])
            self.assertEqual(len(record["experiments"]), 2)
            assert_allclose(prediction.score, score_model(fitted, val[FEATURE_COLUMNS]))
            self.assertEqual(json.loads(freeze.read_text()), manifest)
            with patch("unsupervised.importlib.metadata.version", return_value="wrong"):
                with self.assertRaisesRegex(ValueError, "pinned shared environment"):
                    reproduce_b_for_c(
                        train, val, freeze, root / "wrong_environment", "unit_digest"
                    )
            with patch(
                "classification.train_model",
                side_effect=AssertionError("identity must be checked before fit"),
            ):
                with self.assertRaises(ValueError):
                    reproduce_b_for_c(train, val, freeze, root / "bad", "wrong_digest")
                changed = val.copy()
                changed.row_id += 10000
                with self.assertRaises(ValueError):
                    reproduce_b_for_c(
                        train, changed, freeze, root / "bad", "unit_digest"
                    )

    def test_frozen_C_bundle_scores_its_persisted_B_without_refitting(self):
        from classification import train_model, score_model
        from unsupervised import configurations, fit_configuration
        from evaluation import evaluate_improved_frozen

        train, val = _demo()
        selected = {"kmeans": "km_base_raw", "isolation_forest": "if_base"}
        B = train_model(
            train[FEATURE_COLUMNS],
            train.Class,
            "logistic_regression",
            strategy="class_weight",
        )
        bundle = {
            "models": {
                "kmeans": fit_configuration(
                    train[FEATURE_COLUMNS], configurations()[0]
                ),
                "isolation_forest": fit_configuration(
                    train[FEATURE_COLUMNS], configurations()[6]
                ),
            },
            "B_model": B,
            "config": {
                "selected_candidates": selected,
                "selected_anomaly": "if_base",
                "distance_share": 0.0,
                "B_combination": {"method": "B_actual", "share": 0.75},
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch(
                    "classification.train_model", side_effect=AssertionError("no refit")
                ),
                patch(
                    "unsupervised.select_pair",
                    side_effect=AssertionError("no reselection"),
                ),
            ):
                replay = evaluate_improved_frozen(
                    bundle, val, folder, phase="validation_replay",
                    window_seconds=100, window_origin=7,
                )
            self.assertIn("B_actual", replay["metrics"])
            self.assertEqual(len(replay["B_comparisons"]), 4)
            self.assertTrue(
                all(
                    len(set(r["combined_ids"])) == r["K"]
                    and r["classifier_share"] == 0.75
                    for r in replay["B_comparisons"]
                )
            )
            written = pd.read_csv(Path(folder) / "B_actual_predictions.csv")
            assert_allclose(written.score, score_model(B, val[FEATURE_COLUMNS]))
            self.assertEqual(replay["time_window_config"]["origin"], 7)
            self.assertEqual(replay["time_window_config"]["window_seconds"], 100)
            from temporal import evaluate_time_windows
            expected = evaluate_time_windows(written, window_seconds=100, origin=7)
            actual = [r for r in replay["time_windows"] if r["method"] == "B_actual"]
            for row in expected:
                if row["evaluation"] is not None:
                    row["evaluation"].pop("pr_curve", None)
            self.assertEqual(actual, expected)

            from evaluation import publish_frozen_c_results
            with tempfile.TemporaryDirectory() as public_folder:
                provenance = {"source_commit": "fixture", "dataset_sha256": "fixture",
                              "command": "synthetic validation replay"}
                paths = publish_frozen_c_results(folder, public_folder, provenance=provenance)
                self.assertEqual(len(paths), 8)  # JSON, writing aid, six PNG figures
                summary_path = Path(public_folder) / "tables/c_validation_replay.json"
                summary = json.loads(summary_path.read_text())
                self.assertEqual(summary["phase"], "validation_replay")
                self.assertEqual(summary["time_window_config"], replay["time_window_config"])
                self.assertEqual(summary["partition"], replay["partition"])
                for metric in summary["metrics"].values():
                    self.assertNotIn("pr_curve", metric)
                for key in ["comparisons", "B_comparisons"]:
                    for row in summary[key]:
                        self.assertFalse(any(k.endswith("_ids") for k in row))
                self.assertIn("combined_ids", replay["B_comparisons"][0])
                import hashlib
                self.assertEqual(summary["publication"]["evaluation_sha256"],
                                 hashlib.sha256((Path(folder) / "evaluation.json").read_bytes()).hexdigest())
                for filename, digest in summary["publication"]["figure_sha256"].items():
                    self.assertEqual(hashlib.sha256((Path(public_folder) / filename).read_bytes()).hexdigest(), digest)
                writing = (Path(public_folder) / "tables/c_validation_replay.md").read_text()
                self.assertIn("Validation replay", writing)
                self.assertIn("B_actual", writing)
                self.assertIn("N/A", writing)  # declared empty clusters
                before = summary_path.read_bytes()
                with self.assertRaisesRegex(ValueError, "already exist"):
                    publish_frozen_c_results(folder, public_folder, provenance=provenance)
                self.assertEqual(summary_path.read_bytes(), before)
                with self.assertRaisesRegex(ValueError, "overlap"):
                    publish_frozen_c_results(folder, Path(folder) / "public", provenance=provenance)
                with self.assertRaisesRegex(ValueError, "provenance"):
                    publish_frozen_c_results(folder, Path(public_folder) / "missing", provenance={})
                # A's G2 reference is required for traceable test publication; no scoring here.
                source = Path(folder) / "evaluation.json"
                recorded = json.loads(source.read_text())
                recorded["phase"] = "test"
                source.write_text(json.dumps(recorded))
                with self.assertRaisesRegex(ValueError, "g2_reference"):
                    publish_frozen_c_results(folder, Path(public_folder) / "test", provenance=provenance)
                self.assertFalse((Path(public_folder) / "test").exists())
                source.write_text(json.dumps(replay))
            bundle["config"]["B_combination"]["share"] = None
            with self.assertRaises(ValueError):
                evaluate_improved_frozen(bundle, val, Path(folder) / "blocked")

    def test_frozen_windows_reject_invalid_settings_before_scoring(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "unsupervised.score_improved_bundle", side_effect=AssertionError("must not score")
        ):
            for seconds, origin in [(0, 0), (-1, 0), (float("nan"), 0),
                                    (100, float("inf"))]:
                with self.subTest(seconds=seconds, origin=origin), self.assertRaises(ValueError):
                    evaluate_improved_frozen({}, pd.DataFrame(), folder,
                                             window_seconds=seconds, window_origin=origin,
                                             phase="validation_replay")



class CExtensionMetricTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "row_id": [10, 11, 12, 13], "Class": [1, 0, 1, 0],
            "Time": [0.0, 1.0, 2.0, 3.0], "Amount": [100.0, 1000.0, 10.0, 0.0],
        })
        self.p = make_prediction_frame(self.data, [0.9, 0.8, 0.7, 0.6], "classifier")
        self.a = make_prediction_frame(self.data, [0.1, 0.4, 0.9, 0.8], "anomaly")

    def test_amounts_align_by_id_and_global_group_sums(self):
        from evaluation import fit_amount_bins, evaluate_amounts
        bins = fit_amount_bins(self.data)
        result = evaluate_amounts(self.p.iloc[::-1], self.data, bins, [0.5])
        row = result["budgets"][0]
        self.assertEqual((row["K"], row["TP"], row["FP"]), (2, 1, 1))
        self.assertAlmostEqual(row["amount_coverage"], 100/110)
        self.assertEqual(row["precision_lift"], 1)
        self.assertEqual(sum(g["n_samples"] for g in row["amount_groups"]), 4)
        self.assertEqual(sum(g["reviewed_from_global_list"] for g in row["amount_groups"]), 2)
        self.assertEqual(sum(g["fraud_amount_detected"] for g in row["amount_groups"]), 100)
        self.assertEqual(sum(g["transaction_amount"] for g in row["amount_groups"]), 1110)
        original = evaluate_scores(self.p.Class, self.p.score, [0.5], row_ids=self.p.row_id)
        self.assertEqual(result["AP"], original["AP"])
        self.assertEqual(bins["cuts"], fit_amount_bins(self.data.assign(Class=1))["cuts"])

    def test_duplicate_cuts_zero_fraud_amount_and_empty_groups(self):
        from evaluation import fit_amount_bins, evaluate_amounts
        bins = fit_amount_bins(self.data.assign(Amount=10))
        self.assertEqual(bins["cuts"], [10])
        zero = self.data.assign(Amount=0)
        result = evaluate_amounts(self.p, zero, bins, [0.5])["budgets"][0]
        self.assertIsNone(result["amount_coverage"])
        self.assertEqual(result["amount_groups"][1]["n_samples"], 0)
        self.assertIsNone(result["amount_groups"][1]["recall"])
        no_fraud = self.data.assign(Class=0)
        p = make_prediction_frame(no_fraud, self.p.score, "classifier")
        result = evaluate_amounts(p, no_fraud, bins, [0.5])
        self.assertIsNone(result["AP"])
        self.assertIsNone(result["budgets"][0]["precision_lift"])

    def test_monetary_metadata_and_amount_errors_rejected(self):
        from evaluation import fit_amount_bins, evaluate_amounts
        bins = fit_amount_bins(self.data)
        for data in [self.data.assign(Amount=-1), self.data.assign(Amount=np.nan),
                     self.data.assign(Class=0), self.data.assign(Time=99),
                     self.data.assign(row_id=[10, 10, 12, 13]), self.data.drop(columns="Amount")]:
            with self.subTest(columns=list(data)), self.assertRaises(ValueError):
                evaluate_amounts(self.p, data, bins)
        for cuts in [[1, 1], [-1], [2, 1], [float("nan")]]:
            with self.assertRaises(ValueError):
                evaluate_amounts(self.p, self.data, {"cuts": cuts})
        with self.assertRaises(ValueError):
            fit_amount_bins(self.data.iloc[:0])
        with self.assertRaises(ValueError):
            fit_amount_bins(self.data.assign(Amount=-1))

    def test_costs_use_FP_and_FN_with_no_budget_selection(self):
        from evaluation import cost_sensitivity
        metric = evaluate_scores(self.p.Class, self.p.score, [0.5], row_ids=self.p.row_id)
        rows = cost_sensitivity({"m": metric})
        self.assertEqual([r["hypothetical_loss"] for r in rows], [11, 51, 101])
        self.assertTrue(all(r["K"] == 2 for r in rows))
        for ratios in [[0], [10, 10], [float("inf")], []]:
            with self.assertRaises(ValueError):
                cost_sensitivity({"m": metric}, ratios)

    def test_weighted_AP_and_topK_match_explicit_bootstrap_instances(self):
        from evaluation import _weighted_rank_metric, _order
        from sklearn.metrics import average_precision_score
        import itertools
        # Exhaust all 3^3 draws, including duplicated IDs, score ties, zero weights,
        # all-normal samples and different empirical positive proportions.
        y = np.array([1, 0, 1])
        scores = np.array([0.8, 0.8, 0.2])
        ids = np.array([20, 10, 30])
        order = _order(scores, ids)
        ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(y)-1]
        for draw in itertools.product(range(3), repeat=3):
            draw = np.asarray(draw)
            counts = np.bincount(draw, minlength=3)
            actual = _weighted_rank_metric(y, scores, counts, order, ends, 0.5)
            occurrence_order = np.lexsort((np.arange(3), ids[draw], -scores[draw]))
            expected_tp = int(y[draw][occurrence_order[:2]].sum())
            self.assertEqual(actual["TP"], expected_tp)
            self.assertEqual(actual["K"], 2)
            if y[draw].sum():
                self.assertAlmostEqual(actual["AP"], average_precision_score(y[draw], scores[draw]))
                self.assertEqual(actual["recall"], expected_tp/int(y[draw].sum()))
            else:
                self.assertIsNone(actual["AP"])
                self.assertIsNone(actual["recall"])

    def test_pairing_reproducibility_sign_symmetry_and_endpoint_zero_gain(self):
        from evaluation import paired_uncertainty
        frames = {"classifier": self.p, "anomaly": self.a.iloc[::-1]}
        policy = {"name": "policy", "classifier": "classifier", "anomaly": "anomaly", "share": 1.0}
        kwargs = dict(q=0.5, row_resamples=50, block_resamples=30, block_seconds=2,
                      combination=policy, pairs=[("classifier", "anomaly"), ("anomaly", "classifier"), ("policy", "classifier")])
        first = paired_uncertainty(frames, **kwargs)
        self.assertEqual(first, paired_uncertainty(frames, **kwargs))
        for scheme in first["schemes"].values():
            for metric in ["AP", "recall"]:
                zero = scheme["differences"]["policy minus classifier"][metric]
                self.assertEqual(zero["lower"], 0)
                self.assertEqual(zero["upper"], 0)
                self.assertTrue(zero["degenerate"])
                left = scheme["differences"]["classifier minus anomaly"][metric]
                right = scheme["differences"]["anomaly minus classifier"][metric]
                self.assertAlmostEqual(left["lower"], -right["upper"])
                self.assertAlmostEqual(left["upper"], -right["lower"])
        self.assertEqual(first["point_estimates"]["classifier"]["K"], 2)

    def test_interior_quota_replicates_match_public_list_policy_and_have_no_AP(self):
        from evaluation import paired_uncertainty
        policy = {"name": "policy", "classifier": "classifier", "anomaly": "anomaly", "share": 0.5}
        result = paired_uncertainty({"classifier": self.p, "anomaly": self.a},
                                   combination=policy, pairs=[("policy", "classifier")],
                                   q=0.5, row_resamples=20, block_resamples=10, block_seconds=2)
        expected = compare_review_lists(self.p, self.a, 2, 0.5)
        self.assertEqual(result["point_estimates"]["policy"]["TP"], expected["combined"]["TP"])
        self.assertIsNone(result["point_estimates"]["policy"]["AP"])
        for scheme in result["schemes"].values():
            self.assertEqual(scheme["methods"]["policy"]["AP"]["valid_replicates"], 0)
            self.assertIsNone(scheme["differences"]["policy minus classifier"]["AP"]["lower"])
        # Independently enumerate the sampled multiset through the existing
        # list-policy implementation; compare its interval with the new kernel.
        rng = np.random.default_rng(42)
        recalls = []
        for _ in range(20):
            draw = rng.integers(4, size=4)
            draw = draw[np.lexsort((np.arange(4), self.data.row_id.to_numpy()[draw]))]
            sampled = self.data.iloc[draw].reset_index(drop=True).assign(row_id=np.arange(4))
            cp = make_prediction_frame(sampled, self.p.score.to_numpy()[draw], "c")
            ap = make_prediction_frame(sampled, self.a.score.to_numpy()[draw], "a")
            row = compare_review_lists(cp, ap, 2, 0.5)["combined"]
            if row["recall_at_k"] is not None: recalls.append(row["recall_at_k"])
        actual = result["schemes"]["paired_rows"]["methods"]["policy"]["recall"]
        self.assertAlmostEqual(actual["lower"], np.quantile(recalls, 0.025))
        self.assertAlmostEqual(actual["upper"], np.quantile(recalls, 0.975))

    def test_block_gaps_empty_resamples_and_no_positive_replicates_reported(self):
        from evaluation import paired_uncertainty
        data = self.data.iloc[:2].assign(Time=[0, 20], Class=[0, 0])
        p = make_prediction_frame(data, [1, 0], "m")
        result = paired_uncertainty({"m": p}, row_resamples=20, block_resamples=100,
                                   q=0.5, block_seconds=10)
        self.assertEqual(result["block_config"]["n_blocks"], 3)
        self.assertEqual(result["block_config"]["empty_blocks"], 1)
        blocks = result["schemes"]["paired_time_blocks"]
        self.assertGreater(blocks["empty_replicates"], 0)
        self.assertNotEqual(*blocks["sample_size_range"])
        self.assertEqual(blocks["empty_replicates"] + blocks["no_positive_replicates"], 100)
        self.assertEqual(blocks["methods"]["m"]["AP"]["valid_replicates"], 0)
        self.assertIsNone(blocks["methods"]["m"]["recall"]["lower"])

    def test_uncertainty_rejects_invalid_alignment_and_parameters(self):
        from evaluation import paired_uncertainty
        for kwargs in [dict(row_resamples=0), dict(block_seconds=0), dict(seed=-1),
                       dict(q=0), dict(pairs=[("missing", "classifier")])]:
            with self.assertRaises(ValueError):
                paired_uncertainty({"classifier": self.p}, **kwargs)
        with self.assertRaises(ValueError):
            paired_uncertainty({"classifier": self.p, "anomaly": self.a.assign(Class=0)})


class CExtensionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from classification import train_model
        from evaluation import partition_description, build_c_extension_config
        cls.train, cls.val = _demo()
        cls.bundle = {
            "models": {"kmeans": fit_configuration(cls.train[FEATURE_COLUMNS], configurations()[0]),
                       "isolation_forest": fit_configuration(cls.train[FEATURE_COLUMNS], configurations()[6])},
            "B_model": train_model(cls.train[FEATURE_COLUMNS], cls.train.Class,
                                   "logistic_regression", strategy="class_weight"),
            "config": {"partitions": {"train": partition_description(cls.train),
                                        "validation": partition_description(cls.val)},
                       "selected_candidates": {"kmeans": "km_base_raw", "isolation_forest": "if_base"},
                       "selected_anomaly": "if_base", "B_combination": {"method": "B", "share": 1.0}},
        }
        cls.config = build_c_extension_config(cls.train, cls.bundle)
        cls.config["bootstrap"].update(row_resamples=8, block_resamples=6, block_seconds=100)

    def test_full_extensions_publish_aggregates_replay_and_refuse_overwrite(self):
        from evaluation import run_c_extensions
        before = joblib.hash(self.bundle)
        with tempfile.TemporaryDirectory() as folder, patch(
            "classification.train_model", side_effect=AssertionError("no fitting")
        ), patch("unsupervised.fit_configuration", side_effect=AssertionError("no fitting")):
            root = Path(folder)
            first = run_c_extensions(self.bundle, self.val, self.config, root/"one",
                                     results=root/"shared", progress=None)
            replay = run_c_extensions(self.bundle, self.val, self.config, root/"two", progress=None)
            self.assertEqual(first["metrics"], replay["metrics"])
            self.assertEqual(first["uncertainty"], replay["uncertainty"])
            self.assertFalse(first["test_evaluated"])
            self.assertFalse(first["group_G2_complete"])
            self.assertEqual(first["metrics"]["B"]["AP"], first["metrics"]["frozen_B_IF_policy"]["AP"])
            for base, policy in zip(first["metrics"]["B"]["budgets"],
                                    first["metrics"]["frozen_B_IF_policy"]["budgets"], strict=True):
                for key in ["K", "TP", "FP", "FN", "TN", "precision_at_k", "recall_at_k",
                            "fraud_amount_detected", "amount_coverage", "amount_groups"]:
                    self.assertEqual(base[key], policy[key])
            random = first["metrics"]["random_expectation"]
            self.assertTrue(random["expected_only"])
            self.assertIsNone(random["AP"])
            for row in random["budgets"]:
                self.assertAlmostEqual(row["TP"], row["K"]*10/200)
                self.assertAlmostEqual(row["amount_coverage"], row["K"]/200)
                self.assertEqual(row["precision_lift"], 1)
            public = root/"shared"
            self.assertEqual(len(list((public/"figures").glob("*.png"))), 3)
            self.assertEqual(len(list((public/"tables").glob("*.json"))), 1)
            for path in (public/"figures").glob("*.png"):
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))
            self.assertFalse(list(public.rglob("*predictions*")))
            self.assertNotIn('"combined_ids"', (public/"tables/c_extensions.json").read_text())
            with patch("unsupervised.score_improved_bundle", side_effect=AssertionError("preflight")):
                with self.assertRaisesRegex(ValueError, "already exists"):
                    run_c_extensions(self.bundle, self.val, self.config, root/"blocked", results=public)
                self.assertFalse((root/"blocked").exists())
        self.assertEqual(joblib.hash(self.bundle), before)

    def test_extension_phase_and_frozen_config_checked_before_scoring(self):
        from evaluation import run_c_extensions, build_c_extension_config, partition_description
        with tempfile.TemporaryDirectory() as folder, patch(
            "unsupervised.score_improved_bundle", side_effect=AssertionError("must not score")
        ):
            with self.assertRaises(ValueError):
                run_c_extensions(self.bundle, self.val.assign(row_id=self.val.row_id+10000),
                                 self.config, folder)
            changed = copy.deepcopy(self.bundle)
            changed["config"]["B_combination"]["share"] = 0.5
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                run_c_extensions(changed, self.val, self.config, folder)
            with self.assertRaisesRegex(ValueError, "overlap"):
                run_c_extensions(self.bundle, self.val, self.config, folder, results=Path(folder)/"shared")
            cfg = copy.deepcopy(self.config)
            cfg["partitions"]["test"] = partition_description(self.val)
            with self.assertRaisesRegex(ValueError, "G2 reference"):
                run_c_extensions(self.bundle, self.val, cfg, folder, phase="test")
        with self.assertRaisesRegex(ValueError, "Training identities"):
            build_c_extension_config(self.train.assign(row_id=self.train.row_id+10000), self.bundle)

    def test_extension_evidence_ignores_mapping_and_prediction_table_order(self):
        from evaluation import evaluate_c_extensions
        from classification import score_model
        from unsupervised import score_improved_bundle
        predictions, _ = score_improved_bundle(self.bundle, self.val)
        predictions["B"] = make_prediction_frame(
            self.val, score_model(self.bundle["B_model"], self.val[FEATURE_COLUMNS]), "B")
        first = evaluate_c_extensions(predictions, self.val, self.config, progress=None)
        reversed_input = {name: predictions[name].iloc[::-1] for name in reversed(predictions)}
        second = evaluate_c_extensions(reversed_input, self.val, self.config, progress=None)
        self.assertEqual(first, second)


class CTeamIntegrationTests(unittest.TestCase):
    setUpClass = classmethod(CExtensionWorkflowTests.setUpClass.__func__)
    def test_precomputed_scores_align_to_source_ids_without_rescoring(self):
        from evaluation import evaluate_improved_frozen
        from classification import score_model
        from unsupervised import score_improved_bundle
        bundle = copy.deepcopy(self.bundle)
        bundle["config"].update(distance_share=0.0, n_clusters=3)
        predictions, groups = score_improved_bundle(bundle, self.val)
        classifier = make_prediction_frame(self.val, score_model(bundle["B_model"], self.val[FEATURE_COLUMNS]), "B")
        before = joblib.hash((predictions, groups, bundle))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            normal = evaluate_improved_frozen(bundle, self.val, root/"normal", classifier=classifier,
                                             phase="validation_replay")
            keyed_groups = pd.Series(groups, index=self.val.row_id).iloc[::-1]
            reversed_frames = {n: p.iloc[::-1] for n,p in predictions.items()}
            with patch("unsupervised.score_improved_bundle", side_effect=AssertionError("no rescore")), \
                 patch("classification.score_model", side_effect=AssertionError("no classifier rescore")):
                cached = evaluate_improved_frozen(
                    bundle, self.val, root/"cached", classifier=classifier.iloc[::-1],
                    phase="validation_replay", scored_unsupervised=(reversed_frames, keyed_groups))
            for key in ["metrics", "groups", "time_windows", "comparisons", "B_comparisons"]:
                self.assertEqual(normal[key], cached[key])
        self.assertEqual(joblib.hash((predictions, groups, bundle)), before)

    def test_invalid_precomputed_metadata_or_groups_fail_before_output(self):
        from classification import score_model
        from unsupervised import score_improved_bundle
        bundle = copy.deepcopy(self.bundle)
        bundle["config"].update(distance_share=0.0, n_clusters=3)
        predictions, groups = score_improved_bundle(bundle, self.val)
        classifier = make_prediction_frame(self.val, score_model(bundle["B_model"], self.val[FEATURE_COLUMNS]), "B")
        first = next(iter(predictions))
        invalid = []
        for field,value in [("Class",0),("Time",-1),("row_id",-1),("method","wrong")]:
            changed = dict(predictions);changed[first]=predictions[first].assign(**{field:value})
            invalid.append((changed,groups))
        invalid += [(dict(list(predictions.items())[:1]),groups),
                    (predictions,groups[:-1]),(predictions,np.full(len(groups),0.5)),
                    (predictions,np.full(len(groups),9)),
                    (predictions,pd.Series(groups,index=np.arange(len(groups))))]
        with tempfile.TemporaryDirectory() as folder, patch(
            "unsupervised.score_improved_bundle",side_effect=AssertionError("no scoring")):
            for n,scored in enumerate(invalid):
                output=Path(folder)/str(n)
                with self.subTest(case=n), self.assertRaises(ValueError):
                    evaluate_improved_frozen(bundle,self.val,output,classifier=classifier,
                                             phase="validation_replay",scored_unsupervised=scored)
                self.assertFalse(output.exists())


class CReviewPolicyFigureTests(unittest.TestCase):
    def test_plot_reuses_A_counts_and_refuses_invalid_or_existing_evidence(self):
        from temporal import select_review_threshold,evaluate_review_threshold
        from evaluation import save_review_policy_figure
        data=pd.DataFrame({"row_id":[0,1,2,3],"Time":[0,10,30,31],"Class":[1,0,1,0]})
        results={}
        for name,scores in [("B",[.9,.1,.6,.2]),("IF",[.1,.2,.7,.3])]:
            p=make_prediction_frame(data,scores,name)
            selected=select_review_threshold(p,.5)
            results[name]={**selected,**evaluate_review_threshold(p,selected,window_seconds=10)}
        before=copy.deepcopy(results)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"policy.png"
            self.assertEqual(save_review_policy_figure(results,output,phase="validation"),output)
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))
            with self.assertRaisesRegex(ValueError,"already exists"):
                save_review_policy_figure(results,output,phase="validation")
            wrong=copy.deepcopy(results);wrong["IF"]["windows"][0]["n_samples"]+=1
            with self.assertRaisesRegex(ValueError,"identical windows"):
                save_review_policy_figure(wrong,Path(folder)/"bad.png",phase="validation")
            self.assertFalse((Path(folder)/"bad.png").exists())
            with self.assertRaises(ValueError):
                save_review_policy_figure(results,Path(folder)/"bad.png",phase="unknown")
        self.assertEqual(results,before)

if __name__ == "__main__":
    unittest.main()
