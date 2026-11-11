"""负责人 C：有界改进、内层边界、训练标定、保存恢复及冻结评价检查。"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from c_improvements import (
    TransactionFeatures,
    configurations,
    candidate_order,
    inner_time_split,
    fit_configuration,
    score_configuration,
    select_families,
    select_pair,
    evaluate_improved_frozen,
    SHARES,
)
from run_c_improvements import run_experiment
from run_c_validation import _demo
from preprocessing import FEATURE_COLUMNS


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
            with patch(
                "c_improvements.select_pair",
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
                    "c_improvements.select_families",
                    side_effect=AssertionError("no reselection"),
                ),
                patch(
                    "c_improvements.select_pair",
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
            with patch("run_c_improvements.fit_configuration") as fit:
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
