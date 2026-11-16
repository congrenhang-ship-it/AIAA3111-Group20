"""D1 synthetic checks, not ULB experimental evidence. Standard-library runner."""
import copy
import itertools
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from pattern_mining import mine_rules, validate_rules, analyze_rule_overlap


def training_data():
    # 120 rows, 30 fraud. Every one of the four cells has 30 rows.
    return pd.DataFrame({
        "row_id": np.arange(120), "V1": np.repeat([0., 0., 1., 1.], 30),
        "V2": np.repeat([0., 1., 0., 1.], 30),
        "Class": [1] * 24 + [0] * 6 + [1] * 6 + [0] * 84,
    })


class PatternMiningChecks(unittest.TestCase):
    def test_hand_calculated_metrics_and_ids(self):
        train = training_data()
        binner, _ = mine_rules(train, ["V1"], bins=2)
        # 12 transactions, antecedent covers six, four fraud in total, all covered.
        sample = pd.DataFrame({"row_id": np.arange(200, 212),
                               "V1": [0.] * 6 + [1.] * 6,
                               "Class": [1] * 4 + [0] * 8})
        rule = {"rule_id": "hand", "antecedent": [{"feature": "V1", "bin": 0}],
                "consequent": {"Class": 1}}
        result = validate_rules(sample, binner, [rule])[0]
        for key, value in {"support": 1/3, "confidence": 2/3, "lift": 2.,
                           "kulczynski": 5/6, "fraud_recall": 1.}.items():
            self.assertAlmostEqual(result[key], value)
        self.assertEqual(result["coverage_count"], 6)
        self.assertEqual(result["fraud_count"], 4)
        self.assertEqual(result["normal_count"], 2)
        self.assertEqual(result["covered_row_ids"], list(range(200, 206)))

    def test_apriori_matches_exhaustive_short_itemsets(self):
        # Independent exhaustive oracle checks candidate completeness and counts.
        data = training_data()
        for strategy in ["equal_width", "equal_frequency"]:
            with self.subTest(strategy=strategy):
                binner, rules = mine_rules(data, ["V1", "V2"], bins=2, binning=strategy)
                expected = {}
                items = [(f, b) for f in ["V1", "V2"] for b in [0, 1]]
                for size in [1, 2]:
                    for antecedent in itertools.combinations(items, size):
                        if len({f for f, _ in antecedent}) != size:
                            continue
                        mask = np.ones(len(data), dtype=bool)
                        for feature, value in antecedent:
                            mask &= data[feature].to_numpy() == value
                        count, fraud = int(mask.sum()), int(data.loc[mask, "Class"].sum())
                        if count >= 20 and fraud >= 5 and fraud/count > .25:
                            expected[antecedent] = (count, fraud)
                actual = {tuple((x["feature"], x["bin"]) for x in rule["antecedent"]):
                          (rule["train"]["coverage_count"], rule["train"]["fraud_count"])
                          for rule in rules}
                self.assertEqual(actual, expected)
                self.assertTrue(any(len(a) == 2 for a in actual))
                self.assertEqual(json.loads(json.dumps(binner, allow_nan=False)), binner)
                json.dumps(rules, allow_nan=False)

    def test_validation_never_refits_and_preserves_order(self):
        train = training_data()
        train_before = train.copy(deep=True)
        binner, rules = mine_rules(train, ["V1", "V2"], bins=2)
        before = copy.deepcopy((binner, rules))
        held_out = pd.DataFrame({"row_id": [999, 500, 700], "Class": [1, 0, 1],
                                 "V1": [-100., .5, 100.], "V2": [0., 0., 1.]})
        results = validate_rules(held_out, binner, rules)
        low = next(r for r in results if r["antecedent"] == [{"feature": "V1", "bin": 0}])
        self.assertEqual(low["covered_row_ids"], [999])  # .5 goes right
        self.assertEqual((binner, rules), before)
        pd.testing.assert_frame_equal(train, train_before)
        # Labels affect evaluation, not training bin boundaries.
        changed = train.assign(Class=1 - train.Class)
        other_binner, _ = mine_rules(changed, ["V1", "V2"], bins=2)
        self.assertEqual(other_binner, binner)

    def test_zero_denominators_and_empty_holdout(self):
        train = training_data()
        binner, rules = mine_rules(train, ["V1"], bins=2)
        for data in [train.assign(Class=0), train.iloc[:0]]:
            for result in validate_rules(data, binner, rules):
                for metric in ["lift", "fraud_recall", "kulczynski"]:
                    self.assertIsNone(result[metric])
                json.dumps(result, allow_nan=False)
        no_match = train.assign(V1=100.)
        result = validate_rules(no_match, binner, rules)[0]
        self.assertEqual(result["coverage_count"], 0)
        self.assertEqual(result["support"], 0.)
        self.assertIsNone(result["confidence"])

    def test_constant_features_and_rare_rules(self):
        train = training_data().assign(V1=7.)
        for strategy in ["equal_width", "equal_frequency"]:
            binner, rules = mine_rules(train, ["V1"], bins=3, binning=strategy)
            self.assertEqual(binner["features"]["V1"]["cut_points"], [])
            self.assertEqual(rules, [])  # whole-dataset antecedent has lift 1
        binner, rules = mine_rules(training_data().iloc[:19], ["V1"], bins=2)
        self.assertEqual(rules, [])  # minimum 20 covered transactions
        binner, rules = mine_rules(training_data().assign(Class=[1]*4 + [0]*116), ["V1"])
        self.assertEqual(rules, [])  # minimum 5 covered frauds

    def test_bad_inputs_fail_explicitly(self):
        train = training_data()
        for features in [["Class"], ["row_id"], ["V1", "V1"]]:
            with self.assertRaises(ValueError):
                mine_rules(train, features)
        for bad in [train.assign(V1=np.nan), train.assign(V1=np.inf),
                    train.assign(row_id=1), train.assign(Class=2), train.iloc[:0]]:
            with self.assertRaises(ValueError):
                mine_rules(bad, ["V1"])

    def test_distinct_binning_schemes_and_repeated_quantiles(self):
        data = training_data().assign(V1=np.arange(120, dtype=float) ** 2)
        width, _ = mine_rules(data, ["V1"], bins=3, binning="equal_width")
        frequency, rules = mine_rules(data, ["V1"], bins=3)
        np.testing.assert_allclose(width["features"]["V1"]["cut_points"],
                                   [14161/3, 2*14161/3])
        np.testing.assert_allclose(frequency["features"]["V1"]["cut_points"],
                                   [1573 + 2/3, 6294.])
        results = validate_rules(data, json.loads(json.dumps(frequency)), rules)
        for rule, result in zip(rules, results):
            self.assertEqual(rule["train"], {key: result[key] for key in rule["train"]})
        repeated = training_data().assign(V1=[0.] * 100 + [1.] * 20)
        binner, _ = mine_rules(repeated, ["V1"], bins=3)
        self.assertEqual(binner["features"]["V1"]["effective_bins"], 1)


class RuleOverlapChecks(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"row_id": [0, 1, 2, 3], "Class": [1, 0, 1, 0],
                                  "V1": [0., 0., 1., 1.], "V2": [0., 1., 0., 1.]})
        self.binner, _ = mine_rules(self.data, ["V1", "V2"], bins=2)
        self.rules = [{"rule_id": name, "antecedent": items, "consequent": {"Class": 1}}
                      for name, items in [("A", [{"feature": "V1", "bin": 0}]),
                                          ("B", [{"feature": "V2", "bin": 0}]),
                                          ("duplicate", [{"feature": "V1", "bin": 0}])]]
        self.metrics = {name: {"lift": lift, "confidence": .5}
                        for name, lift in [("A", 3), ("B", 2), ("duplicate", 1)]}

    def test_hand_calculated_union_intersection_and_increments(self):
        out = analyze_rule_overlap(self.data, self.binner, self.rules, self.metrics)
        self.assertAlmostEqual(out["jaccard"][0][1], 1/3)
        self.assertEqual(out["jaccard"][0][2], 1.)
        self.assertEqual([r["transactions"] for r in out["cumulative"]], [2, 3, 3])
        self.assertEqual([r["frauds"] for r in out["cumulative"]], [1, 2, 2])
        self.assertEqual([r["increment_frauds"] for r in out["cumulative"]], [1, 1, 0])
        self.assertTrue(out["cumulative"][-1]["redundant_given_previous"])
        self.assertEqual(out["display_rule_ids"], ["A", "B"])

    def test_empty_and_no_positive_coverage(self):
        out = analyze_rule_overlap(self.data.iloc[:0], self.binner, self.rules, self.metrics)
        self.assertTrue(all(v is None for row in out["jaccard"] for v in row))
        self.assertEqual(out["covered_frauds"], 0)
        other = analyze_rule_overlap(self.data.assign(Class=0), self.binner, self.rules, self.metrics)
        self.assertEqual(other["display_rule_ids"], [])
        self.assertEqual(other["covered_transactions"], 3)

    def test_order_is_training_fixed_not_input_or_labels(self):
        original = analyze_rule_overlap(self.data, self.binner, self.rules, self.metrics)
        changed = analyze_rule_overlap(self.data.assign(Class=1), self.binner, self.rules[::-1], self.metrics)
        self.assertEqual(original["rule_order"], changed["rule_order"])
        self.assertEqual(original["jaccard"], changed["jaccard"])
        self.assertEqual(original["rule_order"], ["A", "B", "duplicate"])


if __name__ == "__main__":
    unittest.main()
