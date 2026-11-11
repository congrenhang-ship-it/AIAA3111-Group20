"""B1 synthetic interface checks; these are not real-data experiment results.

Run from the repository root:
    py -3.12 -m unittest discover -s code -p test_classification.py -v
"""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from classification import MODEL_NAMES, score_model, train_model


class ClassificationInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        features, labels = make_classification(
            n_samples=160, n_features=5, n_informative=3,
            weights=[0.8, 0.2], random_state=42,
        )
        data = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 6)])
        data["Class"] = labels
        data["row_id"] = np.arange(len(data))
        cls.train, cls.validation = data.iloc[:120].copy(), data.iloc[120:].copy()
        cls.labels = labels[:120]

    def test_all_baselines_score_unseen_rows_and_preserve_order(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = train_model(self.train, self.labels, name)
                scores = score_model(model, self.validation)
                self.assertEqual(scores.shape, (40,))
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(((scores >= 0) & (scores <= 1)).all())
                np.testing.assert_allclose(
                    score_model(model, self.validation.iloc[::-1]), scores[::-1]
                )

                changed = self.validation.copy()
                changed["Class"] = 1 - changed["Class"]
                changed["row_id"] = -changed["row_id"]
                changed = changed.loc[:, changed.columns[::-1]]
                np.testing.assert_allclose(score_model(model, changed), scores)
                np.testing.assert_allclose(
                    score_model(model, self.validation.drop(columns=["Class", "row_id"])),
                    scores,
                )
                # Metadata must also be excluded during training.
                clean_model = train_model(
                    self.train.drop(columns=["Class", "row_id"]), self.labels, name
                )
                np.testing.assert_allclose(score_model(clean_model, self.validation), scores)

    def test_scaler_uses_only_training_rows_and_is_not_refit_during_scoring(self):
        for name in ("logistic_regression", "gaussian_nb"):
            with self.subTest(model=name):
                model = train_model(self.train, self.labels, name)
                scaler = model.named_steps["preprocessing"].named_transformers_["features"]
                expected = self.train.drop(columns=["Class", "row_id"]).mean().to_numpy()
                np.testing.assert_allclose(scaler.mean_, expected)
                self.assertEqual(scaler.n_samples_seen_, len(self.train))
                shifted = self.validation.copy()
                shifted.loc[:, "V1":"V5"] += 1000
                score_model(model, shifted)
                np.testing.assert_allclose(scaler.mean_, expected)
                self.assertEqual(scaler.n_samples_seen_, len(self.train))

    def test_numeric_arrays_are_supported(self):
        train = self.train.drop(columns=["Class", "row_id"]).to_numpy()
        validation = self.validation.drop(columns=["Class", "row_id"]).to_numpy()
        model = train_model(train, self.labels, "logistic_regression")
        self.assertEqual(score_model(model, validation).shape, (40,))

    def test_imbalance_strategies_are_training_only_and_reproducible(self):
        model = train_model(self.train, self.labels, "logistic_regression", "class_weight")
        self.assertEqual(model.named_steps["classifier"].class_weight, "balanced")
        self.assertEqual(len(model.training_positions_), len(self.train))
        first = train_model(self.train, self.labels, "logistic_regression", "random_undersample")
        second = train_model(self.train, self.labels, "logistic_regression", "random_undersample")
        np.testing.assert_array_equal(first.training_positions_, second.training_positions_)
        self.assertEqual(first.training_class_counts_[0], first.training_class_counts_[1])
        self.assertEqual(len(np.unique(first.training_positions_)), len(first.training_positions_))
        self.assertTrue(set(np.flatnonzero(self.labels == 1)) <= set(first.training_positions_))
        expected = self.train.iloc[first.training_positions_].drop(columns=["Class", "row_id"]).mean()
        scaler = first.named_steps["preprocessing"].named_transformers_["features"]
        np.testing.assert_allclose(scaler.mean_, expected.to_numpy())
        np.testing.assert_allclose(score_model(first, self.validation), score_model(second, self.validation))
        self.assertEqual(score_model(first, self.validation).shape, (len(self.validation),))

    def test_selection_uses_primary_budget_then_ap(self):
        from run_b_validation import select_method
        def record(name, tp, ap):
            return dict(method=name, metrics=dict(AP=ap, budgets=[dict(review_fraction=0.01, TP=tp)]))
        records = [record("a", 2, 0.99), record("b", 3, 0.2), record("c", 3, 0.3)]
        self.assertEqual(select_method(records)["method"], "c")
        self.assertEqual(select_method([record("b", 3, 0.3), record("a", 3, 0.3)])["method"], "a")

    def test_positive_class_column_is_located_by_label(self):
        model = SimpleNamespace(
            classes_=np.array([1, 0]),
            predict_proba=lambda X: np.array([[0.8, 0.2], [0.3, 0.7]]),
        )
        np.testing.assert_array_equal(score_model(model, np.zeros((2, 1))), [0.8, 0.3])

    def test_invalid_scores_and_missing_positive_class_are_rejected(self):
        for probabilities in (np.array([[np.nan, 0.2]]), np.array([[0.1, 1.2]]), np.ones((1, 3))):
            with self.subTest(probabilities=probabilities):
                model = SimpleNamespace(classes_=np.array([0, 1]), predict_proba=lambda X: probabilities)
                with self.assertRaises(ValueError):
                    score_model(model, np.zeros((1, 1)))
        with self.assertRaises(ValueError):
            score_model(SimpleNamespace(classes_=np.array([0])), np.zeros((1, 1)))

    def test_invalid_training_inputs_are_rejected(self):
        cases = [
            (self.train, self.labels, "unknown", "original"),
            (self.train, self.labels, "gaussian_nb", "class_weight"),
            (self.train, np.zeros(len(self.train)), "logistic_regression", "original"),
            (self.train, self.labels.reshape(-1, 1), "logistic_regression", "original"),
            (self.train, np.full(len(self.train), 2), "logistic_regression", "original"),
            (self.train, self.labels[:-1], "logistic_regression", "original"),
            (self.train[["Class", "row_id"]], self.labels, "logistic_regression", "original"),
        ]
        for X, y, name, strategy in cases:
            with self.subTest(name=name, strategy=strategy, shape=np.shape(y)):
                with self.assertRaises(ValueError):
                    train_model(X, y, name, strategy)


if __name__ == "__main__":
    unittest.main()
