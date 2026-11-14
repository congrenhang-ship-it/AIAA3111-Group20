"""Regression checks for data reproducibility and the Task A handoff."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import download_data as downloader
from main import REVIEW_FRACTIONS, run_validation, split_summary
from preprocessing import DEFAULT_DATA_PATH, FEATURE_COLUMNS, ULB_SHA256, audit_data, data_sha256
from preprocessing import get_features, load_data, split_data
from temporal import evaluate_time_windows, summarize_time_windows
from temporal import evaluate_review_threshold, select_review_threshold


class DownloadTests(unittest.TestCase):
    def test_download_reuse_and_corruption(self):
        class Response(BytesIO):
            status = 200

        content = b"valid CSV bytes"
        digest = sha256(content).hexdigest()
        with TemporaryDirectory() as directory, patch.multiple(
            downloader, ULB_SHA256=digest, ULB_SIZE_BYTES=len(content),
        ):
            target = Path(directory) / "data" / "creditcard.csv"
            with patch.object(downloader, "urlopen", return_value=Response(content)) as request:
                self.assertTrue(downloader.download_data(target))
                self.assertFalse(downloader.download_data(target))
                self.assertEqual(request.call_count, 1)
            self.assertEqual(target.read_bytes(), content)

            wrong = b"x" * len(content)
            target.write_bytes(wrong)
            with patch.object(downloader, "urlopen") as request:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    downloader.download_data(target)
                request.assert_not_called()
            self.assertEqual(target.read_bytes(), wrong)

            other = target.with_name("bad.csv")
            for payload, message in ((wrong, "SHA-256"), (b"short", "bytes")):
                with patch.object(downloader, "urlopen", return_value=Response(payload)):
                    with self.assertRaisesRegex(ValueError, message):
                        downloader.download_data(other)
                self.assertFalse(other.exists())
                self.assertEqual(list(target.parent.iterdir()), [target])


class SplitTests(unittest.TestCase):
    def test_stable_identity_and_boundary_ties(self):
        times = [30, 0, 10, 10, 10, 20, 30, 40, 50, 60]
        data = pd.DataFrame(0.0, index=range(len(times)), columns=FEATURE_COLUMNS)
        data["Time"] = times
        data["Amount"] = range(len(times))
        data["Class"] = [0, 0, 1, 0, 0, 0, 0, 1, 0, 1]
        data["row_id"] = range(len(times))
        duplicate = data.iloc[[2]].assign(row_id=10)
        shuffled = pd.concat([duplicate, data]).sample(frac=1, random_state=42)
        train, validation, test = split_data(shuffled, (0.3, 0.4, 0.3))
        self.assertEqual(train.row_id.tolist(), [1])
        self.assertEqual(validation.row_id.tolist(), [2, 3, 4, 5, 0, 6])
        self.assertEqual(test.row_id.tolist(), [7, 8, 9])
        self.assertEqual(list(get_features(train)), FEATURE_COLUMNS)
        for bad in (data.assign(Time=0), data.assign(Amount=float("nan")), data.assign(row_id=0)):
            with self.assertRaises(ValueError):
                split_data(bad)

    @unittest.skipUnless(DEFAULT_DATA_PATH.is_file(), "Run code/download_data.py for full-data checks")
    def test_pinned_ulb_audit_and_ordered_split_fingerprints(self):
        self.assertEqual(data_sha256(DEFAULT_DATA_PATH), ULB_SHA256)
        data = load_data(DEFAULT_DATA_PATH)
        audit = audit_data(data)
        self.assertEqual(audit["rows"], 284_807)
        self.assertEqual(audit["class_counts"], {"0": 284_315, "1": 492})
        self.assertEqual(audit["duplicate_class_counts"], {"0": 1_062, "1": 19})
        self.assertEqual(sum(audit["missing_values"].values()), 0)
        summary = split_summary(split_data(data))
        self.assertEqual(
            [(row["rows"], row["frauds"], row["time_min"], row["time_max"])
             for row in summary.values()],
            [(170_235, 342, 0, 120_392), (56_745, 57, 120_393, 145_233),
             (56_746, 74, 145_234, 172_792)],
        )
        self.assertEqual([row["row_ids_sha256"] for row in summary.values()], [
            "1f69bf9bb0e7e5439a0896426eae142926d5e77b7d759a4e64bfa39c0a946333",
            "2b3d3d2dd89b8a07a411d51d5d2afdf04b44922551bb7e6aad90aaef7a876de1",
            "4909103f5579d282c4faa1ca32c3c57fdf328e5d2ade9533a297d3e0254ca5c9",
        ])


class HandoffTests(unittest.TestCase):
    def test_fixed_review_threshold_keeps_boundary_ties_and_empty_windows(self):
        predictions = pd.DataFrame({
            "row_id": [0, 1, 2, 3, 4],
            "Class": [1, 0, 1, 0, 0],
            "Time": [0, 1, 2, 30, 40],
            "method": ["classifier"] * 5,
            "score": [0.9, 0.8, 0.8, 0.1, 0.85],
        })
        selection = select_review_threshold(predictions, 0.4)
        self.assertEqual((selection["target_K"], selection["threshold"]), (2, 0.85))
        selection = select_review_threshold(predictions, 0.6)
        self.assertEqual((selection["target_K"], selection["equal_to_threshold"]), (3, 2))
        result = evaluate_review_threshold(predictions, selection, window_seconds=10)
        self.assertEqual(result["overall"]["fixed_threshold"]["K"], 4)
        self.assertEqual(result["overall"]["top_k"]["K"], 3)
        windows = result["windows"]
        self.assertEqual([w["status"] for w in windows], [
            "ok", "empty", "empty", "no_positives", "no_positives",
        ])
        self.assertEqual([w["fixed_threshold"]["K"] for w in windows], [3, 0, 0, 0, 1])
        self.assertIsNone(windows[1]["fixed_threshold"]["actual_review_fraction"])
        self.assertEqual(windows[3]["fixed_threshold"]["actual_review_fraction"], 0)
        self.assertIsNone(windows[4]["fixed_threshold"]["recall_at_k"])
        self.assertEqual(windows[4]["fixed_threshold"]["FP"], 1)
        self.assertTrue(all(w["fixed_threshold"]["value"] == 0.8 for w in windows))

    def test_windows_align_and_delegate_to_c(self):
        first = pd.DataFrame({
            "row_id": [8, 3, 2], "Class": [1, 0, 1], "Time": [30, 10, 0],
            "method": ["a"] * 3, "score": [0.9, 0.1, 0.7],
        })
        predictions = pd.concat([first, first.iloc[::-1].assign(method="b")])
        with patch("evaluation.evaluate_scores", return_value={"ap": None}) as evaluator:
            windows = evaluate_time_windows(predictions, 10, 0, (0.1,), 0.5)
        self.assertEqual([row["status"] for row in windows], ["ok", "no_positives", "empty", "ok"] * 2)
        self.assertEqual([row["n_positives"] for row in windows], [1, 0, 0, 1] * 2)
        self.assertIsNone(windows[2]["evaluation"])
        self.assertEqual(evaluator.call_count, 6)
        labels, scores, fractions = evaluator.call_args_list[1].args
        self.assertEqual((labels.tolist(), scores.tolist(), fractions), ([0], [0.1], (0.1,)))
        self.assertEqual(evaluator.call_args_list[1].kwargs["row_ids"].tolist(), [3])
        self.assertEqual(evaluator.call_args_list[1].kwargs["threshold"], 0.5)
        with self.assertRaisesRegex(ValueError, "same row_id"):
            summarize_time_windows(predictions.iloc[:-1], 10)

    def test_g1_routes_training_validation_and_reports_dependencies(self):
        train = pd.DataFrame(0.0, index=range(2), columns=FEATURE_COLUMNS)
        train = train.assign(Class=[0, 1], row_id=[1, 2])
        validation = train.assign(Time=[3600, 3601], row_id=[3, 4])
        frame = validation[["row_id", "Class", "Time"]].assign(
            method="logistic_regression", score=[0.3, 0.8],
        )
        with patch("classification.train_model", side_effect=NotImplementedError):
            with self.assertRaisesRegex(RuntimeError, "B2"):
                run_validation(train, validation)
        with (
            patch("classification.train_model", return_value=object()) as fit,
            patch("classification.score_model", return_value=[0.3, 0.8]) as score,
            patch("evaluation.make_prediction_frame", return_value=frame) as assemble,
            patch("evaluation.evaluate_scores", return_value={}) as evaluate,
        ):
            result = run_validation(train, validation)
            self.assertEqual(result["split"], "validation")
            self.assertTrue(fit.call_args.args[0].equals(get_features(train)))
            self.assertEqual(fit.call_args.kwargs, {"strategy": "original", "seed": 42})
            self.assertTrue(score.call_args.args[1].equals(get_features(validation)))
            self.assertIs(assemble.call_args.args[0], validation)
            self.assertEqual(evaluate.call_args_list[0].args[2], REVIEW_FRACTIONS)
            self.assertEqual(evaluate.call_args_list[0].kwargs["row_ids"].tolist(), [3, 4])
            assemble.side_effect = NotImplementedError
            with self.assertRaisesRegex(RuntimeError, "C2"):
                run_validation(train, validation)
            assemble.side_effect = None
            assemble.return_value = frame.iloc[::-1]
            with self.assertRaisesRegex(ValueError, "preserve validation rows"):
                run_validation(train, validation)
