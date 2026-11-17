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
from workflows import download_data as downloader
from main import REVIEW_FRACTIONS, run_validation, split_summary
from preprocessing import DEFAULT_DATA_PATH, FEATURE_COLUMNS, ULB_SHA256, audit_data, data_sha256
from preprocessing import get_features, load_data, split_data
from temporal import DAY_SECONDS, evaluate_time_windows, handbook_history_builder, summarize_time_windows
from temporal import evaluate_review_threshold, select_review_threshold


class BExtensionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from evaluation import partition_description
        rng=np.random.default_rng(42)
        def fixture(rows,start):
            frame=pd.DataFrame(rng.normal(size=(rows,len(FEATURE_COLUMNS))),columns=FEATURE_COLUMNS,
                               index=range(start,start+rows))
            frame["Amount"]=rng.uniform(1,100,rows)
            frame["Time"]=range(start,start+rows)
            return frame.assign(row_id=range(start,start+rows),Class=(np.arange(rows)%11==0).astype(int))
        cls.train,cls.validation,cls.test=fixture(320,0),fixture(96,1000),fixture(64,2000)
        cls.partitions={n:partition_description(f) for n,f in
                        (("train",cls.train),("validation",cls.validation),("test",cls.test))}

    def test_standalone_delegates_the_verified_split_to_the_same_implementation(self):
        import validation.run_b_validation as b
        marker=object()
        with (patch.object(b,"data_sha256",return_value=ULB_SHA256),
              patch.object(b,"load_data",return_value="loaded"),
              patch.object(b,"split_data",return_value=(self.train,self.validation,self.test)),
              patch.object(b,"run_ulb_extensions_on_split",return_value=marker) as shared):
            self.assertIs(b.run_ulb_extensions("source.csv",Path("output")),marker)
            args=shared.call_args.args
            self.assertIs(args[0],self.train);self.assertIs(args[1],self.validation)
            self.assertEqual(args[2],self.partitions)
            self.assertEqual(shared.call_args.kwargs["results"],b.ROOT/"results")

    def test_actual_shared_split_keeps_models_and_cases_private_and_all_conditions(self):
        import json
        import validation.run_b_validation as b
        from sklearn.inspection import permutation_importance
        before=data_sha256(b.ROOT/"results/tables/b_extensions.json")
        with TemporaryDirectory(dir=b.ROOT/"data") as folder:
            private,public=Path(folder)/"private",Path(folder)/"aggregate"
            with (patch.object(b,"load_data",side_effect=AssertionError("do not reload test data")),
                  patch.object(b,"split_data",side_effect=AssertionError("do not repartition")),
                  patch.object(b,"train_model",wraps=b.train_model) as fit,
                  patch("sklearn.inspection.permutation_importance",wraps=permutation_importance) as importance):
                result=b.run_ulb_extensions_on_split(self.train,self.validation,self.partitions,
                                                     private,results=public)
            self.assertEqual(fit.call_count,13)  # 15 conditions, identical full subsets reused.
            for call in fit.call_args_list:
                self.assertTrue(call.args[0].index.isin(self.train.index).all())
            self.assertTrue(importance.call_args.args[1].equals(get_features(self.validation)))
            self.assertEqual(importance.call_args.kwargs["n_repeats"],5)
            self.assertEqual(importance.call_args.kwargs["scoring"],"average_precision")
            self.assertEqual(len(result["E03"]["conditions"]),15)
            self.assertEqual(len(result["E05"]["importances"]),30)
            self.assertTrue(all(len(r["repeats"])==5 for r in result["E05"]["importances"]))
            self.assertEqual(len(list(private.glob("ulb_mask_*.npy"))),15)
            self.assertTrue((private/"ulb_error_cases.csv").is_file())
            self.assertFalse(list(public.rglob("*.csv")))
            self.assertFalse(list(public.rglob("*.joblib")))
            summary=json.loads((public/"tables/b_extensions.json").read_text())
            self.assertEqual(summary["partitions"],self.partitions)
            self.assertTrue((public/"figures/b_extensions.png").is_file())
            self.assertFalse(summary["metadata"]["test_scored"])
        self.assertEqual(data_sha256(b.ROOT/"results/tables/b_extensions.json"),before)

    def test_wrong_partition_identity_stops_before_training(self):
        import copy
        import validation.run_b_validation as b
        wrong=copy.deepcopy(self.partitions);wrong["validation"]["frauds"]+=1
        with TemporaryDirectory(dir=b.ROOT/"data") as folder,patch.object(b,"train_model") as fit:
            with self.assertRaisesRegex(ValueError,"different validation split"):
                b.run_ulb_extensions_on_split(self.train,self.validation,wrong,
                                             Path(folder)/"private",results=Path(folder)/"aggregate")
            fit.assert_not_called()
            self.assertFalse((Path(folder)/"private").exists())


class HandbookHistoryTests(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame({
            "row_id": [90, 12, 14, 18, 20, 33],
            "Time": [0.0, 100.0, 100.0, float(DAY_SECONDS), float(7 * DAY_SECONDS), float(7 * DAY_SECONDS + 1)],
            "CUSTOMER_ID": [1, 1, 1, 1, 1, 1],
            "TERMINAL_ID": [2, 2, 3, 3, 2, 2],
            "Amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        })

    def test_hand_calculated_boundaries_and_same_time_isolation(self):
        both = handbook_history_builder(self.events)["both_history"]
        self.assertEqual(both.loc[12, "customer_count_1d"], 1)
        self.assertEqual(both.loc[14, "customer_mean_amount_1d"], 10)
        self.assertEqual(both.loc[14, "terminal_count_1d"], 0)
        self.assertEqual(both.loc[18, "customer_count_1d"], 3)
        self.assertEqual(both.loc[18, "customer_mean_amount_1d"], 20)
        self.assertEqual(both.loc[18, "customer_amount_ratio_1d"], 2)
        self.assertEqual(both.loc[20, "customer_count_7d"], 4)
        self.assertEqual(both.loc[33, "customer_count_7d"], 4)

    def test_mask_and_future_do_not_delete_or_change_current_features(self):
        clean = handbook_history_builder(self.events)
        keep = pd.Series([False, True, True, True, True, True], index=self.events.row_id)
        masked = handbook_history_builder(self.events.sample(frac=1, random_state=42), history_keep=keep)
        pd.testing.assert_frame_equal(clean["transaction_only"].sort_index(), masked["transaction_only"].sort_index())
        self.assertEqual(masked["both_history"].loc[14, "customer_count_1d"], 0)
        self.assertEqual(masked["both_history"].loc[14, "customer_amount_ratio_unavailable_1d"], 1)
        future = self.events.copy()
        future.loc[5, "Amount"] = 1000000
        after = handbook_history_builder(future)
        for name in clean:
            pd.testing.assert_frame_equal(clean[name].iloc[:-1], after[name].iloc[:-1])
        with self.assertRaisesRegex(ValueError, "unlabeled"):
            handbook_history_builder(self.events.assign(Class=0))


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

    @unittest.skipUnless(DEFAULT_DATA_PATH.is_file(), "Run code/workflows/download_data.py for full-data checks")
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


class FinalWorkflowTests(unittest.TestCase):
    def test_frozen_assessment_never_trains_and_preserves_validation_threshold(self):
        from evaluation import fit_amount_bins, make_prediction_frame
        from workflows.finalize import _assessment
        from temporal import select_review_threshold
        rows=pd.DataFrame({'row_id':[1,2,3,4], 'Class':[0,1,0,1],
                           'Time':[0.,1.,2.,3.], 'Amount':[10.,20.,30.,40.]})
        val=make_prediction_frame(rows,[.1,.9,.2,.8],'fixed')
        selection=select_review_threshold(val,.5)
        test=rows.assign(row_id=[10,11,12,13],Time=[10.,11.,12.,13.])
        pred=make_prediction_frame(test,[.95,.4,.3,.2],'fixed')
        with TemporaryDirectory() as folder,patch('classification.train_model',side_effect=AssertionError('no fitting')):
            result=_assessment(test,{'fixed':pred},fit_amount_bins(rows),{'fixed':selection},folder)
        self.assertEqual(result['threshold_policies']['fixed']['threshold'],selection['threshold'])
        self.assertEqual(result['threshold_policies']['fixed']['overall']['fixed_threshold']['K'],1)
        self.assertEqual(result['partition']['rows'],4)

    def test_incomplete_freeze_and_changed_models_are_rejected_before_data_load(self):
        from argparse import Namespace
        import workflows.finalize as f
        with TemporaryDirectory() as folder:
            path=Path(folder)/'freeze.json';f.write(path,{'group_G2_complete':False})
            args=Namespace(freeze=path,output=Path(folder)/'out',results=Path(folder)/'shared')
            with patch('workflows.finalize.load_data',side_effect=AssertionError('do not load test')):
                with self.assertRaisesRegex(ValueError,'G2'):f.run_final_test(args)
            f.write(path,{'group_G2_complete':True,'provenance':{'source_sha256':{}}})
            with self.assertRaisesRegex(ValueError,'Source differs'):f.run_final_test(args)
            self.assertFalse(args.output.exists())

    def test_semantic_rules_do_not_use_mechanism_codes_and_global_groups_add_up(self):
        import numpy as np
        from pattern_mining import fit_handbook_rules,evaluate_handbook_rules
        from evaluation import make_prediction_frame
        n=120
        rows=pd.DataFrame({'row_id':np.arange(n),'Time':np.arange(n,dtype=float),
                           'Amount':np.arange(n,dtype=float)+1,
                           'customer_amount_ratio_7d':np.arange(n)/10,
                           'customer_count_7d':np.arange(n)%12,
                           'terminal_count_7d':np.arange(n)%9,
                           'Class':(np.arange(n)>99).astype(int),
                           'TX_FRAUD_SCENARIO':(np.arange(n)>99).astype(int)})
        val=rows.assign(row_id=rows.row_id+1000,Time=rows.Time+1000)
        frozen=fit_handbook_rules(rows,val)
        other=fit_handbook_rules(rows.assign(TX_FRAUD_SCENARIO=999),val.assign(TX_FRAUD_SCENARIO=999))
        self.assertEqual(frozen,other)
        pred={n:make_prediction_frame(val,val.Amount,n) for n in ['classifier','IF']}
        with patch('pattern_mining.mine_rules',side_effect=AssertionError('test must not mine')):
            result,cases=evaluate_handbook_rules(val,pred,frozen,phase='test')
        self.assertEqual(sum(g['rows'] for g in result['mechanisms']),120)
        self.assertEqual(sum(g['models']['classifier']['global_reviews'] for g in result['mechanisms']),2)
        self.assertEqual(sum(g['models']['classifier']['TP'] for g in result['mechanisms']),2)
        self.assertTrue(frozen['frozen_rules'])
        self.assertNotIn('TX_FRAUD_SCENARIO',frozen['feature_columns'])


class FinalArtifactTests(unittest.TestCase):
    def test_exported_package_source_commit_does_not_require_git(self):
        from preprocessing import source_commit
        with TemporaryDirectory() as folder:
            marker=Path(folder)/'SOURCE_COMMIT';marker.write_text('a'*40+'\n')
            (Path(folder)/'code').mkdir()
            with patch('subprocess.check_output',side_effect=AssertionError('no Git required')):
                self.assertEqual(source_commit(Path(folder)/'code'),'a'*40)

    def test_B_frozen_test_scoring_and_plot_do_not_refit_models(self):
        import json,numpy as np
        from sklearn.pipeline import Pipeline
        from classification import train_model
        from unsupervised import fit_isolation_forest
        from workflows.finalize import _b_test
        from validation.run_b_validation import save_b_extension_figure
        rng=np.random.default_rng(42)
        X=pd.DataFrame(rng.normal(size=(320,len(FEATURE_COLUMNS))),columns=FEATURE_COLUMNS)
        X['Amount']=np.abs(X.Amount)+1;X['Time']=np.arange(len(X))
        labels=(np.arange(len(X))%17==0).astype(int)
        lr=train_model(X,labels,'logistic_regression','class_weight',42)
        iff=fit_isolation_forest(X,seed=42)
        source=json.loads((Path(__file__).resolve().parents[1]/'results/tables/b_extensions.json').read_text())
        frozen={'label_summary':source,'label_models':{r['mask_sha256']:lr for r in source['E03']['conditions']},'if256':iff}
        data=X.iloc[:64].copy().assign(row_id=np.arange(1000,1064),Class=labels[:64])
        with TemporaryDirectory() as folder,patch.object(Pipeline,'fit',side_effect=AssertionError('no test fit')):
            result=_b_test(data,frozen,Path(folder)/'private')
            save_b_extension_figure(result,Path(folder)/'plot.png',phase='test')
            self.assertEqual(len(result['E03']['conditions']),15)
            self.assertEqual(len(result['E05']['importances']),30)
            self.assertTrue((Path(folder)/'plot.png').is_file())


class FrozenColdStartRegressionTests(unittest.TestCase):
    def test_all_four_groups_share_the_same_global_capacity_and_empty_groups(self):
        from workflows.finalize import cold_start_metrics
        from evaluation import make_prediction_frame
        data=pd.DataFrame({'row_id':[10,11,12,13],'Time':[1.,2.,3.,4.],
                           'Class':[0,1,0,1],'CUSTOMER_ID':[1,2,1,2],'TERMINAL_ID':[10,10,20,20]})
        entities=pd.DataFrame({'CUSTOMER_ID':[1],'TERMINAL_ID':[10]})
        views={'both_history':pd.DataFrame({'customer_no_history_7d':[0,1,0,1],
                                            'terminal_no_history_7d':[0,0,1,1]},index=data.row_id)}
        pred=make_prediction_frame(data,[.4,.6,.3,.8],'LR')
        with patch('classification.train_model',side_effect=AssertionError('no test training')):
            result=cold_start_metrics(data,views,{'LR':pred},entities)['LR']
        self.assertEqual(len(result['groups']),4)
        for i,q in enumerate(REVIEW_FRACTIONS):
            self.assertEqual(sum(g['global_budget_allocation'][i]['reviewed_in_group'] for g in result['groups']),1)
        self.assertTrue(all('budgets' not in g['within_group'] for g in result['groups']))
        seen=data.assign(CUSTOMER_ID=1,TERMINAL_ID=10)
        other=cold_start_metrics(seen,views,{'LR':pred},entities)['LR']
        self.assertEqual(sum(g['within_group']['n_samples']==0 for g in other['groups']),3)
        self.assertTrue(all(g['history_availability']['customer'] is None for g in other['groups'] if g['within_group']['n_samples']==0))

    def test_reproduction_reports_RF_diagnostics_without_hiding_core_differences(self):
        from workflows.finalize import _compare_numeric
        differences=[]
        _compare_numeric(.78,.7795,'ulb_test.json.metrics.random_forest_original.AP',platform_differences=differences)
        _compare_numeric(154,153,'ulb_test.json.metrics.random_forest_original.budgets[1].amount_groups[0].reviewed_from_global_list',platform_differences=differences)
        self.assertEqual(len(differences),2)
        for path in ['ulb_test.json.metrics.weighted_lr.AP','ulb_test.json.metrics.random_forest_original.budgets[1].TP','handbook_test.json.metrics.lr.AP']:
            with self.assertRaises(ValueError):_compare_numeric(10,9,path,platform_differences=differences)
