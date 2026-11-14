"""负责人 A：运行入口与流程整合。

任务：统一数据路径、随机种子、时间划分和审核预算；接入 B/C/D 的模块。
路线：数据 → 分类/聚类/异常检测/规则 → 验证冻结 → 测试 → 时间分析与图表。
当前可用：数据审计、划分检查、G1、C 实验与全组统一验证。
C 负责 --run-c / --run-c-baseline 的接入，A 的原入口与接口保持不变。
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from preprocessing import (
    DEFAULT_DATA_PATH,
    ID_COLUMN,
    SPLIT_FRACTIONS,
    TARGET_COLUMN,
    ULB_SHA256,
    audit_data,
    data_sha256,
    get_features,
    load_data,
    split_data,
)


ROOT = DEFAULT_DATA_PATH.parents[1]
SEED = 42
REVIEW_FRACTIONS = (0.001, 0.005, 0.01, 0.02)
TIME_WINDOW_SECONDS = 3600  # 初始建议；窗口口径须在最终测试前确定。

TASKS = {
    "A": [
        "main.py: 流程整合",
        "preprocessing.py: 数据与时间划分",
        "temporal.py: 时间窗口分析",
    ],
    "B": ["classification.py: 分类模型、不平衡处理、风险分数"],
    "C": [
        "unsupervised.py: K-means、Isolation Forest、簇距离",
        "evaluation.py: 评价、等预算组合、图表",
    ],
    "D": [
        "pattern_mining.py: 分箱、Apriori、规则验证",
        "report/、poster/: 文档整合，全组提供内容",
    ],
}


def split_summary(parts):
    """Report split counts and a fingerprint of the ordered source row IDs."""
    summary = {}
    for name, data in zip(("train", "validation", "test"), parts, strict=True):
        fingerprint = sha256()
        for row_id in data[ID_COLUMN]:
            fingerprint.update(f"{row_id}\n".encode("ascii"))
        summary[name] = {
            "rows": len(data),
            "frauds": int(data[TARGET_COLUMN].sum()),
            "time_min": float(data["Time"].iloc[0]),
            "time_max": float(data["Time"].iloc[-1]),
            "row_ids_sha256": fingerprint.hexdigest(),
        }
    return summary


def run_validation(train, validation):
    """Run the first G1 classification baseline using B and C's interfaces."""
    import numpy as np

    from classification import score_model, train_model
    from evaluation import evaluate_scores, make_prediction_frame
    from temporal import evaluate_time_windows

    model_name = "logistic_regression"
    strategy = "original"
    train_features = get_features(train)
    try:
        model = train_model(
            train_features,
            train[TARGET_COLUMN],
            model_name,
            strategy=strategy,
            seed=SEED,
        )
        scores = score_model(model, get_features(validation))
    except NotImplementedError as exc:
        raise RuntimeError(f"G1 is waiting for B2 (classification): {exc}") from exc

    try:
        predictions = make_prediction_frame(validation, scores, model_name)
        metadata = [ID_COLUMN, TARGET_COLUMN, "Time"]
        if (
            not predictions[metadata]
            .reset_index(drop=True)
            .equals(validation[metadata].reset_index(drop=True))
        ):
            raise ValueError(
                "C's prediction frame must preserve validation rows and their order."
            )
        if not predictions["method"].isin([model_name]).all():
            raise ValueError(
                "C's prediction frame must preserve the requested method name."
            )
        if not np.isfinite(predictions["score"].to_numpy(dtype=float)).all():
            raise ValueError("C's prediction frame must contain finite risk scores.")
        overall = evaluate_scores(
            predictions[TARGET_COLUMN],
            predictions["score"],
            REVIEW_FRACTIONS,
            row_ids=predictions[ID_COLUMN],
        )
        windows = evaluate_time_windows(
            predictions,
            window_seconds=TIME_WINDOW_SECONDS,
            origin=0,
            review_fractions=REVIEW_FRACTIONS,
        )
    except NotImplementedError as exc:
        raise RuntimeError(f"G1 is waiting for C2 (shared evaluation): {exc}") from exc

    return {
        "split": "validation",
        "model": model_name,
        "strategy": strategy,
        "seed": SEED,
        "feature_columns": list(train_features.columns),
        "review_fractions": REVIEW_FRACTIONS,
        "window_seconds": TIME_WINDOW_SECONDS,
        "window_origin": 0,
        "overall": overall,
        "time_windows": windows,
    }


def run_c_workflow(args):
    """C: adapt shared CLI options; keep detailed exports in the local run folder."""
    import pandas as pd
    from evaluation import publish_c_results
    from unsupervised import (
        _demo,
        _read_partition,
        run_validation as run_c_baseline,
        run_experiment,
        reproduce_b_for_c,
    )
    from threadpoolctl import threadpool_limits

    if args.demo and (
        args.train
        or args.validation
        or args.classification_predictions
        or args.classification_freeze
    ):
        raise ValueError(
            "--demo cannot be combined with real partition/classifier inputs"
        )
    if args.classification_freeze and args.classification_predictions:
        raise ValueError("Choose classifier predictions OR a B freeze manifest")
    if args.classification_freeze and args.run_c_baseline:
        raise ValueError("B freeze integration uses --run-c")
    source_info = {}
    if args.demo:
        train, validation = _demo()
        files = []
        source_info = {"mode": "synthetic_demo", "verified_ulb_split": False}
    elif args.train or args.validation:
        if not (args.train and args.validation):
            raise ValueError("Provide both --train and --validation")
        train, validation = (
            _read_partition(args.train),
            _read_partition(args.validation),
        )
        files = [args.train, args.validation]
    else:
        digest = data_sha256(args.data)
        if digest != ULB_SHA256:
            raise ValueError("ULB SHA-256 mismatch")
        train, validation, held_out = split_data(load_data(args.data))
        del held_out
        files = [args.data]
        source_info = {
            "mode": "validation",
            "verified_ulb_split": True,
            "dataset_sha256": digest,
        }
    classifier = (
        pd.read_csv(args.classification_predictions)
        if args.classification_predictions
        else None
    )
    if args.classification_predictions:
        files.append(args.classification_predictions)
    mode = "baseline" if args.run_c_baseline else "improved"
    output = args.output or DEFAULT_DATA_PATH.parent / ("c_" + mode)
    private_path, shared_path = output.resolve(), args.results.resolve()
    if (
        private_path == shared_path
        or private_path in shared_path.parents
        or shared_path in private_path.parents
    ):
        raise ValueError("Private and shared output folders must be separate")
    # Check destination before fitting; other team members' files may coexist.
    table_name = "c_baseline.json" if mode == "baseline" else "c_improved.json"
    if (args.results / "tables" / table_name).exists():
        raise ValueError(
            "C result files already exist; choose another --results directory"
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output folder must be new or empty")
    from tempfile import TemporaryDirectory
    import shutil

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="c_export_", dir=output.parent) as temporary:
        staging = Path(temporary) / "export"
        with threadpool_limits(limits=1):
            b_reproduction = None
            b_model = None
            if args.classification_freeze:
                classifier, b_model, b_reproduction = reproduce_b_for_c(
                    train,
                    validation,
                    args.classification_freeze,
                    Path(temporary) / "b_reproduction",
                    source_info.get("dataset_sha256"),
                )
            if mode == "baseline":
                run_c_baseline(
                    train,
                    validation,
                    output,
                    classifier,
                    args.demo,
                    files,
                    source_info=source_info,
                    public_output=staging,
                )
            else:
                result, bundle = run_experiment(
                    train,
                    validation,
                    output,
                    staging,
                    source_info=source_info,
                    classifier=classifier,
                )
        if b_reproduction is not None:
            import joblib

            result["B_reproduction"] = b_reproduction
            status = (
                "C_quota_frozen_B_reproduced_in_common_environment"
                if b_reproduction["selection_matches_published"]
                else "C_quota_selected_B_choice_needs_environment_review"
            )
            bundle["config"]["B_combination"].update(
                status=status,
                manifest_sha256=b_reproduction["B_manifest_sha256"],
                prediction_sha256=b_reproduction["prediction_sha256"],
                source_branch_commit=b_reproduction["B_branch_commit"],
            )
            bundle["B_model"] = b_model
            result["config"] = bundle["config"]
            joblib.dump(bundle, output / "improved_models.joblib")
            (staging / "summary.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            (staging / "c_improved_freeze.json").write_text(
                json.dumps(bundle["config"], indent=2, allow_nan=False) + "\n"
            )
            shutil.copytree(
                Path(temporary) / "b_reproduction", output / "b_reproduction"
            )
        paths = publish_c_results(staging, args.results, mode)
        shutil.copytree(staging, output / "aggregate")
    return {"mode": mode, "private_output": str(output), "shared_files": paths}


def run_team_validation(data_path, output):
    """Rebuild the delivered candidates and evaluate only A's validation rows."""
    from importlib.metadata import version
    from math import isclose
    import platform
    import subprocess
    from tempfile import TemporaryDirectory

    import joblib
    from threadpoolctl import threadpool_limits

    from classification import score_model, train_model
    from evaluation import (
        build_c_extension_config,
        evaluate_c_extensions,
        evaluate_improved_frozen,
        make_prediction_frame,
        publish_frozen_c_results,
        save_c_extension_figures,
    )
    from run_d_validation import evaluate_rule_extensions, rule_training_metrics
    from temporal import evaluate_review_threshold, select_review_threshold
    from unsupervised import fit_configuration, score_improved_bundle

    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def source_sha256(path):
        return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()

    output = output.resolve()
    if DEFAULT_DATA_PATH.parent.resolve() not in output.parents:
        raise ValueError("Team validation output must be inside the ignored data/ directory.")
    if output.exists():
        raise ValueError("Choose a new --output directory for team validation.")
    lock = ROOT / "requirements.txt"
    pins = dict(line.split("==", 1) for line in lock.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#"))
    versions = {name: version(name) for name in pins}
    if versions != pins:
        raise ValueError("Team validation requires the versions pinned in requirements.txt.")
    digest = data_sha256(data_path)
    if digest != ULB_SHA256:
        raise ValueError("ULB SHA-256 mismatch.")
    parts = split_data(load_data(data_path))
    partitions = split_summary(parts)
    train, validation = parts[:2]
    del parts

    tables = ROOT / "results/tables"
    b_path, c_path, d_path = (tables / name for name in
                             ("b_freeze.json", "c_improved.json", "d_freeze.json"))
    extension_path = tables / "c_extensions.json"
    g1_path = tables / "d_g1_reproduction.json"
    rule_path = ROOT / "report/d_validation/train_validation_rules.csv"
    b, c, d = map(read_json, (b_path, c_path, d_path))
    config = c["config"]
    for record in (b, config, d):
        for name in ("train", "validation"):
            actual = partitions[name]
            expected = record["partitions"][name]
            if any(actual[key] != expected[key] for key in ("rows", "frauds", "row_ids_sha256")):
                raise ValueError(f"Published candidate uses a different {name} split.")
    if b["dataset_sha256"] != digest or config["dataset_sha256"] != digest:
        raise ValueError("B/C candidates use a different raw dataset.")
    if (
        config["B_combination"]["method"] != b["method"]
        or config["B_combination"]["manifest_sha256"] != source_sha256(b_path)
        or config["feature_columns"] != b["feature_columns"]
        or list(get_features(train).columns) != b["feature_columns"]
    ):
        raise ValueError("B manifest differs from C's recorded classifier handoff.")

    specs = c["selection_record"]["candidate_specifications"]
    selected_specs = {}
    for family, name in config["selected_candidates"].items():
        (selected_specs[family],) = (
            spec for spec in specs if spec["family"] == family
            and name in (spec["id"], spec["id"] + "_raw", spec["id"] + "_relative")
        )
    with threadpool_limits(limits=1):
        g1 = run_validation(train, validation)["overall"]
        reference = read_json(g1_path)
        if not isclose(g1["AP"], reference["AP"], rel_tol=1e-12) or g1["budgets"] != reference["budgets"]:
            raise ValueError("The G1 baseline differs from D's independent reproduction.")
        x_train = get_features(train)
        classifier = train_model(x_train, train.Class, b["model_name"], b["strategy"], b["seed"])
        if classifier.named_steps["classifier"].get_params() != b["classifier_parameters"]:
            raise ValueError("Rebuilt classifier parameters differ from the B manifest.")
        bundle = {
            "config": config,
            "models": {family: fit_configuration(x_train, spec, config["final_seed"])
                       for family, spec in selected_specs.items()},
            "B_model": classifier,
        }
        classifier_scores = score_model(classifier, get_features(validation))
        scored = score_improved_bundle(bundle, validation)
    b_frame = make_prediction_frame(validation, classifier_scores, b["method"])
    predictions = {**scored[0], b["method"]: b_frame}
    q = config["primary_review_fraction"]
    main_predictions = {"classifier": b_frame, "IF": predictions[config["selected_anomaly"]]}
    thresholds = {}
    for frame in main_predictions.values():
        selection = select_review_threshold(frame, q)
        thresholds[frame.method.iloc[0]] = {
            **selection, **evaluate_review_threshold(frame, selection, window_seconds=TIME_WINDOW_SECONDS),
        }
    rules, rule_model_metrics = evaluate_rule_extensions(
        validation, main_predictions, d, rule_training_metrics(rule_path), q,
    )
    extension_config = build_c_extension_config(train, bundle)
    published_extensions = read_json(extension_path)
    if extension_config != published_extensions["config"]:
        raise ValueError("C extension settings differ from the published validation configuration.")

    published_d_path = tables / "d_extensions.json"
    provenance = {
        "command": subprocess.list2cmdline(["python", "code/main.py", *sys.argv[1:]]),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "dataset_sha256": digest,
        "python": platform.python_version(),
        "versions": versions,
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): source_sha256(path)
            for path in [b_path, c_path, d_path, published_d_path, extension_path,
                         g1_path, rule_path, lock, *sorted((ROOT / "code").glob("*.py"))]
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".team-validation-", dir=output.parent) as temporary:
        stage = Path(temporary) / "run"
        core = evaluate_improved_frozen(
            bundle, validation, stage / "core", classifier=b_frame, phase="validation_replay",
            window_seconds=TIME_WINDOW_SECONDS, window_origin=0, scored_unsupervised=scored,
        )
        for name, metrics in core["metrics"].items():
            previous = c["validation_metrics"][name]
            if not isclose(metrics["AP"], previous["AP"], rel_tol=1e-12) or metrics["budgets"] != previous["budgets"]:
                raise ValueError(f"Rebuilt validation metrics differ for {name}.")
        prior_policy = [row for row in c["B_comparisons"]
                        if row["classifier_share"] == config["B_combination"]["share"]]
        replay_policy = [{key: row[key] for key in expected}
                         for row, expected in zip(core["B_comparisons"], prior_policy, strict=True)]
        if replay_policy != prior_policy:
            raise ValueError("Equal-budget policy differs from C's published validation.")
        published_d = read_json(published_d_path)
        if rules != published_d["E04"]:
            raise ValueError("Shared E04 predictions did not reproduce D's rule evidence.")
        extensions = evaluate_c_extensions(predictions, validation, extension_config, phase="validation")
        for key in ("metrics", "cost_sensitivity", "uncertainty"):
            if extensions[key] != published_extensions[key]:
                raise ValueError(f"C validation replay differs from published {key}.")

        share = stage / "aggregate"
        publish_frozen_c_results(stage / "core", share, provenance=provenance)
        extensions["provenance"] = provenance
        figures = save_c_extension_figures(extensions, share / "figures")
        extensions["publication"] = {"figure_sha256": {path.name: data_sha256(path) for path in figures}}
        write_json(share / "tables/c_extensions.json", extensions)
        write_json(stage / "rule_cases.json", rules.pop("cases"))
        bundle_path = stage / "models.joblib"
        joblib.dump(bundle, bundle_path)
        report = {
            "phase": "validation",
            "group_G2_complete": False,
            "test_scored": False,
            "provenance": provenance,
            "partitions": partitions,
            "g1": {key: g1[key] for key in ("AP", "budgets")},
            "fixed_fits": {
                "bundle_sha256": data_sha256(bundle_path), "classifier": b["method"],
                "classifier_seed": b["seed"], "classifier_parameters": b["classifier_parameters"],
                "unsupervised_seed": config["final_seed"], "unsupervised_specs": selected_specs,
            },
            "window_settings": {"seconds": TIME_WINDOW_SECONDS, "origin": 0},
            "E04": {"coverage": rules, "model_validation": rule_model_metrics},
            "E08": thresholds,
            "pending_extensions": ["E03", "E05"],
        }
        write_json(share / "tables/team_validation.json", report)
        stage.rename(output)
    return {"phase": "validation", "output": str(output), "test_scored": False}


def main():
    parser = argparse.ArgumentParser(
        description="Group 20 data preparation and validation baseline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the original creditcard.csv file.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check-data", action="store_true", help="Run a read-only data audit."
    )
    action.add_argument(
        "--prepare-data",
        action="store_true",
        help="Verify ULB and report the shared splits.",
    )
    action.add_argument(
        "--run-validation",
        action="store_true",
        help="Run the G1 baseline on validation only.",
    )
    action.add_argument(
        "--run-c", action="store_true", help="Run C improvements on A shared split."
    )
    action.add_argument(
        "--run-c-baseline",
        action="store_true",
        help="Reproduce C original validation workflow.",
    )
    action.add_argument(
        "--run-team-validation", action="store_true",
        help="Replay the delivered A/B/C/D candidates and extensions on validation.",
    )
    parser.add_argument(
        "--output", type=Path, help="New private run folder under data/."
    )
    parser.add_argument(
        "--results",
        "--publish",
        dest="results",
        type=Path,
        default=DEFAULT_DATA_PATH.parents[1] / "results",
        help="Shared results root (figures/ and tables/).",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Synthetic check for C only."
    )
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--classification-predictions", type=Path)
    parser.add_argument(
        "--classification-freeze",
        type=Path,
        help="Reproduce B manifest in shared environment and complete C combination.",
    )
    parser.add_argument(
        "--tasks", action="store_true", help="Show agreed module ownership."
    )
    args = parser.parse_args()
    run_data_action = (
        args.check_data
        or args.prepare_data
        or args.run_validation
        or args.run_c
        or args.run_c_baseline
        or args.run_team_validation
    )
    if args.tasks:
        print(json.dumps(TASKS, indent=2, ensure_ascii=False))
    if not run_data_action:
        if not args.tasks:
            parser.print_help()
        return 0
    if args.run_team_validation:
        if not args.output:
            parser.error("--run-team-validation requires a new --output directory")
        if args.demo or args.train or args.validation or args.classification_predictions or args.classification_freeze:
            parser.error("--run-team-validation uses the shared ULB split and published candidate files")
        if args.results != ROOT / "results":
            parser.error("--run-team-validation stores aggregate results inside --output")
        try:
            print(json.dumps(run_team_validation(args.data, args.output), indent=2))
        except (ValueError, OSError, KeyError, ImportError) as exc:
            parser.error(str(exc))
        return 0
    if args.run_c or args.run_c_baseline:
        try:
            print(json.dumps(run_c_workflow(args), indent=2, ensure_ascii=False))
        except (ValueError, OSError, KeyError, ImportError) as exc:
            parser.error(str(exc))
        return 0
    if (
        args.demo
        or args.train
        or args.validation
        or args.output
        or args.classification_predictions
        or args.classification_freeze
    ):
        parser.error("C run options require --run-c or --run-c-baseline")
    if not args.data.is_file():
        parser.error(f"Data file not found: {args.data}. See data/README.md.")
    try:
        sha256 = data_sha256(args.data)
        matches_ulb = sha256 == ULB_SHA256
        if not args.check_data and not matches_ulb:
            raise ValueError(f"ULB SHA-256 mismatch: {sha256}. See data/README.md.")
        data = load_data(args.data)
        if args.check_data:
            summary = audit_data(data)
        else:
            parts = split_data(data)
            summary = {
                "split_fractions_target": SPLIT_FRACTIONS,
                "rows_removed_as_exact_duplicates": len(data) - sum(map(len, parts)),
                "partitions": split_summary(parts),
            }
            if args.run_validation:
                summary["experiment"] = run_validation(parts[0], parts[1])
        summary.update(sha256=sha256, matches_ulb_v1=matches_ulb)
        output = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
    except ModuleNotFoundError:
        parser.error(
            "Install dependencies with: python -m pip install -r requirements.txt"
        )
    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
