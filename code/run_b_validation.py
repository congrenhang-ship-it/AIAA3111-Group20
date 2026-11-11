"""B2-B5: bounded classification comparison using A's splits and C's metrics.

Only training and validation are scored. Full predictions/models stay in data/.
Run: py -3.12 code/run_b_validation.py
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from classification import MODEL_NAMES, score_model, train_model
from evaluation import DEFAULT_BUDGETS, PRIMARY_REVIEW_FRACTION, evaluate_scores, make_prediction_frame, save_figures
from main import split_summary
from preprocessing import DEFAULT_DATA_PATH, FEATURE_COLUMNS, ULB_SHA256, data_sha256, get_features, load_data, split_data

ROOT = Path(__file__).resolve().parents[1]
# Declared before any validation run. No grid search or test evaluation.
EXPERIMENTS = [(name, "original") for name in MODEL_NAMES] + [
    ("logistic_regression", "class_weight"),
    ("logistic_regression", "random_undersample"),
]
SELECTION_RULE = "maximum validation TP at 1%; then AP; then method name ascending"


def select_method(results):
    def key(record):
        metric = record["metrics"]
        budget = next(row for row in metric["budgets"] if row["review_fraction"] == PRIMARY_REVIEW_FRACTION)
        return (-budget["TP"], -metric["AP"], record["method"])
    return min(results, key=key)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run(data_path=DEFAULT_DATA_PATH):
    digest = data_sha256(data_path)
    if digest != ULB_SHA256:
        raise ValueError(f"ULB SHA-256 mismatch: {digest}")
    train, validation, test = split_data(load_data(data_path))
    partitions = split_summary((train, validation, test))
    # No test scores, test-based selection or train+validation refitting.
    del test
    X_train, X_validation = get_features(train), get_features(validation)
    local = ROOT / "data" / "b_validation"
    local.mkdir(parents=True, exist_ok=True)
    tables = ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    records, rows, predictions, models = [], [], {}, {}
    for name, strategy in EXPERIMENTS:
        method = f"{name}_{strategy}"
        start = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with threadpool_limits(limits=1):
                model = train_model(X_train, train.Class, name, strategy=strategy, seed=42)
                train_seconds = perf_counter() - start
                scores = score_model(model, X_validation)
        frame = make_prediction_frame(validation, scores, method)
        frame.to_csv(local / f"{method}.csv", index=False, float_format="%.17g")
        metric = evaluate_scores(frame.Class, frame.score, DEFAULT_BUDGETS, row_ids=frame.row_id)
        # PR arrays are large; preserve full metrics locally, compact evidence in Git.
        write_json(local / f"{method}_metrics.json", metric)
        compact = {key: value for key, value in metric.items() if key != "pr_curve"}
        record = dict(method=method, model_name=name, strategy=strategy,
                      train_seconds=train_seconds, seed=42,
                      classifier_parameters=model.named_steps["classifier"].get_params(),
                      fitted_class_counts=model.training_class_counts_, metrics=compact)
        records.append(record)
        predictions[method] = frame
        models[method] = model
        for budget in metric["budgets"]:
            rows.append(dict(method=method, AP=metric["AP"], train_seconds=train_seconds, **budget))
        print(f"{method}: AP={metric['AP']:.6f}, train={train_seconds:.2f}s", flush=True)

    selected = select_method(records)
    metrics = {record["method"]: json.loads((local / f"{record['method']}_metrics.json").read_text(encoding="utf-8")) for record in records}
    for figure in save_figures({"methods": metrics, "label": "B classification - validation only"}, local / "figures"):
        path = Path(figure)
        target = ROOT / "results" / "figures" / ("b_" + path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    selected_frame = predictions[selected["method"]]
    selected_path = local / "selected_validation_predictions.csv"
    selected_frame.to_csv(selected_path, index=False, float_format="%.17g")
    model_path = local / "selected_model.joblib"
    joblib.dump(models[selected["method"]], model_path)
    np.testing.assert_array_equal(
        score_model(joblib.load(model_path), X_validation), selected_frame.score.to_numpy()
    )
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")}
    locked = dict(line.split("==", 1) for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if "==" in line and not line.startswith("#"))
    mismatches = {name: dict(actual=version, required=locked.get(name)) for name, version in versions.items() if locked.get(name) != version}
    code_files = ["code/classification.py", "code/run_b_validation.py", "code/preprocessing.py", "code/evaluation.py"]
    metadata = dict(
        owner="B", phase="validation", dataset_sha256=digest, partitions=partitions,
        python=platform.python_version(), versions=versions,
        dependency_lock_mismatches=mismatches,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in code_files},
        command="py -3.12 code/run_b_validation.py",
    )
    write_json(tables / "b_validation.json", dict(metadata=metadata, selection_rule=SELECTION_RULE, experiments=records, selected=selected["method"]))
    pd.DataFrame(rows).to_csv(tables / "b_validation.csv", index=False)
    freeze = dict(
        schema_version=1, owner="B", status="B_validation_selection_pending_group_review",
        group_G2_complete=False, test_evaluated=False, dataset_sha256=digest,
        partitions=partitions, feature_columns=FEATURE_COLUMNS, seed=42,
        method=selected["method"], model_name=selected["model_name"], strategy=selected["strategy"],
        classifier_parameters=selected["classifier_parameters"],
        preprocessing="StandardScaler for logistic_regression/gaussian_nb; passthrough for trees",
        resampling="training only; seeded 1:1 random undersampling without replacement when selected",
        selection_rule=SELECTION_RULE, primary_review_fraction=PRIMARY_REVIEW_FRACTION,
        review_fractions=DEFAULT_BUDGETS, ranking="score descending; row_id ascending; K=ceil(N*q)",
        threshold=None, threshold_reason="Budget ranking defines decisions; no fixed probability threshold selected.",
        refit_policy="training partition only, never train+validation",
        predictions_path="data/b_validation/selected_validation_predictions.csv",
        predictions_sha256=data_sha256(selected_path), provenance=metadata,
        fitted_model_path="data/b_validation/selected_model.joblib",
    )
    write_json(tables / "b_freeze.json", freeze)
    print(f"Selected: {selected['method']}; evidence: results/tables/b_validation.csv", flush=True)
    return freeze


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    run(parser.parse_args().data)
