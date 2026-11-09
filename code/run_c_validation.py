"""负责人 C：独立的验证运行入口，不修改 A 的划分/调度和 B 的分类代码。

--demo 仅为合成检查。正式运行必须接收 A 已确定的 train/validation CSV。
不接收测试数据，不负责替 A 创建正式划分；输出默认在 Git 忽略的 data/ 下。
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from evaluation import (
    DEFAULT_BUDGETS,
    make_prediction_frame,
    evaluate_scores,
    evaluate_groups,
    compare_review_lists,
    save_figures,
    save_group_figures,
    save_comparison_figures,
)
from preprocessing import FEATURE_COLUMNS
from unsupervised import (
    compare_kmeans_candidates,
    fit_kmeans,
    score_kmeans,
    fit_isolation_forest,
    score_isolation_forest,
)

ROOT = Path(__file__).resolve().parents[1]


def _read_partition(path):
    frame = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    make_prediction_frame(frame, np.zeros(len(frame)), "input_check")
    return frame


def _demo():
    rng = np.random.default_rng(42)
    frames = []
    for start, n, frauds in [(0, 600, 20), (600, 200, 10)]:
        x = rng.normal(size=(n, len(FEATURE_COLUMNS)))
        x[-frauds:, 1:5] += 5
        frame = pd.DataFrame(x, columns=FEATURE_COLUMNS)
        frame["Time"] = np.arange(start, start + n, dtype=float)
        frame["Amount"] = rng.lognormal(size=n)
        frame["row_id"] = np.arange(start, start + n)
        frame["Class"] = 0
        frame.loc[n - frauds :, "Class"] = 1
        frames.append(frame)
    return frames


def run_validation(
    train, validation, output, classifier=None, demo=False, source_files=()
):
    """Run only validation experiments; return metrics and a proposed C config."""
    for frame in [train, validation]:
        make_prediction_frame(frame, np.zeros(len(frame)), "input_check")
    if not len(train) or not len(validation):
        raise ValueError("Both partitions must be nonempty")
    if set(train.row_id) & set(validation.row_id):
        raise ValueError("Train and validation row IDs overlap")
    if train.Time.max() >= validation.Time.min():
        raise ValueError(
            "Partitions must be chronological with disjoint timestamp ranges"
        )
    if classifier is not None and not demo:
        if (
            classifier.empty
            or classifier["method"].isna().any()
            or classifier["method"].nunique() != 1
        ):
            raise ValueError("Provide exactly one nonempty validation classifier")
        name = classifier["method"].iloc[0]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(
                "Classifier method name must use letters, numbers, underscores or hyphens"
            )
        if name in {"kmeans_distance", "isolation_forest", "random_seed42"}:
            raise ValueError("Classifier name conflicts with a built-in method")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output folder must be new or empty, preserving earlier run evidence"
        )
    Xtrain, Xval = train[FEATURE_COLUMNS], validation[FEATURE_COLUMNS]
    candidates = compare_kmeans_candidates(Xtrain)
    averages = {}
    for row in candidates:
        if row["sample_silhouette"] is not None:
            averages.setdefault(row["k"], []).append(row["sample_silhouette"])
    if not averages:
        raise ValueError(
            "No valid training silhouette; investigate data before selecting k"
        )
    k = min(averages, key=lambda key: (-float(np.mean(averages[key])), key))
    km = fit_kmeans(Xtrain, k, 42)
    clusters, distance = score_kmeans(km, Xval)
    forest = fit_isolation_forest(Xtrain, 42)
    anomaly = score_isolation_forest(forest, Xval)
    predictions = {
        "kmeans_distance": make_prediction_frame(
            validation, distance, "kmeans_distance", clusters
        ),
        "isolation_forest": make_prediction_frame(
            validation, anomaly, "isolation_forest", clusters
        ),
    }
    rng = np.random.default_rng(42)
    predictions["random_seed42"] = make_prediction_frame(
        validation, rng.random(len(validation)), "random_seed42"
    )
    if demo:
        # A deterministic illustration, not B's classifier or a scientific result.
        predictions["synthetic_proxy"] = make_prediction_frame(
            validation, Xval["V1"].to_numpy(), "synthetic_proxy"
        )
        classifier = predictions["synthetic_proxy"]
    elif classifier is not None:
        classifier = make_prediction_frame(
            classifier, classifier["score"], classifier["method"].iloc[0]
        )
        # Validate metadata and row alignment before writing any output.
        compare_review_lists(classifier, predictions["isolation_forest"], 1)
        classifier = classifier.set_index("row_id").loc[validation.row_id].reset_index()
        predictions[classifier["method"].iloc[0]] = classifier
    metrics = {
        name: evaluate_scores(frame.Class, frame.score, row_ids=frame.row_id)
        for name, frame in predictions.items()
    }
    groups = {
        name: evaluate_groups(frame, clusters, DEFAULT_BUDGETS)
        for name, frame in predictions.items()
    }
    combos = []
    if classifier is not None:
        for q in DEFAULT_BUDGETS:
            count = math.ceil(len(validation) * q)
            for share in [0.0, 0.25, 0.5, 0.75, 1.0]:
                row = compare_review_lists(
                    classifier, predictions["isolation_forest"], count, share
                )
                row["review_fraction"] = q
                combos.append(row)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in predictions.items():
        frame.to_csv(output / f"{name}_predictions.csv", index=False)
    pd.DataFrame(candidates).to_csv(
        output / "kmeans_training_structure.csv", index=False
    )
    profile = validation[FEATURE_COLUMNS].copy()
    profile["cluster_id"] = clusters
    profile.groupby("cluster_id").mean().to_csv(
        output / "validation_cluster_feature_means.csv"
    )
    group_rows = [
        {
            "method": name,
            "cluster_id": g["cluster_id"],
            "n_samples": g["within_group"]["n_samples"],
            "n_positive": g["within_group"]["n_positive"],
            "prevalence": g["within_group"]["prevalence"],
            "fraud_coverage": g["fraud_coverage"],
            "AP": g["within_group"]["AP"],
        }
        for name, result in groups.items()
        for g in result["groups"]
    ]
    pd.DataFrame(group_rows).to_csv(output / "validation_groups.csv", index=False)
    budget_rows = [
        {"method": name, **row} for name, m in metrics.items() for row in m["budgets"]
    ]
    pd.DataFrame(budget_rows).to_csv(output / "validation_budgets.csv", index=False)
    label = (
        "SYNTHETIC DEMO - not project results"
        if demo
        else "Validation - not final test"
    )
    save_figures({"methods": metrics, "label": label}, output / "figures")
    save_group_figures(groups["isolation_forest"], output / "figures", label)
    if combos:
        save_comparison_figures(
            [c for c in combos if c["classifier_share"] == 0.5],
            output / "figures",
            label + " (50/50 quota illustration)",
        )

    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    )
    metadata = {
        "mode": "synthetic_demo" if demo else "validation",
        "label": label,
        "source_commit": commit,
        "source_dirty": dirty,
        "code_sha256": {
            p.name: digest(p)
            for p in [
                Path(__file__),
                ROOT / "code/evaluation.py",
                ROOT / "code/unsupervised.py",
            ]
        },
        "input_sha256": {str(p): digest(p) for p in source_files},
        "versions": {
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "scikit-learn", "matplotlib"]
        },
        "train_count": len(train),
        "validation_count": len(validation),
        "train_positive": int(train.Class.sum()),
        "validation_positive": int(validation.Class.sum()),
        "selected_k": k,
        "k_selection": "highest mean training sample silhouette; smaller k on ties",
        "seed": 42,
        "kmeans_n_init": 10,
        "forest_params": forest.named_steps["model"].get_params(),
        "scaler": "StandardScaler fitted on training only",
        "forest_score": "negative score_samples",
        "review_fractions": list(DEFAULT_BUDGETS),
        "combination_shares": [0, 0.25, 0.5, 0.75, 1],
        "pending": "A-approved data provenance; B final classifier; validation choices and G2 freeze; final test and final report",
        "NOT_A_FROZEN_FINAL_CONFIG": True,
    }
    report = {
        "metadata": metadata,
        "metrics": metrics,
        "groups": groups,
        "comparisons": combos,
    }
    (output / "validation_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--train", type=Path)
    p.add_argument("--validation", type=Path)
    p.add_argument("--classification-predictions", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.demo and any(
        [args.train, args.validation, args.classification_predictions]
    ):
        p.error("--demo cannot be combined with real data inputs")
    if not args.demo and not (args.train and args.validation):
        p.error("Provide --demo OR A-approved --train and --validation exports")
    try:
        if args.demo:
            train, val = _demo()
            files = []
            classifier = None
        else:
            train, val = _read_partition(args.train), _read_partition(args.validation)
            files = [args.train, args.validation]
            classifier = (
                pd.read_csv(args.classification_predictions)
                if args.classification_predictions
                else None
            )
            if args.classification_predictions:
                files.append(args.classification_predictions)
            if classifier is not None and (
                classifier.empty or classifier["method"].nunique() != 1
            ):
                raise ValueError(
                    "Classifier predictions must contain exactly one method"
                )
        output = args.output or ROOT / "data" / (
            "c_demo" if args.demo else "c_validation"
        )
        report = run_validation(train, val, output, classifier, args.demo, files)
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(output),
                "mode": report["metadata"]["mode"],
                "selected_k": report["metadata"]["selected_k"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
