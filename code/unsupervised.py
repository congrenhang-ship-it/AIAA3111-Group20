"""负责人 C：K-means、Isolation Forest、特征消融及验证实验。

基线与改进实验均从 main.py 运行；公共评价和导出在 evaluation.py。
所有缩放与距离标定仅拟合训练段，冻结模型评分不重新拟合或选参。
"""

import itertools
import numpy as np
import copy
import math
import pandas as pd
import hashlib
import importlib.metadata
import json
import re
import subprocess
import joblib
import platform
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from preprocessing import FEATURE_COLUMNS, data_sha256
from evaluation import (
    make_prediction_frame,
    evaluate_scores,
    compare_review_lists,
    evaluate_groups,
    DEFAULT_BUDGETS,
    save_figures,
    save_group_figures,
    save_comparison_figures,
    partition_description,
    choose_anomaly,
    build_freeze,
    write_summary,
    save_diagnostic_figures,
    COMBINATION_SHARES,
    select_combination,
)
from time import perf_counter
from datetime import datetime, timezone
from temporal import evaluate_time_windows
from evaluation import save_ablation_figure

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _features(X, allow_empty=False):
    forbidden = {"Class", "row_id", "cluster_id", "method", "score"}
    if hasattr(X, "columns"):
        if not X.columns.is_unique or forbidden & set(X.columns):
            raise ValueError(
                "Inputs must have unique feature columns, excluding target/metadata"
            )
    try:
        a = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be numeric") from exc
    if (
        a.ndim != 2
        or not a.shape[1]
        or (not allow_empty and not len(a))
        or not np.isfinite(a).all()
    ):
        raise ValueError(
            "Features must be a finite nonempty 2-D matrix; resolve missing values with A"
        )
    return X


def fit_kmeans(X_train, n_clusters=5, seed=42):
    """Fit StandardScaler + KMeans with explicit deterministic initialization."""
    _features(X_train)
    if (
        isinstance(n_clusters, bool)
        or int(n_clusters) != n_clusters
        or not 2 <= n_clusters <= len(X_train)
    ):
        raise ValueError("n_clusters must be an integer between 2 and training size")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KMeans(n_clusters=int(n_clusters), n_init=10, random_state=seed)),
        ]
    ).fit(X_train)


def score_kmeans(model, X):
    """Return cluster IDs and distances in the fitted standardized feature space."""
    _features(X, allow_empty=True)
    if not len(X):
        return np.array([], dtype=int), np.array([], dtype=float)
    transformed = model.named_steps["scaler"].transform(X)
    estimator = model.named_steps["model"]
    clusters = estimator.predict(transformed)
    distances = np.linalg.norm(
        transformed - estimator.cluster_centers_[clusters], axis=1
    )
    return clusters, distances


def fit_isolation_forest(X_train, seed=42, **params):
    """Fit on mixed training features only, without Class or label-based filtering."""
    _features(X_train)
    if "random_state" in params:
        raise ValueError("Use seed rather than passing random_state twice")
    settings = {
        "n_estimators": 100,
        "max_samples": 256,
        "contamination": "auto",
        "n_jobs": 1,
    }
    settings.update(params)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", IsolationForest(random_state=seed, **settings)),
        ]
    ).fit(X_train)


def score_isolation_forest(model, X):
    """Negate score_samples so higher is more suspicious; do not use hard labels."""
    _features(X, allow_empty=True)
    if not len(X):
        return np.array([], dtype=float)
    return -model.score_samples(X)


def compare_kmeans_candidates(
    X_train, ks=(3, 5, 8), seeds=(42, 43), sample_size=1000, progress=None
):
    """Return structural metrics and seed stability on ONE fixed training subset.

    Candidate selection can use mean defined silhouette with smaller k as tie
    break. Degenerate samples get None, never fabricated silhouette. No labels
    or validation rows enter this routine. Pairwise ARI measures seed stability.
    """
    _features(X_train)
    ks, seeds = tuple(ks), tuple(seeds)
    if (
        not ks
        or len(set(ks)) != len(ks)
        or len(seeds) < 2
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Provide distinct ks and at least two distinct seeds")
    if (
        isinstance(sample_size, bool)
        or int(sample_size) != sample_size
        or sample_size < 2
    ):
        raise ValueError("sample_size must be an integer >= 2")
    rng = np.random.default_rng(42)
    indices = np.sort(
        rng.choice(
            len(X_train), size=min(int(sample_size), len(X_train)), replace=False
        )
    )
    sample = (
        X_train.iloc[indices]
        if hasattr(X_train, "iloc")
        else np.asarray(X_train)[indices]
    )
    records = []
    for k in ks:
        assignments = []
        for seed in seeds:
            if progress is not None:
                progress(f"K-means training: k={k}, seed={seed}")
            pipeline = fit_kmeans(X_train, k, seed)
            labels, _ = score_kmeans(pipeline, sample)
            transformed = pipeline.named_steps["scaler"].transform(sample)
            count = len(np.unique(labels))
            silhouette = (
                float(silhouette_score(transformed, labels))
                if 1 < count < len(sample)
                else None
            )
            assignments.append(labels)
            records.append(
                {
                    "k": int(k),
                    "seed": int(seed),
                    "inertia": float(pipeline.named_steps["model"].inertia_),
                    "sample_size": len(sample),
                    "sample_silhouette": silhouette,
                }
            )
        stability = float(
            np.mean(
                [
                    adjusted_rand_score(x, y)
                    for x, y in itertools.combinations(assignments, 2)
                ]
            )
        )
        for row in records[-len(seeds) :]:
            row["seed_stability_ARI"] = stability
    return records


def score_frozen_unsupervised(bundle, X):
    """A's final scoring hook: reuse persisted training pipelines, never select/refit.

    bundle is the local joblib artifact written by the baseline validation workflow. Load only
    your own trusted artifact. Caller supplies features only; labels are unused.
    Whole-group G2 and permission to evaluate the test partition belong to A.
    """
    _features(X, allow_empty=True)
    config = bundle["config"]
    if hasattr(X, "columns") and list(X.columns) != config["feature_columns"]:
        raise ValueError("Frozen feature order does not match scoring input")
    if bundle["kmeans"].named_steps["model"].n_clusters != config["n_clusters"]:
        raise ValueError("Frozen K-means configuration disagrees with fitted model")
    groups, distance = score_kmeans(bundle["kmeans"], X)
    return {
        "cluster_ids": groups,
        "kmeans_distance": distance,
        "isolation_forest": score_isolation_forest(bundle["isolation_forest"], X),
    }


class TransactionFeatures(TransformerMixin, BaseEstimator):
    """Stateless feature ablation, retaining input order and explicit column names."""

    def __init__(self, drop_time=False, log_amount=False):
        self.drop_time = drop_time
        self.log_amount = log_amount

    def _check(self, X):
        if not isinstance(X, pd.DataFrame) or list(X.columns) != FEATURE_COLUMNS:
            raise ValueError(
                "Pass exactly the declared 30 raw features in their original order"
            )
        a = X.to_numpy(dtype=float)
        if not np.isfinite(a).all() or (X.Amount < 0).any():
            raise ValueError("Features must be finite, with nonnegative Amount")

    def fit(self, X, y=None):
        self._check(X)
        if not len(X):
            raise ValueError("Training features must be nonempty")
        self.feature_names_in_ = np.array(FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        self._check(X)
        columns = [c for c in FEATURE_COLUMNS if not (self.drop_time and c == "Time")]
        out = X.loc[:, columns].copy()
        if self.log_amount:
            out["Amount"] = np.log1p(out.Amount.to_numpy(dtype=float))
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return np.array(
            [c for c in FEATURE_COLUMNS if not (self.drop_time and c == "Time")],
            dtype=object,
        )


def configurations():
    """14 fixed fits yield 20 scoring candidates. No outcome-dependent expansion."""
    km = []
    for name, drop, log, scaler in [
        ("base", False, False, "standard"),
        ("no_time", True, False, "standard"),
        ("log_amount", False, True, "standard"),
        ("no_time_log", True, True, "standard"),
        ("robust", False, False, "robust"),
        ("no_time_log_robust", True, True, "robust"),
    ]:
        km.append(
            {
                "id": "km_" + name,
                "family": "kmeans",
                "drop_time": drop,
                "log_amount": log,
                "scaler": scaler,
                "n_clusters": 3,
            }
        )
    forest = []
    for name, drop, log, trees, sample in [
        ("base", False, False, 100, 256),
        ("no_time", True, False, 100, 256),
        ("log_amount", False, True, 100, 256),
        ("no_time_log", True, True, 100, 256),
        ("sample1024", False, False, 100, 1024),
        ("sample4096", False, False, 100, 4096),
        ("trees300", False, False, 300, 256),
        ("joint", True, True, 300, 1024),
    ]:
        forest.append(
            {
                "id": "if_" + name,
                "family": "isolation_forest",
                "drop_time": drop,
                "log_amount": log,
                "scaler": "standard",
                "n_estimators": trees,
                "max_samples": sample,
            }
        )
    return km + forest


def inner_time_split(train, fraction=0.8):
    """Development split ONLY inside A's training partition; equal times stay together."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0,1)")
    ordered = train.sort_values(["Time", "row_id"])
    if not ordered.row_id.is_unique:
        raise ValueError("Training row IDs must be unique")
    target = int(len(ordered) * fraction)
    if not 0 < target < len(ordered):
        raise ValueError("Not enough rows for an inner split")
    end = ordered.Time.searchsorted(ordered.Time.iloc[target], side="left")
    if not 0 < end < len(ordered):
        raise ValueError("Inner time cut must leave two nonempty parts")
    first, last = ordered.iloc[:end].copy(), ordered.iloc[end:].copy()
    if not last.Class.sum():
        raise ValueError("Inner selection requires positive cases")
    return first, last


def fit_configuration(X, spec, seed=42):
    """Fit one label-free pipeline and, for K-means, training-only distance stats."""
    features = TransactionFeatures(spec["drop_time"], spec["log_amount"])
    scaler = StandardScaler() if spec["scaler"] == "standard" else RobustScaler()
    if spec["family"] == "kmeans":
        estimator = KMeans(n_clusters=spec["n_clusters"], n_init=10, random_state=seed)
    elif spec["family"] == "isolation_forest":
        estimator = IsolationForest(
            n_estimators=spec["n_estimators"],
            max_samples=spec["max_samples"],
            contamination="auto",
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError("Unknown model family")
    pipe = Pipeline(
        [("features", features), ("scaler", scaler), ("model", estimator)]
    ).fit(X)
    fitted = {"pipeline": pipe, "spec": copy.deepcopy(spec), "seed": int(seed)}
    if spec["family"] == "kmeans":
        z = pipe[:-1].transform(X)
        groups = estimator.predict(z)
        distance = np.linalg.norm(z - estimator.cluster_centers_[groups], axis=1)
        stats = []
        for group in range(spec["n_clusters"]):
            d = distance[groups == group]
            if not len(d):
                raise ValueError("Degenerate empty training cluster")
            median = float(np.median(d))
            iqr = float(np.quantile(d, 0.75) - np.quantile(d, 0.25))
            floor = 1e-9 * max(1.0, abs(median))
            stats.append(
                {
                    "cluster_id": group,
                    "n_train": len(d),
                    "median": median,
                    "iqr": iqr,
                    "denominator": max(iqr, floor),
                }
            )
        fitted["distance_stats"] = stats
    return fitted


def score_configuration(fitted, X):
    """Return scores in input order; no fit, label access or per-query calibration."""
    pipe = fitted["pipeline"]
    spec = fitted["spec"]
    name = spec["id"]
    if not len(X):
        pipe.named_steps["features"].transform(X)
        suffixes = ("_raw", "_relative") if spec["family"] == "kmeans" else ("",)
        return {name + s: np.array([], dtype=float) for s in suffixes}, np.array(
            [], dtype=int
        )
    if spec["family"] == "isolation_forest":
        return {name: -pipe.score_samples(X)}, None
    z = pipe[:-1].transform(X)
    model = pipe.named_steps["model"]
    groups = model.predict(z)
    distance = np.linalg.norm(z - model.cluster_centers_[groups], axis=1)
    medians = np.array([s["median"] for s in fitted["distance_stats"]])
    denominators = np.array([s["denominator"] for s in fitted["distance_stats"]])
    relative = (distance - medians[groups]) / denominators[groups]
    return {name + "_raw": distance, name + "_relative": relative}, groups


def candidate_order(specs=None):
    return [
        name
        for s in (configurations() if specs is None else specs)
        for name in (
            [s["id"] + "_raw", s["id"] + "_relative"]
            if s["family"] == "kmeans"
            else [s["id"]]
        )
    ]


def candidate_summary(rows, order):
    table = pd.DataFrame(rows)
    result = []
    for name in order:
        group = table[table.candidate == name]
        if group.empty:
            continue
        result.append(
            {
                "candidate": name,
                "mean_TP": float(group.TP.mean()),
                "min_TP": int(group.TP.min()),
                "max_TP": int(group.TP.max()),
                "mean_AP": float(group.AP.mean()),
                "seeds": len(group),
                "TP_std": float(group.TP.std(ddof=0)),
            }
        )
    return result


def select_families(rows, order):
    """Training-inner mean TP, mean AP, declaration order. Never validation selection."""
    summary = candidate_summary(rows, order)
    key = lambda r: (-r["mean_TP"], -r["mean_AP"], order.index(r["candidate"]))
    return {
        family: min([r for r in summary if r["candidate"].startswith(prefix)], key=key)[
            "candidate"
        ]
        for family, prefix in [("kmeans", "km_"), ("isolation_forest", "if_")]
    }


def select_pair(comparison_rows):
    """Fixed shares: mean inner TP, pure method on ties, then larger distance share."""
    return min(
        SHARES,
        key=lambda share: (
            -float(
                np.mean(
                    [
                        r["combined"]["TP"]
                        for r in comparison_rows
                        if r["classifier_share"] == share
                    ]
                )
            ),
            0 if share in (0.0, 1.0) else 1,
            -share,
        ),
    )


def score_improved_bundle(bundle, data):
    """Use persisted selected fits and statistics; return standard prediction tables."""
    config = bundle["config"]
    X = data[FEATURE_COLUMNS]
    out = {}
    groups = None
    for family, selected in config["selected_candidates"].items():
        scores, cluster_ids = score_configuration(bundle["models"][family], X)
        if selected not in scores:
            raise ValueError("Frozen score name disagrees with artifact")
        out[selected] = make_prediction_frame(data, scores[selected], selected)
        if family == "kmeans":
            groups = cluster_ids
    return out, groups


def pair_comparisons(predictions, selected, share, budgets=DEFAULT_BUDGETS):
    first = predictions[selected["kmeans"]]
    second = predictions[selected["isolation_forest"]]
    rows = []
    for q in budgets:
        if not len(first):
            continue
        row = compare_review_lists(first, second, math.ceil(len(first) * q), share)
        row["review_fraction"] = q
        rows.append(row)
    return rows


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
    train,
    validation,
    output,
    classifier=None,
    demo=False,
    source_files=(),
    *,
    source_info=None,
    public_output=None,
    progress=None,
):
    """Run only validation experiments; return metrics and a proposed C config."""
    started = perf_counter()
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
    if public_output is not None:
        public_path = Path(public_output).resolve()
        private_path = output.resolve()
        if (
            public_path == private_path
            or public_path in private_path.parents
            or private_path in public_path.parents
        ):
            raise ValueError("Public and private output folders must be separate")
        if public_path.exists() and any(public_path.iterdir()):
            raise ValueError("Public summary folder must be new or empty")
    Xtrain, Xval = train[FEATURE_COLUMNS], validation[FEATURE_COLUMNS]
    candidates = compare_kmeans_candidates(Xtrain, progress=progress)
    averages = {}
    for row in candidates:
        if row["sample_silhouette"] is not None:
            averages.setdefault(row["k"], []).append(row["sample_silhouette"])
    if not averages:
        raise ValueError(
            "No valid training silhouette; investigate data before selecting k"
        )
    k = min(averages, key=lambda key: (-float(np.mean(averages[key])), key))
    if progress is not None:
        progress(f"Selected k={k} from training structure; fitting final C models")
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
        name: evaluate_groups(frame, clusters, DEFAULT_BUDGETS, group_values=range(k))
        for name, frame in predictions.items()
    }
    selected_anomaly = choose_anomaly(metrics)
    anomaly_combos = []
    for q in DEFAULT_BUDGETS:
        for share in COMBINATION_SHARES:
            row = compare_review_lists(
                predictions["isolation_forest"],
                predictions["kmeans_distance"],
                math.ceil(len(validation) * q),
                share,
            )
            row["review_fraction"] = q
            anomaly_combos.append(row)
    combos = []
    if classifier is not None:
        for q in DEFAULT_BUDGETS:
            count = math.ceil(len(validation) * q)
            for share in COMBINATION_SHARES:
                row = compare_review_lists(
                    classifier, predictions[selected_anomaly], count, share
                )
                row["review_fraction"] = q
                combos.append(row)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in predictions.items():
        frame.to_csv(output / f"{name}_predictions.csv", index=False)
    pd.DataFrame(candidates).to_csv(
        output / "kmeans_training_structure.csv", index=False
    )
    train_clusters, _ = score_kmeans(km, Xtrain)
    train_groups = (
        train.assign(cluster_id=train_clusters)
        .groupby("cluster_id")
        .Class.agg(["size", "sum", "mean"])
    )
    train_groups.columns = ["n_samples", "n_positive", "prevalence"]
    train_groups["fraud_coverage"] = (
        train_groups.n_positive / train.Class.sum() if train.Class.sum() else np.nan
    )
    train_groups.to_csv(output / "train_cluster_profile.csv")
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
    save_group_figures(
        groups[selected_anomaly], output / "figures", label + "\n" + selected_anomaly
    )
    if combos:
        save_comparison_figures(
            [c for c in combos if c["classifier_share"] == 0.5],
            output / "figures",
            label + " (50/50 quota illustration)",
        )

    save_comparison_figures(
        [r for r in anomaly_combos if r["classifier_share"] == 0.5],
        output / "figures",
        label + "\nAnomaly pair at 50/50 quota (diagnostic)",
        method_labels={
            "classifier": "Isolation Forest",
            "anomaly": "K-means distance",
            "combined": "Combined",
        },
        filename="anomaly_complementarity.png",
    )
    cases = []
    for name, frame in predictions.items():
        ranked = frame.sort_values(["score", "row_id"], ascending=[False, True])
        count = math.ceil(len(ranked) * 0.01)
        for reason, candidates_ in [
            ("reviewed_false_positive", ranked.iloc[:count].query("Class == 0")),
            ("missed_fraud", ranked.iloc[count:].query("Class == 1")),
        ]:
            sample = candidates_.head(5).copy()
            sample["case_type"] = reason
            cases.append(sample)
    pd.concat(cases, ignore_index=True).to_csv(
        output / "private_error_examples.csv", index=False
    )
    save_diagnostic_figures(candidates, groups, output / "figures", label)
    from temporal import evaluate_time_windows

    windows = evaluate_time_windows(pd.concat(predictions.values(), ignore_index=True))
    for window in windows:
        if window["evaluation"] is not None:
            window["evaluation"].pop("pr_curve", None)

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
                ROOT / "code/main.py",
                ROOT / "code/preprocessing.py",
                ROOT / "code/temporal.py",
            ]
        },
        "input_sha256": {str(p): digest(p) for p in source_files},
        "versions": {
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "scikit-learn", "matplotlib"]
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "partitions": {
            "train": partition_description(train),
            "validation": partition_description(validation),
        },
        "classifier_method": None if classifier is None else classifier.method.iloc[0],
        "elapsed_seconds": perf_counter() - started,
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
        "pending": "B5 classifier provenance; whole-group G2; A8 final test",
        "NOT_A_FROZEN_FINAL_CONFIG": True,
    }
    metadata.update(source_info or {})
    report = {
        "metadata": metadata,
        "metrics": metrics,
        "groups": groups,
        "comparisons": combos,
        "anomaly_comparisons": anomaly_combos,
        "time_windows": windows,
    }
    (output / "validation_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    frozen = build_freeze(report)
    (output / "c_freeze.json").write_text(
        json.dumps(frozen, indent=2, allow_nan=False) + "\n"
    )
    joblib.dump(
        {"kmeans": km, "isolation_forest": forest, "config": frozen},
        output / "c_models.joblib",
    )
    if public_output is not None:
        write_summary(report, public_output)
        import shutil

        destination = Path(public_output)
        for name in [
            "kmeans_training_structure.csv",
            "train_cluster_profile.csv",
            "validation_cluster_feature_means.csv",
        ]:
            shutil.copy2(output / name, destination / name)
        shutil.copytree(output / "figures", destination / "figures")
    return report


def _dump(value, path):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _safe_comparisons(rows):
    return [
        {
            k: v
            for k, v in row.items()
            if k not in {"classifier_ids", "anomaly_ids", "combined_ids"}
        }
        for row in rows
    ]


def _source():
    paths = [
        ROOT / "code" / name
        for name in [
            "unsupervised.py",
            "evaluation.py",
            "main.py",
            "preprocessing.py",
            "temporal.py",
        ]
    ]
    paths.append(ROOT / "report/C_IMPROVEMENTS.md")
    return {
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip(),
        "source_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "code_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]
        },
        "python": platform.python_version(),
    }


def _outcomes(predictions):
    metrics = {}
    for name, p in predictions.items():
        m = evaluate_scores(p.Class, p.score, row_ids=p.row_id)
        m.pop("pr_curve", None)
        metrics[name] = m
    return metrics


def run_experiment(
    train,
    validation,
    private,
    public,
    *,
    specs=None,
    seeds=SEEDS,
    source_info=None,
    classifier=None,
    progress=print,
):
    """Return aggregate report and a persisted selected bundle; never score A's test."""
    started = perf_counter()
    specs = configurations() if specs is None else specs
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct and nonempty")
    for frame in [train, validation]:
        make_prediction_frame(frame, np.zeros(len(frame)), "input_check")
    if (
        not len(train)
        or not len(validation)
        or set(train.row_id) & set(validation.row_id)
        or train.Time.max() >= validation.Time.min()
    ):
        raise ValueError("Need A chronological nonempty, disjoint train/validation")
    private, public = Path(private), Path(public)
    a, b = private.resolve(), public.resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError("Private/public outputs must be disjoint")
    for folder in [private, public]:
        if folder.exists() and any(folder.iterdir()):
            raise ValueError("Output folders must be new or empty")
    origin = _source()
    inner_fit, inner_selection = inner_time_split(train)
    order = candidate_order(specs)
    scores_by_seed = {}
    inner_rows = []
    timings = []
    for seed in seeds:
        scores_by_seed[seed] = {}
        for spec in specs:
            progress(f"Inner fit seed={seed}: {spec['id']}")
            clock = perf_counter()
            fitted = fit_configuration(inner_fit[FEATURE_COLUMNS], spec, seed)
            scores, _ = score_configuration(fitted, inner_selection[FEATURE_COLUMNS])
            timings.append(
                {
                    "stage": "inner",
                    "seed": seed,
                    "configuration": spec["id"],
                    "seconds": perf_counter() - clock,
                }
            )
            for name, values in scores.items():
                prediction = make_prediction_frame(inner_selection, values, name)
                scores_by_seed[seed][name] = prediction
                metric = evaluate_scores(
                    prediction.Class,
                    prediction.score,
                    [0.01],
                    row_ids=prediction.row_id,
                )
                row = metric["budgets"][0]
                inner_rows.append(
                    {
                        "candidate": name,
                        "seed": seed,
                        "K": row["K"],
                        "TP": row["TP"],
                        "AP": metric["AP"],
                        "precision": row["precision_at_k"],
                        "recall": row["recall_at_k"],
                    }
                )
    selected = select_families(inner_rows, order)
    aggregates = candidate_summary(inner_rows, order)
    by_name = {r["candidate"]: r for r in aggregates}
    overall = min(
        selected.values(),
        key=lambda name: (
            -by_name[name]["mean_TP"],
            -by_name[name]["mean_AP"],
            order.index(name),
        ),
    )
    inner_pairs = []
    for seed in seeds:
        for share in SHARES:
            row = pair_comparisons(
                scores_by_seed[seed], selected, share, budgets=(0.01,)
            )[0]
            row["seed"] = seed
            inner_pairs.append(row)
    chosen_share = select_pair(inner_pairs)
    info = dict(source_info or {})
    config = {
        "schema_version": 2,
        "owner": "C",
        "round": "bounded_improvement_v2",
        "status": "inner_selected_C_candidate"
        if info.get("verified_ulb_split")
        else "synthetic_or_unverified_not_for_official_test",
        "group_G2_complete": False,
        "test_evaluated": False,
        "feature_columns": FEATURE_COLUMNS,
        "n_clusters": 3,
        "kmeans_n_init": 10,
        "selected_candidates": selected,
        "selected_anomaly": overall,
        "distance_share": chosen_share,
        "selection_rule": "training-inner mean TP at 1%; mean AP; declaration order",
        "pair_selection_rule": "training-inner mean TP at 1%; pure on ties; larger distance share",
        "selection_seeds": list(seeds),
        "final_seed": 42,
        "candidate_count": len(order),
        "primary_review_fraction": 0.01,
        "review_fractions": list(DEFAULT_BUDGETS),
        "ranking": "score descending; row_id ascending; K=ceil(N*q)",
        "refill": "distance quota floor(K*share) first; forest scan skipping duplicates to K",
        "partitions": {
            "train": partition_description(train),
            "validation": partition_description(validation),
            "inner_fit": partition_description(inner_fit),
            "inner_selection": partition_description(inner_selection),
        },
        "B_combination": {"status": "waiting_for_B5", "method": None, "share": None},
        "refit_policy": "full A training only, final seed 42; never train+validation",
        "selection_locked_at_UTC": datetime.now(timezone.utc).isoformat(),
        "prior_validation_already_observed": True,
        **origin,
        **info,
    }
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    selection_record = {
        "config": config,
        "candidate_specifications": specs,
        "inner_summary": aggregates,
        "inner_per_seed": inner_rows,
        "inner_pair_comparisons": _safe_comparisons(inner_pairs),
    }
    # Persist the actual choices BEFORE any full-training fit or validation score.
    _dump(selection_record, private / "selection_locked_before_validation.json")
    _dump(selection_record, public / "selection_locked_before_validation.json")
    progress(
        f"LOCKED before validation: {selected}; overall={overall}; distance_share={chosen_share}"
    )
    predictions = {}
    final_models = {}
    cluster_vectors = {}
    stats = {}
    for spec in specs:
        progress(f"Full-training fit / validation report: {spec['id']}")
        clock = perf_counter()
        fitted = fit_configuration(train[FEATURE_COLUMNS], spec, 42)
        values, groups = score_configuration(fitted, validation[FEATURE_COLUMNS])
        final_models[spec["id"]] = fitted
        if groups is not None:
            for name in values:
                cluster_vectors[name] = groups
            stats[spec["id"]] = fitted["distance_stats"]
        for name, score in values.items():
            predictions[name] = make_prediction_frame(validation, score, name)
        timings.append(
            {
                "stage": "full_train",
                "seed": 42,
                "configuration": spec["id"],
                "seconds": perf_counter() - clock,
            }
        )
    selected_models = {
        family: final_models[
            next(
                s["id"]
                for s in specs
                if name == s["id"] or name in {s["id"] + "_raw", s["id"] + "_relative"}
            )
        ]
        for family, name in selected.items()
    }
    bundle = {"config": config, "models": selected_models}
    metrics = _outcomes(predictions)
    comparisons = pair_comparisons(predictions, selected, chosen_share)
    all_shares = []
    for share in SHARES:
        all_shares += pair_comparisons(predictions, selected, share)
    group_ids = cluster_vectors[selected["kmeans"]]
    display_names = list(dict.fromkeys(["km_base_raw", "if_base", *selected.values()]))
    display = {name: predictions[name] for name in display_names if name in predictions}
    B_comparisons = []
    if classifier is not None:
        if (
            classifier.empty
            or classifier.method.isna().any()
            or classifier.method.nunique() != 1
        ):
            raise ValueError("One classifier method required")
        name = classifier.method.iloc[0]
        if (
            not isinstance(name, str)
            or not name.replace("_", "").replace("-", "").isalnum()
            or name in predictions
        ):
            raise ValueError("Invalid/conflicting classifier name")
        c = make_prediction_frame(classifier, classifier.score, name)
        compare_review_lists(c, predictions[overall], 1)
        c = c.set_index("row_id").loc[validation.row_id].reset_index()
        for q in DEFAULT_BUDGETS:
            for share in SHARES:
                row = compare_review_lists(
                    c, predictions[overall], math.ceil(len(validation) * q), share
                )
                row["review_fraction"] = q
                B_comparisons.append(row)
        chosen_B = select_combination(B_comparisons)
        config["B_combination"] = {
            "status": "validation_selected_requires_B5_provenance",
            "method": name,
            "share": chosen_B["classifier_share"],
        }
        config["B_prediction_sha256"] = hashlib.sha256(
            c.to_csv(index=False).encode()
        ).hexdigest()
        display[name] = c
        metrics[name] = evaluate_scores(c.Class, c.score, row_ids=c.row_id)
        metrics[name].pop("pr_curve", None)
    display_metrics = {
        name: evaluate_scores(p.Class, p.score, row_ids=p.row_id)
        for name, p in display.items()
    }
    group_metrics = {
        name: evaluate_groups(p, group_ids, DEFAULT_BUDGETS, group_values=range(3))
        for name, p in display.items()
    }
    windows = evaluate_time_windows(pd.concat(display.values(), ignore_index=True))
    for window in windows:
        if window["evaluation"] is not None:
            window["evaluation"].pop("pr_curve", None)
    figures = public / "figures"
    figures.mkdir()
    save_figures(
        {
            "methods": display_metrics,
            "label": "Validation - C improvements selected on inner data",
        },
        figures,
    )
    save_group_figures(
        group_metrics[overall],
        figures,
        "Validation - selected improved C\n" + overall,
        comparison_groups=(
            {
                "B classifier": group_metrics[config["B_combination"]["method"]],
                "C anomaly": group_metrics[overall],
            }
            if classifier is not None
            else None
        ),
    )
    save_comparison_figures(
        comparisons,
        figures,
        "Validation - inner-frozen anomaly pair",
        method_labels={
            "classifier": selected["kmeans"],
            "anomaly": selected["isolation_forest"],
            "combined": "Frozen pair",
        },
    )
    summary_rows = []
    for name in order:
        m = metrics[name]
        budget = next(r for r in m["budgets"] if r["review_fraction"] == 0.01)
        summary_rows.append(
            {
                "candidate": name,
                "AP": m["AP"],
                "K": budget["K"],
                "TP": budget["TP"],
                "FP": budget["FP"],
                "Recall": budget["recall_at_k"],
                "Precision": budget["precision_at_k"],
                "inner_selected": name in selected.values(),
            }
        )
    pd.DataFrame(summary_rows).to_csv(public / "validation_ablation.csv", index=False)
    save_ablation_figure(summary_rows, selected, figures / "ablation.png")
    if B_comparisons:
        save_comparison_figures(
            [
                row
                for row in B_comparisons
                if row["classifier_share"] == config["B_combination"]["share"]
            ],
            figures,
            f"Validation - frozen classifier share {config['B_combination']['share']:.0%}",
            method_labels={
                "classifier": config["B_combination"]["method"],
                "anomaly": overall,
                "combined": "Frozen policy",
            },
            filename="B_complementarity.png",
        )
    budget_rows = [
        {"candidate": name, **row}
        for name, m in metrics.items()
        for row in m["budgets"]
    ]
    pd.DataFrame(budget_rows).to_csv(public / "validation_budgets.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(public / "inner_per_seed.csv", index=False)
    pd.DataFrame(aggregates).to_csv(public / "inner_summary.csv", index=False)
    pd.DataFrame(timings).to_csv(public / "timings.csv", index=False)
    _dump(stats, public / "training_distance_calibration.json")
    # Only selected and baseline predictions need full standard tables. All scores remain reproducible locally.
    for name, p in display.items():
        p.to_csv(private / f"{name}_predictions.csv", index=False)
    validation[["row_id", "Class", "Time"]].to_csv(
        private / "validation_metadata.csv", index=False
    )
    np.savez_compressed(
        private / "all_candidate_scores.npz",
        **{name: p.score.to_numpy() for name, p in predictions.items()},
    )
    case_rows = []
    for name, p in display.items():
        ordered = p.sort_values(["score", "row_id"], ascending=[False, True])
        k = math.ceil(len(p) * 0.01)
        for kind, subset in [
            ("false_positive", ordered.iloc[:k].query("Class==0")),
            ("missed_fraud", ordered.iloc[k:].query("Class==1")),
        ]:
            sample = subset.head(5).copy()
            sample["case_type"] = kind
            case_rows.append(sample)
    pd.concat(case_rows, ignore_index=True).to_csv(
        private / "private_error_examples.csv", index=False
    )
    joblib.dump(bundle, private / "improved_models.joblib")
    _dump(config, public / "c_improved_freeze.json")
    result = {
        "metadata": {
            "phase": "validation",
            "run_complete": True,
            "elapsed_seconds": perf_counter() - started,
            **origin,
            **info,
        },
        "config": config,
        "validation_metrics": metrics,
        "inner_summary": aggregates,
        "groups": group_metrics,
        "time_windows": windows,
        "frozen_pair_comparisons": _safe_comparisons(comparisons),
        "all_pair_shares_diagnostic_only": _safe_comparisons(all_shares),
        "B_comparisons": _safe_comparisons(B_comparisons),
    }
    _dump(result, public / "summary.json")
    # Verify saved scores and source stability before declaring the run complete.
    restored = joblib.load(private / "improved_models.joblib")
    replay, _ = score_improved_bundle(restored, validation)
    for name, p in replay.items():
        np.testing.assert_allclose(
            p.score, predictions[name].score, rtol=1e-12, atol=1e-12
        )
    if _source()["code_sha256"] != origin["code_sha256"]:
        raise ValueError("Source changed while experiment was running")
    verification = {
        "persisted_selected_scores_match": True,
        "test_evaluated": False,
        "baseline_reproduced": None,
    }
    baseline = ROOT / "results/tables/c_baseline.json"
    if info.get("verified_ulb_split") and baseline.is_file():
        old = json.loads(baseline.read_text())
        for before, after in [
            ("kmeans_distance", "km_base_raw"),
            ("isolation_forest", "if_base"),
        ]:
            assert abs(old["metrics"][before]["AP"] - metrics[after]["AP"]) < 1e-10
            assert [r["TP"] for r in old["metrics"][before]["budgets"]] == [
                r["TP"] for r in metrics[after]["budgets"]
            ]
        verification["baseline_reproduced"] = True
    _dump(verification, public / "verification.json")
    return result, bundle


def reproduce_b_for_c(train, validation, freeze_path, output, dataset_digest):
    """C: reproduce B's declared six candidates in the shared pinned environment.

    B's source and published artifacts are read-only. The supplied B proposal
    remains the classifier under analysis; a changed winner is reported for
    group review, never silently substituted. No test features are accepted.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from classification import train_model, score_model
    from run_b_validation import EXPERIMENTS, select_method

    freeze_path, output = Path(freeze_path), Path(output)
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen.get("owner") != "B" or frozen.get("test_evaluated"):
        raise ValueError("Expected B's validation-only classifier manifest")
    if not dataset_digest or frozen.get("dataset_sha256") != dataset_digest:
        raise ValueError("B manifest does not match the verified dataset")
    for name, frame in [("train", train), ("validation", validation)]:
        expected = frozen["partitions"][name]
        actual = partition_description(frame)
        if any(
            expected[key] != actual[key] for key in ["rows", "frauds", "row_ids_sha256"]
        ):
            raise ValueError("B manifest split does not match A: " + name)
    if frozen.get("feature_columns") != FEATURE_COLUMNS or frozen.get("seed") != 42:
        raise ValueError("B feature order or seed differs from the common protocol")
    actual_versions = {
        p: importlib.metadata.version(p)
        for p in ["numpy", "pandas", "scikit-learn", "scipy"]
    }
    lock = dict(
        line.split("==", 1)
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if "==" in line and not line.startswith("#")
    )
    mismatches = {
        p: {"actual": v, "required": lock.get(p)}
        for p, v in actual_versions.items()
        if lock.get(p) != v
    }
    if mismatches:
        raise ValueError(
            "B reproduction requires the pinned shared environment: "
            + json.dumps(mismatches)
        )
    proposed = frozen["method"]
    if proposed != f"{frozen['model_name']}_{frozen['strategy']}":
        raise ValueError("B method name is inconsistent with model/strategy")
    if output.exists() and any(output.iterdir()):
        raise ValueError("B reproduction output must be new or empty")
    records = []
    selected_model = None
    selected_frame = None
    X_train, X_validation = train[FEATURE_COLUMNS], validation[FEATURE_COLUMNS]
    for name, strategy in EXPERIMENTS:
        method = f"{name}_{strategy}"
        clock = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = train_model(X_train, train.Class, name, strategy=strategy, seed=42)
            scores = score_model(model, X_validation)
        metric = evaluate_scores(validation.Class, scores, row_ids=validation.row_id)
        metric.pop("pr_curve", None)
        records.append(
            {
                "method": method,
                "model_name": name,
                "strategy": strategy,
                "fit_score_evaluation_seconds": perf_counter() - clock,
                "metrics": metric,
                "classifier_parameters": model.named_steps["classifier"].get_params(),
            }
        )
        print(
            f"B common-environment reproduction: {method}, AP={metric['AP']:.6f}",
            flush=True,
        )
        if method == proposed:
            selected_model = model
            selected_frame = make_prediction_frame(validation, scores, method)
    if selected_model is None:
        raise ValueError("B proposed method is not among its declared candidates")
    actual_params = selected_model.named_steps["classifier"].get_params()
    differences = {
        key: {"published": value, "actual": actual_params.get(key)}
        for key, value in frozen["classifier_parameters"].items()
        if actual_params.get(key) != value
    }
    if differences:
        raise ValueError(
            "B declared classifier parameters changed: " + json.dumps(differences)
        )
    winner = select_method(records)["method"]
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, output / "selected_model.joblib")
    replay = score_model(joblib.load(output / "selected_model.joblib"), X_validation)
    np.testing.assert_allclose(replay, selected_frame.score, rtol=1e-12, atol=1e-15)
    ids = validation.row_id.to_numpy()
    np.testing.assert_array_equal(
        np.lexsort((ids, -replay)), np.lexsort((ids, -selected_frame.score.to_numpy()))
    )
    selected_frame.to_csv(
        output / "selected_validation_predictions.csv",
        index=False,
        float_format="%.17g",
    )
    record = {
        "owner": "C",
        "purpose": "B handoff reproduction, not a new classifier search",
        "B_manifest_sha256": data_sha256(freeze_path),
        "B_manifest_source_commit": frozen["provenance"]["source_commit"],
        "B_branch_commit": subprocess.run(
            ["git", "rev-parse", "origin/b/classification-interface"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "B_published_versions": frozen["provenance"]["versions"],
        "actual_versions": actual_versions,
        "dependency_lock_mismatches": {},
        "code_sha256": {
            name: data_sha256(ROOT / name)
            for name in [
                "code/classification.py",
                "code/run_b_validation.py",
                "requirements.txt",
            ]
        },
        "dataset_sha256": dataset_digest,
        "partitions": {
            n: partition_description(f)
            for n, f in [("train", train), ("validation", validation)]
        },
        "experiments": records,
        "proposed_method": proposed,
        "selected_under_common_environment": winner,
        "selection_matches_published": winner == proposed,
        "parameters_match_published": True,
        "model_replay_scores_and_full_ranking_match": True,
        "test_evaluated": False,
        "prediction_sha256": data_sha256(
            output / "selected_validation_predictions.csv"
        ),
        "published_prediction_sha256": frozen["predictions_sha256"],
    }
    _dump(record, output / "reproduction.json")
    return selected_frame, selected_model, record


# C-N1/N2/N3: generic, causal Handbook interfaces. A owns data generation and
# customer/terminal count/amount features; C never substitutes its own dataset.
HANDBOOK_VIEWS = ("transaction_only", "customer_history", "terminal_history", "both_history")
RELATION_COLUMNS = ("customer_terminal_degree_7d", "terminal_customer_degree_7d", "new_customer_terminal_pair_7d")


def _relation_events(transactions):
    from evaluation import _row_ids, _vector
    required = {"row_id", "Time", "CUSTOMER_ID", "TERMINAL_ID"}
    if not isinstance(transactions, pd.DataFrame) or not required <= set(transactions):
        raise ValueError("Relation stream requires row_id/Time/CUSTOMER_ID/TERMINAL_ID")
    ids = _row_ids(transactions.row_id, len(transactions))
    times = _vector(transactions.Time, "relation Time", len(transactions))
    if (times < 0).any():
        raise ValueError("Relation times must be nonnegative seconds")
    for column in ["CUSTOMER_ID", "TERMINAL_ID"]:
        for value in transactions[column]:
            if not isinstance(value, (str, int, np.integer)) or isinstance(value, (bool, np.bool_)) or (isinstance(value, str) and not value.strip()):
                raise ValueError("Entity keys must be nonempty strings or integers")
    return pd.DataFrame({"row_id": ids, "Time": times,
                         "CUSTOMER_ID": transactions.CUSTOMER_ID.to_numpy(),
                         "TERMINAL_ID": transactions.TERMINAL_ID.to_numpy()})


def history_visibility_mask(transactions, missing_fraction=0.0, seed=42):
    """Label/amount-independent nested mask, stable under chunking and row order.

    SplitMix64 mixes source IDs with seed; top 53 bits define a fixed uniform.
    Unsigned wraparound is intentional. True means the event enters history
    AFTER its time batch is scored; current transactions are never deleted.
    """
    from evaluation import _integer, _row_ids, _vector
    rate = float(_vector([missing_fraction], "missing fraction")[0])
    if not 0 <= rate <= 1:
        raise ValueError("Missing fraction must be in [0,1]")
    seed = _integer(seed, "history seed", 0, 2**32-1)
    ids = _row_ids(transactions.row_id, len(transactions))
    x = ids.astype(np.uint64) ^ np.uint64(seed)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    uniform = (x >> np.uint64(11)).astype(float) / 2**53
    return pd.Series(uniform >= rate, index=pd.Index(ids, name="row_id"), name="history_keep")


def _checked_history_mask(events, history_keep):
    if history_keep is None:
        return pd.Series(True, index=pd.Index(events.row_id, name="row_id"))
    if not isinstance(history_keep, pd.Series) or not history_keep.index.is_unique or set(history_keep.index) != set(events.row_id):
        raise ValueError("History mask must be a boolean Series keyed by all source row IDs")
    mask = history_keep.loc[events.row_id]
    if mask.isna().any() or mask.dtype.kind != "b":
        raise ValueError("History mask must contain only booleans")
    return mask


def build_relation_features(transactions, *, history_keep=None, window_seconds=7*86400):
    """Past-only bipartite statistics on [t-window,t), no labels or amounts read.

    Same-time queries are all answered before any events from that batch enter
    the graph. Multiplicity counters preserve an edge until its LAST event ages
    out. Returned numeric frame is keyed by original row_id in input order.
    """
    from collections import Counter, defaultdict, deque
    from evaluation import _vector
    events = _relation_events(transactions)
    window = float(_vector([window_seconds], "relation window")[0])
    if window <= 0:
        raise ValueError("Relation window must be positive")
    keep = _checked_history_mask(events, history_keep)
    ordered = events.sort_values(["Time", "row_id"])
    ordered = ordered.assign(keep=keep.loc[ordered.row_id].to_numpy())
    customers, terminals = defaultdict(Counter), defaultdict(Counter)
    queue = deque()
    features = {}
    batches = itertools.groupby(ordered.itertuples(index=False, name=None), key=lambda row: row[1])
    for time, rows in batches:
        batch = list(rows)
        while queue and queue[0][0] < time-window:
            _, c, t = queue.popleft()
            customers[c][t] -= 1; terminals[t][c] -= 1
            if customers[c][t] == 0: del customers[c][t]
            if terminals[t][c] == 0: del terminals[t][c]
            if not customers[c]: del customers[c]
            if not terminals[t]: del terminals[t]
        for row_id, _, c, t, visible in batch:
            features[row_id] = (
                len(customers.get(c, {})), len(terminals.get(t, {})),
                int(t not in customers.get(c, {})),
            )
        for row_id, _, c, t, visible in batch:
            if visible:
                customers[c][t] += 1; terminals[t][c] += 1
                queue.append((time, c, t))
    return pd.DataFrame.from_dict(features, orient="index", columns=RELATION_COLUMNS).reindex(
        pd.Index(events.row_id, name="row_id")).astype(float)


def _checked_handbook_views(data, views, schema=None):
    """A owns the four feature views; no inferred positional pandas alignment."""
    from evaluation import _vector
    make_prediction_frame(data, np.zeros(len(data)), "metadata")
    _vector(data.Amount, "raw Amount", len(data))
    if (data.Amount < 0).any() or not len(data) or set(views) != set(HANDBOOK_VIEWS):
        raise ValueError("Provide nonempty data and exactly the four declared A feature views")
    forbidden = {"class", "row_id", "time", "customer_id", "terminal_id", "dataset_id",
                 "tx_fraud", "tx_fraud_scenario", "isfraud", "isflaggedfraud",
                 "label_available_time", "cluster_id", "method", "score"}
    checked = {}
    for name in HANDBOOK_VIEWS:
        X = views[name]
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Feature views must be DataFrames with explicit row_id index")
        X = X.copy()
        if "row_id" in X: X = X.set_index("row_id")
        if not X.index.is_unique or set(X.index) != set(data.row_id):
            raise ValueError("Feature-view indices must match transaction row_ids")
        if any(not isinstance(c, str) or c.lower() in forbidden or c.lower().startswith("tx_fraud") for c in X.columns):
            raise ValueError("Target, oracle, clock and entity keys cannot be model features")
        if set(RELATION_COLUMNS) & set(X):
            raise ValueError("Base history views must not already contain E12 relation features")
        X = X.loc[data.row_id]
        _features(X)
        if schema is not None:
            if set(X) != set(schema[name]): raise ValueError("Frozen feature columns differ")
            X = X[schema[name]]
        checked[name] = X.astype(float)
    base = set(checked["transaction_only"])
    if "Amount" not in base or not np.array_equal(checked["transaction_only"].Amount.to_numpy(), data.Amount.to_numpy()):
        raise ValueError("Transaction-only Amount must match the raw current transaction")
    if not base <= set(checked["customer_history"]) or not base <= set(checked["terminal_history"]):
        raise ValueError("History views must retain the transaction-only columns")
    if base == set(checked["customer_history"]) or base == set(checked["terminal_history"]):
        raise ValueError("Each history view must add its declared context columns")
    if set(checked["both_history"]) != set(checked["customer_history"]) | set(checked["terminal_history"]):
        raise ValueError("Both-history columns must be the union of customer and terminal views")
    for a, b in itertools.combinations(HANDBOOK_VIEWS, 2):
        common = sorted(set(checked[a]) & set(checked[b]))
        if not np.array_equal(checked[a][common].to_numpy(), checked[b][common].to_numpy()):
            raise ValueError("Shared feature values differ between A views")
    return checked


def fit_handbook_c(train, train_views, relation_features):
    """C-N1/N2: four fixed IF fits plus one graph IF and two fixed LR references.

    The LR references only establish the graph increment; they do not replace
    B-N1's separate delivery. All scalers fit training views; no ULB constants.
    """
    from classification import train_model
    from evaluation import fit_amount_bins
    views = _checked_handbook_views(train, train_views)
    relations = _checked_relations(train, relation_features)
    models, model_views, timings = {}, {}, {}
    extended = {**views, "both_history_graph": pd.concat([views["both_history"], relations], axis=1)}
    if set(train.Class) != {0, 1}:
        raise ValueError("Graph LR references require both training classes; do not resample to conceal this")
    for view in HANDBOOK_VIEWS + ("both_history_graph",):
        name = "if_"+view
        start = perf_counter()
        models[name] = fit_isolation_forest(extended[view], seed=42, n_estimators=100,
                                           max_samples=256, contamination="auto", n_jobs=1)
        timings[name] = perf_counter()-start
        model_views[name] = view
    for view in ["both_history", "both_history_graph"]:
        name = "lr_"+view
        start = perf_counter()
        models[name] = train_model(extended[view], train.Class, "logistic_regression",
                                   strategy="class_weight", seed=42)
        timings[name] = perf_counter()-start
        model_views[name] = view
    return {
        "models": models,
        "config": {"feature_columns": {n:list(x.columns) for n,x in views.items()},
                   "train_view_sha256": _handbook_view_fingerprints(views),
                   "model_views": model_views, "amount_bins": fit_amount_bins(train),
                   "train_partition": partition_description(train),
                   "relation_columns": list(RELATION_COLUMNS), "relation_window_seconds": 7*86400,
                   "IF": {"n_estimators":100,"max_samples":256,"contamination":"auto","seed":42,"n_jobs":1},
                   "LR": {"model":"logistic_regression","strategy":"class_weight","seed":42},
                   "review_fractions": list(DEFAULT_BUDGETS)},
        "fit_seconds": timings,
    }


def _handbook_view_fingerprints(views):
    """Named-column and row-keyed numeric fingerprints under the pinned pandas."""
    result = {}
    for name, frame in views.items():
        h = hashlib.sha256(json.dumps(list(frame.columns)).encode())
        h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="<u8").tobytes())
        result[name] = h.hexdigest()
    return result


def _checked_relations(data, relations):
    if not isinstance(relations, pd.DataFrame) or set(relations) != set(RELATION_COLUMNS):
        raise ValueError("Supply the three declared relation columns")
    if not relations.index.is_unique or not set(data.row_id) <= set(relations.index):
        raise ValueError("Relation features must be keyed by source row_id")
    relations = relations.loc[data.row_id, list(RELATION_COLUMNS)].copy()
    _features(relations)
    degrees = relations[list(RELATION_COLUMNS[:2])].to_numpy()
    if (relations.to_numpy() < 0).any() or (degrees != np.floor(degrees)).any() or not np.isin(relations[RELATION_COLUMNS[2]], [0,1]).all():
        raise ValueError("Relation degrees must be nonnegative and new-edge flag binary")
    return relations.astype(float)


def score_handbook_c(bundle, data, views, relation_features):
    """Frozen C models on a new, A-identified partition; never fit or select."""
    from classification import score_model
    views = _checked_handbook_views(data, views, bundle["config"]["feature_columns"])
    extended = {**views, "both_history_graph": pd.concat(
        [views["both_history"], _checked_relations(data, relation_features)], axis=1)}
    predictions, timings = {}, {}
    for name, model in bundle["models"].items():
        X = extended[bundle["config"]["model_views"][name]]
        start = perf_counter()
        score = score_isolation_forest(model, X) if name.startswith("if_") else score_model(model, X)
        predictions[name] = make_prediction_frame(data, score, name)
        timings[name] = perf_counter()-start
    predictions["amount_descending"] = make_prediction_frame(data, data.Amount, "amount_descending")
    return predictions, timings


def evaluate_history_missing(bundle, transactions, evaluation_data, feature_builder,
                             clean_views, *, rates=(0.0,0.1,0.3), seeds=(42,43,44)):
    """C-N3: callback receives unlabeled raw context and row-keyed visibility mask.

    A supplies feature_builder(unlabeled_transactions, history_keep=Series),
    returning all four views for ALL source IDs. No current row is dropped.
    Models remain trained on full history. Only fixed both-history LR/IF score.
    """
    from classification import score_model
    from evaluation import evaluate_amounts, cost_sensitivity, _vector, _integer
    rates = _vector(rates, "history missing rates")
    if not len(rates) or ((rates < 0)|(rates > 1)).any() or len(np.unique(rates))!=len(rates):
        raise ValueError("Missing rates must be distinct and in [0,1]")
    seeds = [_integer(s,"missing seed",0,2**32-1) for s in seeds]
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError("Seeds must be nonempty and unique")
    events = _relation_events(transactions)
    if "Amount" not in transactions: raise ValueError("History builder needs raw Amount")
    # Physically exclude labels, oracle mechanisms and any extra upstream fields.
    context = events.assign(Amount=transactions.Amount.to_numpy())
    if not set(evaluation_data.row_id) <= set(context.row_id):
        raise ValueError("Evaluation IDs missing from the context stream")
    aligned_context = context.set_index("row_id").loc[evaluation_data.row_id]
    for column in ["Time", "Amount", "CUSTOMER_ID", "TERMINAL_ID"]:
        if not np.array_equal(aligned_context[column].to_numpy(), evaluation_data[column].to_numpy()):
            raise ValueError("Evaluation metadata differs from the context stream")
    clean = _checked_handbook_views(evaluation_data, clean_views, bundle["config"]["feature_columns"])
    all_visible = history_visibility_mask(context, 0.0, seeds[0])
    rebuilt = feature_builder(context.copy(), history_keep=all_visible.copy())
    rebuilt = _checked_handbook_views(context.assign(Class=0), rebuilt, bundle["config"]["feature_columns"])
    rebuilt_eval = {n: x.loc[evaluation_data.row_id] for n,x in rebuilt.items()}
    rebuilt_eval = _checked_handbook_views(evaluation_data,rebuilt_eval,bundle["config"]["feature_columns"])
    for name in HANDBOOK_VIEWS:
        if not np.array_equal(clean[name].to_numpy(),rebuilt_eval[name].to_numpy()):
            raise ValueError("A callback's clean views differ from the delivered feature tables")
    records = []
    for rate in rates:
        for seed in seeds:
            keep = history_visibility_mask(context,float(rate),seed)
            source = rebuilt if rate==0 else feature_builder(context.copy(),history_keep=keep.copy())
            source = _checked_handbook_views(context.assign(Class=0), source, bundle["config"]["feature_columns"])
            views = {n:x.loc[evaluation_data.row_id] for n,x in source.items()}
            views = _checked_handbook_views(evaluation_data,views,bundle["config"]["feature_columns"])
            if not np.array_equal(views["transaction_only"].to_numpy(),clean["transaction_only"].to_numpy()):
                raise ValueError("Missing-history simulation must not change current transaction features")
            metrics = {}
            for name in ["if_both_history","lr_both_history"]:
                model = bundle["models"][name]
                X=views["both_history"]
                scores=score_isolation_forest(model,X) if name.startswith("if_") else score_model(model,X)
                pred=make_prediction_frame(evaluation_data,scores,name)
                metrics[name]=evaluate_amounts(pred,evaluation_data,bundle["config"]["amount_bins"])
            canonical = keep.sort_index()
            digest=hashlib.sha256()
            for row_id,visible in canonical.items():digest.update(f"{int(row_id)}:{int(visible)}\n".encode())
            records.append({"missing_fraction":float(rate),"seed":seed,
                            "source_events":len(context),"visible_events":int(keep.sum()),
                            "actual_visible_fraction":float(keep.mean()),"mask_sha256":digest.hexdigest(),
                            "evaluation_rows":len(evaluation_data),"evaluation_row_ids_sha256":partition_description(evaluation_data)["row_ids_sha256"],
                            "metrics":metrics,"cost_sensitivity":cost_sensitivity(metrics)})
    return {"status":"evaluated","rates":rates.tolist(),"seeds":seeds,"records":records,
            "note":"Inference-only missing context; unchanged current rows/labels and fixed full-history models. Seed ranges are not confidence intervals."}


def run_handbook_c(train, validation, train_views, validation_views, transactions, *,
                   dataset_manifest, output, results=None, feature_builder=None):
    """C new-data validation entry. A supplies official data/views and manifest.

    Synthetic fixtures use a distinct evidence_kind/output namespace and NEVER
    count as official Handbook validation. No dataset is generated here, no
    held-out test rows enter this runner, and no method is selected from results.
    """
    from evaluation import evaluate_amounts, cost_sensitivity, save_handbook_c_figures
    evidence = dataset_manifest.get("evidence_kind")
    if dataset_manifest.get("dataset_id")!="handbook" or evidence not in {"synthetic_fixture","handbook_validation"}:
        raise ValueError("A dataset_id and explicit synthetic/official evidence kind are required")
    if evidence=="handbook_validation":
        if not all(dataset_manifest.get(k) for k in ["data_sha256","generator_source","generation_parameters"]):
            raise ValueError("Official run requires A's data hash, generator source and frozen parameters")
    if train.empty or validation.empty or train.Time.max()>=validation.Time.min() or set(train.row_id)&set(validation.row_id):
        raise ValueError("Training/validation must be nonempty, disjoint and chronological without split-time ties")
    for name,data in [("train",train),("validation",validation)]:
        if partition_description(data)!=dataset_manifest.get("partitions",{}).get(name):
            raise ValueError("Data identities differ from A's declared partition")
    context = _relation_events(transactions)
    if set(context.row_id) != set(train.row_id)|set(validation.row_id):
        raise ValueError("Context must contain exactly train+validation; no test or foreign IDs")
    expected=pd.concat([train,validation]).set_index("row_id")
    ordered=transactions.set_index("row_id").loc[expected.index]
    for column in ["Time","Amount","CUSTOMER_ID","TERMINAL_ID"]:
        if not np.array_equal(ordered[column].to_numpy(),expected[column].to_numpy()):
            raise ValueError("Context metadata differs from the declared partitions")
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError("Handbook C output must be new or empty")
    prefix="handbook_c_interface" if evidence=="synthetic_fixture" else "handbook_c"
    if results is not None:
        results=Path(results)
        a,b=output.resolve(),results.resolve()
        if a==b or a in b.parents or b in a.parents:raise ValueError("Private/public outputs must not overlap")
        targets=[results/"tables"/(prefix+".json")]+[results/"figures"/(prefix+"_"+n+".png") for n in ["ablation","missing"]]
        if any(p.exists() for p in targets):raise ValueError("Published Handbook C evidence already exists")
    # All identity, chronology and path checks precede model fitting.
    train_views=_checked_handbook_views(train,train_views)
    validation_views=_checked_handbook_views(validation,validation_views,
                                            {n:list(x.columns) for n,x in train_views.items()})
    relations=build_relation_features(context)
    bundle=fit_handbook_c(train,train_views,relations)
    predictions,score_times=score_handbook_c(bundle,validation,validation_views,relations)
    metrics={name:evaluate_amounts(p,validation,bundle["config"]["amount_bins"]) for name,p in predictions.items()}
    missing=(evaluate_history_missing(bundle,transactions,validation,feature_builder,validation_views)
             if feature_builder is not None else {"status":"pending_A_history_builder"})
    output.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,output/"models.joblib")
    restored=joblib.load(output/"models.joblib")
    replay,_=score_handbook_c(restored,validation,validation_views,relations)
    for name in predictions:
        np.testing.assert_array_equal(predictions[name].score.to_numpy(),replay[name].score.to_numpy())
        predictions[name].to_csv(output/(name+"_predictions.csv"),index=False,float_format="%.17g")
    result={"dataset_id":"handbook","phase":"validation","evidence_kind":evidence,
            "official_validation_complete":evidence=="handbook_validation" and missing["status"]=="evaluated",
            "group_G2_complete":False,"test_scored":False,"configuration_reselected":False,
            "dataset_manifest":copy.deepcopy(dataset_manifest),"config":bundle["config"],
            "validation_view_sha256":_handbook_view_fingerprints(validation_views),
            "metrics":metrics,"cost_sensitivity":cost_sensitivity(metrics),"history_missing":missing,
            "fit_seconds":bundle["fit_seconds"],"score_seconds":score_times,
            "verification":{"saved_model_scores_replay_exactly":True},
            "model_bundle_sha256":data_sha256(output/"models.joblib"),
            "provenance":_source()}
    _dump(result,output/"summary.json")
    if results is not None:
        paths=save_handbook_c_figures(result,results/"figures")
        result["figure_sha256"]={p.name:data_sha256(p) for p in paths}
        (results/"tables").mkdir(parents=True,exist_ok=True)
        _dump(result,results/"tables"/(prefix+".json"))
    return result,bundle
