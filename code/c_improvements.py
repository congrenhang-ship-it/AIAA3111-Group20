"""负责人 C：第二轮有界消融、训练内时间选择与可恢复的改进模型。

不改变 A 的正式划分或第一轮代码。特征/缩放/距离标定仅 fit 当前训练段。
内层选择使用标签，必须披露；冻结后的评分和评价不再选择候选或配额。
"""

from pathlib import Path
import copy
import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.validation import check_is_fitted
from preprocessing import FEATURE_COLUMNS
from evaluation import (
    make_prediction_frame,
    evaluate_scores,
    compare_review_lists,
    evaluate_groups,
    DEFAULT_BUDGETS,
    save_figures,
    save_group_figures,
    save_comparison_figures,
)
from c_reporting import partition_description

SEEDS = (42, 43, 44)
SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)


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


def evaluate_improved_frozen(bundle, data, output, *, classifier=None, phase="test"):
    """A8 evaluation: no candidate/weight selection or model fitting occurs here."""
    import json
    from temporal import evaluate_time_windows

    if phase not in {"test", "validation_replay"}:
        raise ValueError("Invalid evaluation phase")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    config = copy.deepcopy(bundle["config"])
    b = config["B_combination"]
    if classifier is not None and b["share"] is None:
        raise ValueError("B combination must be selected and frozen before test")
    predictions, groups = score_improved_bundle(bundle, data)
    comparisons = pair_comparisons(
        predictions, config["selected_candidates"], config["distance_share"]
    )
    b_rows = []
    if classifier is not None:
        if classifier.empty or not classifier.method.eq(b["method"]).all():
            raise ValueError("Classifier does not match frozen method")
        c = make_prediction_frame(classifier, classifier.score, b["method"])
        anomaly = predictions[config["selected_anomaly"]]
        compare_review_lists(c, anomaly, 1)
        c = c.set_index("row_id").loc[data.row_id].reset_index()
        predictions[b["method"]] = c
        for q in DEFAULT_BUDGETS:
            row = compare_review_lists(c, anomaly, math.ceil(len(data) * q), b["share"])
            row["review_fraction"] = q
            b_rows.append(row)
    metrics = {
        n: evaluate_scores(p.Class, p.score, row_ids=p.row_id)
        for n, p in predictions.items()
    }
    group_metrics = {
        n: evaluate_groups(p, groups, DEFAULT_BUDGETS, group_values=range(3))
        for n, p in predictions.items()
    }
    windows = evaluate_time_windows(pd.concat(predictions.values(), ignore_index=True))
    for w in windows:
        if w["evaluation"] is not None:
            w["evaluation"].pop("pr_curve", None)
    output.mkdir(parents=True, exist_ok=True)
    label = (
        "Final test - frozen improved C"
        if phase == "test"
        else "Validation replay - frozen improved C"
    )
    save_figures({"methods": metrics, "label": label}, output / "figures")
    save_group_figures(
        group_metrics[config["selected_anomaly"]],
        output / "figures",
        label + "\n" + config["selected_anomaly"],
    )
    save_comparison_figures(
        comparisons,
        output / "figures",
        label + "\nFrozen anomaly pair",
        method_labels={
            "classifier": config["selected_candidates"]["kmeans"],
            "anomaly": config["selected_candidates"]["isolation_forest"],
            "combined": "Frozen pair",
        },
    )
    if b_rows:
        save_comparison_figures(
            b_rows, output / "B_figures", label + "\nFrozen B combination"
        )
    for name, p in predictions.items():
        p.to_csv(output / f"{name}_predictions.csv", index=False)
    result = {
        "phase": phase,
        "config": config,
        "configuration_reselected": False,
        "partition": partition_description(data),
        "metrics": metrics,
        "groups": group_metrics,
        "time_windows": windows,
        "comparisons": comparisons,
        "B_comparisons": b_rows,
    }
    (output / "evaluation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result
