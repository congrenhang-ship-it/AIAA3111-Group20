"""负责人 C：统一预测表、公共评价、等预算组合、群体分析和图表。

已实现；高分表示更可疑。K=ceil(N*q)，按 score 降序、row_id 升序处理并列。
零正类的 AP/Recall 返回 None；空组明确标记。所有阈值/配额须在验证阶段确定。
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

PREDICTION_COLUMNS = ("row_id", "Class", "Time", "method", "score")
DEFAULT_BUDGETS = (0.001, 0.005, 0.01, 0.02)


def _vector(values, name, n=None):
    try:
        a = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if a.ndim != 1 or (n is not None and len(a) != n) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite 1-D array of matching length")
    return a


def _row_ids(values, n):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError("row_ids must be numeric integers")
    a = _vector(values, "row_ids", n)
    if (a < 0).any() or (a >= 2**53).any() or (a != np.floor(a)).any():
        raise ValueError("row_ids must be nonnegative exact integers below 2**53")
    ids = a.astype(np.int64)
    if len(np.unique(ids)) != n:
        raise ValueError("row_ids must be unique")
    return ids


def _labels(values, n=None):
    y = _vector(values, "Class", n)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Class must contain only 0/1")
    return y.astype(np.int64)


def _budgets(values):
    q = _vector(values, "review_fractions")
    if not len(q) or ((q <= 0) | (q > 1)).any() or len(set(q)) != len(q):
        raise ValueError("review_fractions must be distinct values in (0, 1]")
    return q


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be an integer")
    try:
        valid = np.isfinite(value) and int(value) == value and low <= value <= high
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _single_method(predictions):
    if not isinstance(predictions, pd.DataFrame):
        raise ValueError("predictions must be a DataFrame")
    missing = set(PREDICTION_COLUMNS) - set(predictions)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    methods = predictions["method"]
    if (
        methods.isna().any()
        or not methods.map(lambda x: isinstance(x, str) and bool(x.strip())).all()
    ):
        raise ValueError("method must contain nonempty strings")
    if len(methods.unique()) > 1:
        raise ValueError("Pass exactly one method at a time")
    return make_prediction_frame(
        predictions, predictions["score"], methods.iloc[0] if len(methods) else "empty"
    )


def make_prediction_frame(data, scores, method, cluster_ids=None):
    """Create a fresh positional table; pandas indexes never realign score arrays."""
    if not isinstance(data, pd.DataFrame) or not {"row_id", "Class", "Time"} <= set(
        data
    ):
        raise ValueError("data must contain row_id, Class and Time")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a nonempty string")
    n = len(data)
    frame = pd.DataFrame(
        {
            "row_id": _row_ids(data["row_id"], n),
            "Class": _labels(data["Class"], n),
            "Time": _vector(data["Time"], "Time", n),
            "method": method,
            "score": _vector(scores, "scores", n),
        }
    )
    if (frame.Time < 0).any():
        raise ValueError("Time must be nonnegative")
    if cluster_ids is not None:
        groups = np.asarray(cluster_ids)
        if groups.ndim != 1 or len(groups) != n or pd.isna(groups).any():
            raise ValueError(
                "cluster_ids must have matching length and no missing values"
            )
        frame["cluster_id"] = groups
    return frame


def _order(scores, ids):
    return np.lexsort((ids, -scores))


def _counts(y, selected):
    k = len(selected)
    tp = int(y[selected].sum())
    positives = int(y.sum())
    return {
        "K": k,
        "TP": tp,
        "FP": k - tp,
        "FN": positives - tp,
        "TN": int(len(y) - positives - (k - tp)),
        "precision_at_k": tp / k if k else None,
        "recall_at_k": tp / positives if positives else None,
    }


def evaluate_scores(
    y_true, scores, review_fractions=DEFAULT_BUDGETS, *, row_ids, threshold=None
):
    """Return JSON-safe AP, PR points, expected random review and budget counts.

    Random baseline values are expectations, not measured or rounded counts.
    AP uses sklearn's noninterpolated average_precision_score; no trapezoid AUC.
    A supplied threshold is caller-selected on validation; predictions use >=.
    """
    y = _labels(y_true)
    n = len(y)
    score = _vector(scores, "scores", n)
    ids = _row_ids(row_ids, n)
    qs = _budgets(review_fractions)
    if threshold is not None:
        threshold = float(_vector([threshold], "threshold")[0])
    positives = int(y.sum())
    prevalence = positives / n if n else None
    result = {
        "status": "empty" if not n else ("no_positives" if not positives else "ok"),
        "n_samples": n,
        "n_positive": positives,
        "prevalence": prevalence,
        "AP": None,
        "pr_curve": None,
        "budgets": [],
        "threshold": None,
        "ranking_rule": "score descending, row_id ascending; K=ceil(N*q)",
    }
    if positives:
        result["AP"] = float(average_precision_score(y, score))
        precision, recall, thresholds = precision_recall_curve(y, score)
        result["pr_curve"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        }
    order = _order(score, ids)
    for q in qs:
        k = min(n, math.ceil(n * float(q)))
        row = {
            "review_fraction": float(q),
            "actual_review_fraction": k / n if n else None,
            **_counts(y, order[:k]),
        }
        row["random_expected_TP"] = k * prevalence if n else None
        row["random_expected_precision"] = prevalence if k else None
        row["random_expected_recall"] = k / n if positives else None
        result["budgets"].append(row)
    if threshold is not None:
        selected = np.flatnonzero(score >= threshold)
        result["threshold"] = {"value": threshold, **_counts(y, selected)}
    return result


def compare_review_lists(
    classification_predictions, anomaly_predictions, k, classifier_share=0.5
):
    """Compare top-K lists and a fixed-quota combination with exactly K unique IDs.

    classifier quota = floor(K*share). Take classifier quota first, then anomaly
    quota skipping duplicates; scan anomaly ordering to fill remaining slots.
    Endpoint shares 0/1 reproduce the pure rankings. All selection ignores labels.
    Input table order may differ; ID, label and Time must agree exactly.
    """
    c = _single_method(classification_predictions)
    a = _single_method(anomaly_predictions)
    n = len(c)
    k = _integer(k, "k", 1, n)
    share = float(_vector([classifier_share], "classifier_share")[0])
    if not 0 <= share <= 1:
        raise ValueError("classifier_share must be in [0, 1]")
    if set(c.row_id) != set(a.row_id):
        raise ValueError("Prediction tables must have identical row IDs")
    a = a.set_index("row_id").loc[c.row_id].reset_index()
    if not np.array_equal(c.Class, a.Class) or not np.array_equal(c.Time, a.Time):
        raise ValueError("Labels/Time disagree for matching row IDs")
    y, ids = c.Class.to_numpy(), c.row_id.to_numpy()
    co, ao = _order(c.score.to_numpy(), ids), _order(a.score.to_numpy(), ids)
    cq = math.floor(k * share)
    selected = list(co[:cq])
    used = set(selected)
    for index in ao:
        if len(selected) == k:
            break
        if index not in used:
            selected.append(int(index))
            used.add(int(index))
    cp, ap = set(co[:k]), set(ao[:k])
    shared, conly, aonly = cp & ap, cp - ap, ap - cp
    frauds = lambda positions: int(y[list(positions)].sum()) if positions else 0
    return {
        "K": k,
        "classifier_share": share,
        "classifier_quota": cq,
        "anomaly_quota": k - cq,
        "refill_rule": "classifier first; anomaly scan skipping duplicates",
        "overlap_count": len(shared),
        "shared_frauds": frauds(shared),
        "classifier_only_frauds": frauds(conly),
        "anomaly_only_frauds": frauds(aonly),
        "classifier": _counts(y, co[:k]),
        "anomaly": _counts(y, ao[:k]),
        "combined": _counts(y, np.array(selected, dtype=int)),
        "classifier_ids": ids[co[:k]].tolist(),
        "anomaly_ids": ids[ao[:k]].tolist(),
        "combined_ids": ids[selected].tolist(),
    }


def evaluate_groups(predictions, cluster_ids, review_fractions, threshold=None):
    """Return within-group metrics AND the distribution of global top-K reviews.

    cluster_ids: aligned vector or Series keyed by row_id (not pandas row index).
    Within-group ceil quotas may sum to more than a single global budget.
    """
    frame = _single_method(predictions)
    if isinstance(cluster_ids, pd.Series):
        if not cluster_ids.index.is_unique or set(cluster_ids.index) != set(
            frame.row_id
        ):
            raise ValueError("Cluster Series index must match row_id exactly")
        groups = cluster_ids.loc[frame.row_id].to_numpy()
    else:
        groups = np.asarray(cluster_ids)
    if groups.ndim != 1 or len(groups) != len(frame) or pd.isna(groups).any():
        raise ValueError("cluster_ids must align without missing groups")
    qs = _budgets(review_fractions)
    global_order = _order(frame.score.to_numpy(), frame.row_id.to_numpy())
    records = []
    total_positive = int(frame.Class.sum())
    for group in pd.unique(groups):
        mask = groups == group
        positions = np.flatnonzero(mask)
        sub = frame.iloc[positions]
        metrics = evaluate_scores(
            sub.Class, sub.score, qs, row_ids=sub.row_id, threshold=threshold
        )
        metrics["pr_curve"] = None  # subgroup curves need not inflate result files
        allocations = []
        for q in qs:
            selected = global_order[: math.ceil(len(frame) * float(q))]
            selected = selected[mask[selected]]
            tp = int(frame.Class.iloc[selected].sum())
            allocations.append(
                {
                    "review_fraction": float(q),
                    "reviewed_in_group": len(selected),
                    "frauds_detected_in_group": tp,
                    "recall_within_group": tp / int(sub.Class.sum())
                    if sub.Class.sum()
                    else None,
                }
            )
        records.append(
            {
                "cluster_id": group.item() if isinstance(group, np.generic) else group,
                "fraud_coverage": int(sub.Class.sum()) / total_positive
                if total_positive
                else None,
                "within_group": metrics,
                "global_budget_allocation": allocations,
            }
        )
    return {
        "budget_note": "within-group quotas and global top-K allocations are distinct",
        "groups": records,
    }


def save_figures(results, output_dir):
    """Save plots for a mapping {method: evaluate_scores result} without fitting.

    results may wrap the mapping as {'methods': ..., 'label': 'validation / ...'}.
    Labels must identify synthetic demonstrations. Return paths of saved PNGs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    methods = results.get("methods", results)
    label = results.get("label", "Evaluation")
    paths = []
    for filename, ylabel, mode in [
        ("pr_curve", "Precision", "pr"),
        ("review_recall", "Recall@K", "recall_at_k"),
        ("review_precision", "Precision@K", "precision_at_k"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False
        for method, metrics in methods.items():
            if mode == "pr":
                curve = metrics.get("pr_curve")
                if curve:
                    ax.step(
                        curve["recall"],
                        curve["precision"],
                        where="post",
                        label=f"{method} (AP={metrics['AP']:.3f})",
                    )
                    plotted = True
            else:
                rows = [r for r in metrics["budgets"] if r[mode] is not None]
                if rows:
                    ax.plot(
                        [100 * r["actual_review_fraction"] for r in rows],
                        [r[mode] for r in rows],
                        marker="o",
                        label=method,
                    )
                    plotted = True
        if mode == "pr":
            prev = {
                m["prevalence"] for m in methods.values() if m["prevalence"] is not None
            }
            if len(prev) == 1:
                ax.axhline(
                    next(iter(prev)),
                    color="gray",
                    linestyle="--",
                    label="Random reference",
                )
            ax.set_xlabel("Recall")
        else:
            ax.set_xlabel("Actual review proportion (%)")
        if not plotted:
            ax.text(0.5, 0.5, "No defined values", transform=ax.transAxes, ha="center")
        if plotted:
            ax.legend(fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = output / f"{filename}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def save_group_figures(group_results, output_dir, label="Group evaluation"):
    """Plot group prevalence and recall under each global budget; undefined recall is a gap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = group_results["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = [str(r["cluster_id"]) for r in rows]
    axes[0].bar(names, [r["within_group"]["prevalence"] for r in rows])
    for i, row in enumerate(rows):
        axes[0].annotate(
            f"n={row['within_group']['n_samples']}",
            (i, row["within_group"]["prevalence"]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[0].set(xlabel="Cluster", ylabel="Fraud proportion")
    if rows:
        for i, budget in enumerate(rows[0]["global_budget_allocation"]):
            values = [
                r["global_budget_allocation"][i]["recall_within_group"] for r in rows
            ]
            axes[1].plot(
                names,
                [np.nan if v is None else v for v in values],
                marker="o",
                label=f"Global budget {100 * budget['review_fraction']:g}%",
            )
        axes[1].legend(fontsize=8)
    axes[1].set_xticks(np.arange(len(names)), names)
    if rows:
        axes[1].set_xlim(-0.5, len(rows) - 0.5)
        for i, row in enumerate(rows):
            if not row["within_group"]["n_positive"]:
                axes[1].text(i, 0.04, "N/A\n(no fraud)", ha="center", fontsize=8)
    axes[1].set(
        xlabel="Cluster", ylabel="Group recall from global top-K", ylim=(-0.02, 1.08)
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(label)
    fig.tight_layout()
    path = output / "cluster_prevalence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [str(path)]


def save_comparison_figures(comparisons, output_dir, label="Equal-budget comparison"):
    """Plot detected fraud counts; all three bars in a row share the same K."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(comparisons))
    for offset, key in enumerate(["classifier", "anomaly", "combined"]):
        ax.bar(
            x + (offset - 1) * 0.25,
            [c[key]["TP"] for c in comparisons],
            width=0.25,
            label=key,
        )
    ax.set_xticks(x, [str(c["K"]) for c in comparisons])
    ax.set(xlabel="Review count K", ylabel="Fraud detected", title=label)
    if comparisons:
        ax.legend()
    fig.tight_layout()
    path = output / "complementarity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [str(path)]
