"""负责人 C：公共评价、名单组合、图表与结果导出；A/B/D 复用本模块。

仅整理模块与输出路径；所有训练、预算、并列和无正类处理口径保持不变。
冻结评价不会选择参数；最终测试由 A 在全组冻结后调用。
"""

import math
import numpy as np
import pandas as pd
import copy
import hashlib
import json
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve

PREDICTION_COLUMNS = ("row_id", "Class", "Time", "method", "score")
DEFAULT_BUDGETS = (0.001, 0.005, 0.01, 0.02)
PRIMARY_REVIEW_FRACTION = 0.01
COMBINATION_SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)


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
    _vector(data["Time"], "Time", n)
    frame = pd.DataFrame(
        {
            "row_id": _row_ids(data["row_id"], n),
            "Class": _labels(data["Class"], n),
            "Time": (
                _vector(data["Time"], "Time", n)
                if data["Time"].dtype.kind not in "iuf"
                else data["Time"].to_numpy()
            ),
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


def evaluate_groups(
    predictions, cluster_ids, review_fractions, threshold=None, *, group_values=None
):
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
    values = list(pd.unique(groups)) if group_values is None else list(group_values)
    if (
        len(set(values)) != len(values)
        or pd.isna(values).any()
        or not set(groups) <= set(values)
    ):
        raise ValueError(
            "group_values must be distinct, nonmissing and cover observed groups"
        )
    for group in values:
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
        defined_ap = [m["AP"] for m in methods.values() if m.get("AP") is not None]
        if mode == "pr" and defined_ap and max(defined_ap) < 0.1:
            zoom = ax.inset_axes([0.28, 0.28, 0.68, 0.40])
            for metrics in methods.values():
                curve = metrics.get("pr_curve")
                if curve:
                    zoom.step(curve["recall"], curve["precision"], where="post")
            if len(prev) == 1:
                zoom.axhline(next(iter(prev)), color="gray", linestyle="--")
            zoom.set(xlim=(0, 1), ylim=(0, max(0.02, 3 * max(defined_ap))))
            zoom.set_title("Detail of the same PR curves", fontsize=8)
            zoom.tick_params(labelsize=8)
            zoom.grid(alpha=0.2)
        if mode == "precision_at_k":
            values = [
                row[mode]
                for metrics in methods.values()
                for row in metrics["budgets"]
                if row[mode] is not None
            ]
            if values and max(values) < 0.1:
                ax.set_ylim(0, max(0.05, 1.2 * max(values)))
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = output / f"{filename}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def save_group_figures(
    group_results, output_dir, label="Group evaluation", *, comparison_groups=None
):
    """Plot group prevalence and recall under each global budget; undefined recall is a gap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = group_results["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = [str(r["cluster_id"]) for r in rows]
    axes[0].bar(
        names,
        [
            np.nan
            if r["within_group"]["prevalence"] is None
            else r["within_group"]["prevalence"]
            for r in rows
        ],
    )
    for i, row in enumerate(rows):
        axes[0].annotate(
            f"n={row['within_group']['n_samples']}"
            + (" N/A" if row["within_group"]["prevalence"] is None else ""),
            (i, row["within_group"]["prevalence"] or 0),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axes[0].set(xlabel="Cluster", ylabel="Fraud proportion")
    axes[0].set_xticks(np.arange(len(names)), names)
    if rows:
        axes[0].set_xlim(-0.5, len(rows) - 0.5)
    if comparison_groups:
        width = 0.8 / len(comparison_groups)
        x = np.arange(len(rows))
        for index, (method, result) in enumerate(comparison_groups.items()):
            by_id = {str(g["cluster_id"]): g for g in result["groups"]}
            values = []
            for name in names:
                row = next(
                    r
                    for r in by_id[name]["global_budget_allocation"]
                    if r["review_fraction"] == 0.01
                )
                values.append(
                    np.nan
                    if row["recall_within_group"] is None
                    else row["recall_within_group"]
                )
            axes[1].bar(
                x + (index - (len(comparison_groups) - 1) / 2) * width,
                values,
                width=width,
                label=method,
            )
        axes[1].set_title("Global 1% review", fontsize=10)
        axes[1].legend(fontsize=7)
    else:
        if rows:
            for i, budget in enumerate(rows[0]["global_budget_allocation"]):
                values = [
                    r["global_budget_allocation"][i]["recall_within_group"]
                    for r in rows
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


def save_comparison_figures(
    comparisons,
    output_dir,
    label="Equal-budget comparison",
    *,
    method_labels=None,
    filename="complementarity.png",
):
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
            label=(method_labels or {}).get(key, key),
        )
    ax.set_xticks(x, [str(c["K"]) for c in comparisons])
    ax.set(xlabel="Review count K", ylabel="Fraud detected", title=label)
    if comparisons:
        ax.legend()
    fig.tight_layout()
    path = output / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [str(path)]


def row_fingerprint(frame):
    """A's ordered row identity fingerprint, without publishing individual IDs."""
    digest = hashlib.sha256()
    for value in frame.row_id:
        digest.update(f"{int(value)}\n".encode("ascii"))
    return digest.hexdigest()


def partition_description(frame):
    return {
        "rows": len(frame),
        "frauds": int(frame.Class.sum()),
        "time_min": float(frame.Time.min()),
        "time_max": float(frame.Time.max()),
        "row_ids_sha256": row_fingerprint(frame),
    }


def choose_anomaly(metrics, primary_fraction=PRIMARY_REVIEW_FRACTION):
    """Validation TP at common primary budget, then AP, then stable method name."""
    names = ["isolation_forest", "kmeans_distance"]

    def key(name):
        metric = metrics[name]
        row = next(
            r for r in metric["budgets"] if r["review_fraction"] == primary_fraction
        )
        return (-row["TP"], -(metric["AP"] if metric["AP"] is not None else -1), name)

    return min(names, key=key)


def select_combination(comparisons, primary_fraction=PRIMARY_REVIEW_FRACTION):
    """Max validation TP; ties keep more classifier capacity (pure baseline allowed)."""
    rows = [r for r in comparisons if r["review_fraction"] == primary_fraction]
    if not rows:
        return None
    return min(rows, key=lambda r: (-r["combined"]["TP"], -r["classifier_share"]))


def build_freeze(report):
    """Freeze C's settings only; this document never asserts whole-group G2."""
    metadata = report["metadata"]
    if metadata["mode"] == "synthetic_demo":
        status = "synthetic_only_not_for_test"
    elif not metadata.get("verified_ulb_split"):
        status = "unverified_partitions_not_for_official_test"
    else:
        status = "unsupervised_frozen"
    chosen = choose_anomaly(report["metrics"])
    combo = select_combination(report["comparisons"])
    combination = {
        "status": "waiting_for_B5",
        "classifier_method": None,
        "anomaly_method": chosen,
        "classifier_share": None,
    }
    if combo is not None:
        combination.update(
            status="validation_selected_requires_B5_provenance",
            classifier_method=metadata["classifier_method"],
            classifier_share=combo["classifier_share"],
            validation_TP=combo["combined"]["TP"],
            K=combo["K"],
        )
    return {
        "schema_version": 1,
        "owner": "C",
        "status": status,
        "group_G2_complete": False,
        "dataset_sha256": metadata.get("dataset_sha256"),
        "partitions": metadata["partitions"],
        "feature_columns": metadata["feature_columns"],
        "seed": metadata["seed"],
        "scaler": metadata["scaler"],
        "n_clusters": metadata["selected_k"],
        "kmeans_n_init": metadata["kmeans_n_init"],
        "k_selection": metadata["k_selection"],
        "forest_params": metadata["forest_params"],
        "forest_score": metadata["forest_score"],
        "primary_review_fraction": PRIMARY_REVIEW_FRACTION,
        "review_fractions": metadata["review_fractions"],
        "selected_anomaly_method": chosen,
        "anomaly_selection": "validation TP at 1%; then AP; then method name",
        "combination": combination,
        "combination_selection": "validation TP at 1%; ties prefer larger classifier share",
        "combination_candidates": list(COMBINATION_SHARES),
        "ranking": "score descending; row_id ascending; K=ceil(N*q)",
        "refill": "floor(K*classifier_share) classifier first; scan anomaly skipping duplicates to K",
        "refit_policy": "retain training-only fitted pipelines; no train+validation refit",
        "time_window_diagnostic": {
            "window_seconds": 3600,
            "origin": 0,
            "note": "validation diagnostic using A interface; final windows owned by A",
        },
        "source_commit": metadata["source_commit"],
        "code_sha256": metadata["code_sha256"],
        "versions": metadata["versions"],
        "test_evaluated": False,
    }


def compact_report(report):
    """Strip per-transaction lists and long PR point arrays from public summaries."""
    public = copy.deepcopy(report)
    for metric in public["metrics"].values():
        metric.pop("pr_curve", None)
    for rows in [public.get("comparisons", []), public.get("anomaly_comparisons", [])]:
        for row in rows:
            for key in ["classifier_ids", "anomaly_ids", "combined_ids"]:
                row.pop(key, None)
    public["metadata"]["input_sha256"] = {
        Path(path).name: value
        for path, value in public["metadata"].get("input_sha256", {}).items()
    }
    return public


def write_summary(report, output):
    """Publish aggregate results; fail instead of overwriting earlier evidence."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Public summary folder must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    public = compact_report(report)
    (output / "summary.json").write_text(
        json.dumps(public, indent=2, allow_nan=False) + "\n"
    )
    (output / "c_freeze.json").write_text(
        json.dumps(build_freeze(report), indent=2, allow_nan=False) + "\n"
    )
    budgets = [
        {"method": name, **row}
        for name, m in public["metrics"].items()
        for row in m["budgets"]
    ]
    pd.DataFrame(budgets).to_csv(output / "budgets.csv", index=False)
    groups = []
    for name, data in public["groups"].items():
        for group in data["groups"]:
            base = {
                "method": name,
                "cluster_id": group["cluster_id"],
                "n_samples": group["within_group"]["n_samples"],
                "n_positive": group["within_group"]["n_positive"],
                "prevalence": group["within_group"]["prevalence"],
                "fraud_coverage": group["fraud_coverage"],
                "AP": group["within_group"]["AP"],
            }
            for local, global_ in zip(
                group["within_group"]["budgets"],
                group["global_budget_allocation"],
                strict=True,
            ):
                groups.append(
                    {
                        **base,
                        "review_fraction": local["review_fraction"],
                        "within_group_K": local["K"],
                        "within_group_TP": local["TP"],
                        "within_group_recall": local["recall_at_k"],
                        "global_reviews_in_group": global_["reviewed_in_group"],
                        "global_TP_in_group": global_["frauds_detected_in_group"],
                        "global_recall_in_group": global_["recall_within_group"],
                    }
                )
    pd.DataFrame(groups).to_csv(output / "groups.csv", index=False)
    for field in ["comparisons", "anomaly_comparisons"]:
        rows = []
        for item in public.get(field, []):
            row = {
                key: item[key]
                for key in [
                    "review_fraction",
                    "K",
                    "classifier_share",
                    "shared_frauds",
                    "classifier_only_frauds",
                    "anomaly_only_frauds",
                    "overlap_count",
                ]
            }
            row.update(
                {
                    f"{name}_TP": item[name]["TP"]
                    for name in ["classifier", "anomaly", "combined"]
                }
            )
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(output / f"{field}.csv", index=False)
    return public


def save_diagnostic_figures(candidates, groups, output, label):
    """Structural quality/stability and cross-method global-budget group recall."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(candidates)
    table.to_csv(output.parent / "kmeans_training_structure.csv", index=False)
    means = table.groupby("k").sample_silhouette.agg(["mean", "min", "max"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(
        means.index,
        means["mean"],
        yerr=[means["mean"] - means["min"], means["max"] - means["mean"]],
        marker="o",
        capsize=4,
    )
    axes[0].set(xlabel="Training cluster count k", ylabel="Training sample silhouette")
    ari = table.groupby("k").seed_stability_ARI.first()
    axes[1].bar([str(x) for x in ari.index], ari.values)
    axes[1].set(
        xlabel="Training cluster count k", ylabel="Between-seed ARI", ylim=(-0.05, 1.05)
    )
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(output / "cluster_structure.png", dpi=160)
    plt.close(fig)
    names = list(groups)
    first = groups[names[0]]["groups"]
    matrix = []
    for name in names:
        values = []
        for g in groups[name]["groups"]:
            value = next(
                row["recall_within_group"]
                for row in g["global_budget_allocation"]
                if row["review_fraction"] == PRIMARY_REVIEW_FRACTION
            )
            values.append(np.nan if value is None else value)
        matrix.append(values)
    fig, ax = plt.subplots(figsize=(max(6, len(first) * 1.1), max(3, len(names) * 0.7)))
    ax.imshow(np.asarray(matrix), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(
        range(len(first)),
        [
            f"{g['cluster_id']}\nn={g['within_group']['n_samples']}, fraud={g['within_group']['n_positive']}"
            for g in first
        ],
        fontsize=8,
    )
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                "N/A" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
            )
    ax.set(
        xlabel="Cluster (no fraud: N/A)",
        title=label + "\nGroup recall under global 1% review",
    )
    fig.tight_layout()
    fig.savefig(output / "group_performance.png", dpi=160)
    plt.close(fig)
    return [
        str(output / "cluster_structure.png"),
        str(output / "group_performance.png"),
    ]


def evaluate_frozen_c(bundle, data, output, classifier=None, phase="test"):
    """A8 hook: score existing models and apply existing quotas, without selecting.

    A calls this after whole-group G2. It neither approves G2 nor refits models.
    Output is a private run folder (predictions contain row IDs). If a classifier
    is supplied, its name and quota must already be present in bundle['config'].
    Use phase='validation_replay' to check persistence before any final test.
    """
    from evaluation import (
        make_prediction_frame,
        evaluate_scores,
        evaluate_groups,
        compare_review_lists,
        save_figures,
        save_group_figures,
        save_comparison_figures,
    )
    from unsupervised import score_frozen_unsupervised

    if phase not in {"test", "validation_replay"}:
        raise ValueError("phase must identify test or validation_replay")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Frozen evaluation output must be new or empty")
    config = copy.deepcopy(bundle["config"])
    combination = config["combination"]
    if classifier is not None and (
        combination["classifier_share"] is None
        or combination["classifier_method"] is None
    ):
        raise ValueError("Classifier combination must be frozen on validation first")
    scored = score_frozen_unsupervised(bundle, data[config["feature_columns"]])
    groups = scored.pop("cluster_ids")
    predictions = {
        name: make_prediction_frame(data, score, name, groups)
        for name, score in scored.items()
    }
    comparisons = []
    if classifier is not None:
        name = combination["classifier_method"]
        if not classifier["method"].eq(name).all() or classifier.empty:
            raise ValueError("Classifier method does not match frozen combination")
        classifier = make_prediction_frame(classifier, classifier.score, name)
        compare_review_lists(classifier, predictions[combination["anomaly_method"]], 1)
        classifier = classifier.set_index("row_id").loc[data.row_id].reset_index()
        predictions[name] = classifier
        for q in config["review_fractions"]:
            row = compare_review_lists(
                classifier,
                predictions[combination["anomaly_method"]],
                math.ceil(len(data) * q),
                combination["classifier_share"],
            )
            row["review_fraction"] = q
            comparisons.append(row)
    metrics = {
        name: evaluate_scores(
            frame.Class, frame.score, config["review_fractions"], row_ids=frame.row_id
        )
        for name, frame in predictions.items()
    }
    group_metrics = {
        name: evaluate_groups(
            frame,
            groups,
            config["review_fractions"],
            group_values=range(config["n_clusters"]),
        )
        for name, frame in predictions.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    label = (
        "Final test - frozen C configuration"
        if phase == "test"
        else "Validation replay - frozen C configuration"
    )
    save_figures({"methods": metrics, "label": label}, output / "figures")
    save_group_figures(
        group_metrics[config["selected_anomaly_method"]], output / "figures", label
    )
    if comparisons:
        save_comparison_figures(
            comparisons,
            output / "figures",
            label,
            method_labels={
                "classifier": combination["classifier_method"],
                "anomaly": combination["anomaly_method"],
                "combined": "Frozen combination",
            },
        )
    for name, frame in predictions.items():
        frame.to_csv(output / f"{name}_predictions.csv", index=False)
    result = {
        "phase": phase,
        "config": config,
        "metrics": metrics,
        "groups": group_metrics,
        "comparisons": comparisons,
        "configuration_reselected": False,
        "partition": partition_description(data),
    }
    (output / "frozen_evaluation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def evaluate_improved_frozen(bundle, data, output, *, classifier=None, phase="test"):
    """A8 evaluation: no candidate/weight selection or model fitting occurs here."""
    import json
    from unsupervised import score_improved_bundle, pair_comparisons
    from temporal import evaluate_time_windows

    if phase not in {"test", "validation_replay"}:
        raise ValueError("Invalid evaluation phase")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    config = copy.deepcopy(bundle["config"])
    b = config["B_combination"]
    if classifier is None and bundle.get("B_model") is not None:
        if b["share"] is None:
            raise ValueError("Persisted B model requires an already frozen quota")
        from classification import score_model
        from preprocessing import FEATURE_COLUMNS

        classifier = make_prediction_frame(
            data, score_model(bundle["B_model"], data[FEATURE_COLUMNS]), b["method"]
        )
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
        comparison_groups=(
            {
                "B classifier": group_metrics[b["method"]],
                "C anomaly": group_metrics[config["selected_anomaly"]],
            }
            if classifier is not None
            else None
        ),
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


def save_ablation_figure(rows, selected, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for ax, prefix in zip(axes, ["km_", "if_"], strict=True):
        subset = frame[frame.candidate.str.startswith(prefix)]
        y = np.arange(len(subset))
        ax.barh(
            y,
            subset.TP,
            color=[
                "tab:orange" if n in selected.values() else "tab:blue"
                for n in subset.candidate
            ],
        )
        ax.set_yticks(y, subset.candidate, fontsize=8)
        ax.invert_yaxis()
        ax.set(
            xlabel="Fraud detected at 1% review",
            title="K-means scores" if prefix == "km_" else "Isolation Forest",
        )
        for i, tp in enumerate(subset.TP):
            ax.text(tp + 0.15, i, str(tp), va="center", fontsize=8)
        ax.set_xlim(0, max(1, int(subset.TP.max()) + 3))
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Validation ablations (orange: selected on training-inner data)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def publish_c_results(staging, destination, mode):
    """Collect C's shareable evidence in the original results/{figures,tables}.

    Intermediate per-file exports stay with the local run. Existing files from
    A/B/D are untouched; C outputs must not overwrite an earlier run silently.
    """
    import shutil

    staging, destination = Path(staging), Path(destination)
    if mode not in {"baseline", "improved"}:
        raise ValueError("Unknown C result mode")
    result = json.loads((staging / "summary.json").read_text())
    if mode == "baseline":
        result["frozen_config"] = json.loads((staging / "c_freeze.json").read_text())
        for filename, key in [
            ("kmeans_training_structure.csv", "training_structure"),
            ("train_cluster_profile.csv", "training_cluster_profile"),
            (
                "validation_cluster_feature_means.csv",
                "validation_cluster_feature_means",
            ),
        ]:
            path = staging / filename
            if path.exists():
                result[key] = pd.read_csv(path).to_dict(orient="records")
        figures = {"cluster_structure.png": "c_cluster_structure.png"}
        table_name = "c_baseline.json"
    else:
        lock = json.loads(
            (staging / "selection_locked_before_validation.json").read_text()
        )
        result["selection_record"] = {
            k: v for k, v in lock.items() if k not in {"config", "inner_summary"}
        }
        result["locked_choices"] = {
            k: lock["config"][k]
            for k in [
                "selected_candidates",
                "selected_anomaly",
                "distance_share",
                "selection_locked_at_UTC",
            ]
        }
        result["training_distance_calibration"] = json.loads(
            (staging / "training_distance_calibration.json").read_text()
        )
        result["fit_timings"] = pd.read_csv(staging / "timings.csv").to_dict(
            orient="records"
        )
        figures = {p.name: "c_" + p.name for p in (staging / "figures").glob("*.png")}
        table_name = "c_improved.json"
    if (staging / "verification.json").exists():
        result["verification"] = json.loads((staging / "verification.json").read_text())
    targets = [destination / "tables" / table_name]
    targets += [
        destination / "figures" / name
        for source, name in figures.items()
        if (staging / "figures" / source).exists()
    ]
    if mode == "improved":
        targets.append(destination / "tables/c_ablation.csv")
    if any(path.exists() for path in targets):
        raise ValueError(
            "C result files already exist; choose another --results directory"
        )
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    (destination / "tables" / table_name).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    for source, name in figures.items():
        if (staging / "figures" / source).exists():
            shutil.copy2(staging / "figures" / source, destination / "figures" / name)
    if mode == "improved":
        shutil.copy2(
            staging / "validation_ablation.csv", destination / "tables/c_ablation.csv"
        )
    return [str(path) for path in targets]
