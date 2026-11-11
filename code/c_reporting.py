"""负责人 C：可公开的小型验证汇总、群体/结构图与验证阶段方案记录。

不包含完整预测、交易行 ID 或原始特征值；完整证据留在被 Git 忽略的 data/。
最终测试可复用 write_summary / save_diagnostic_figures，不在此挑选测试最优方案。
"""

from pathlib import Path
import copy
import hashlib
import json
import math
import numpy as np
import pandas as pd

PRIMARY_REVIEW_FRACTION = 0.01
COMBINATION_SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)


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
