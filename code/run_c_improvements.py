"""负责人 C：固定 20 候选的改进实验；内层时间选择，正式验证只报告不重选。"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from c_improvements import (
    SEEDS,
    SHARES,
    configurations,
    candidate_order,
    inner_time_split,
    fit_configuration,
    score_configuration,
    candidate_summary,
    select_families,
    select_pair,
    pair_comparisons,
    score_improved_bundle,
)
from c_reporting import partition_description, select_combination
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
from preprocessing import (
    FEATURE_COLUMNS,
    ULB_SHA256,
    data_sha256,
    load_data,
    split_data,
)
from temporal import evaluate_time_windows

ROOT = Path(__file__).resolve().parents[1]


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
            "c_improvements.py",
            "run_c_improvements.py",
            "evaluation.py",
            "c_reporting.py",
            "preprocessing.py",
            "temporal.py",
        ]
    ]
    paths.append(ROOT / "report/C_IMPROVEMENT_PROTOCOL.md")
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


def _ablation_figure(rows, selected, path):
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
        group_metrics[overall], figures, "Validation - selected improved C\n" + overall
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
    _ablation_figure(summary_rows, selected, figures / "ablation.png")
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
    baseline = ROOT / "results/c_validation/summary.json"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/creditcard.csv")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", type=Path, required=True)
    parser.add_argument("--classification-predictions", type=Path)
    args = parser.parse_args()
    try:
        if args.demo:
            from run_c_validation import _demo

            train, val = _demo()
            info = {"mode": "synthetic_demo", "verified_ulb_split": False}
        else:
            digest = data_sha256(args.data)
            if digest != ULB_SHA256:
                raise ValueError("ULB SHA-256 mismatch")
            train, val, held_out = split_data(load_data(args.data))
            del held_out
            info = {
                "mode": "validation",
                "verified_ulb_split": True,
                "dataset_sha256": digest,
            }
        classifier = (
            pd.read_csv(args.classification_predictions)
            if args.classification_predictions
            else None
        )
        if args.demo and classifier is not None:
            raise ValueError("Do not mix demo and real classifier predictions")
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            result, _ = run_experiment(
                train,
                val,
                args.output,
                args.publish,
                source_info=info,
                classifier=classifier,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "selected": result["config"]["selected_candidates"],
                "anomaly": result["config"]["selected_anomaly"],
                "distance_share": result["config"]["distance_share"],
                "public_output": str(args.publish),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
