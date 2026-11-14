"""D: reproduce A's pinned split, compare two rule schemes on validation only.

No test rule evaluation. Until A3 is merged, --preprocessing-code explicitly
names A's unmodified worktree code directory; this is a dependency-branch check.
Only aggregate metrics/frozen configuration are exported, never covered IDs.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from pattern_mining import mine_rules, validate_rules


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FINGERPRINTS = {
    "train": "1f69bf9bb0e7e5439a0896426eae142926d5e77b7d759a4e64bfa39c0a946333",
    "validation": "2b3d3d2dd89b8a07a411d51d5d2afdf04b44922551bb7e6aad90aaef7a876de1",
    "test": "4909103f5579d282c4faa1ca32c3c57fdf328e5d2ade9533a297d3e0254ca5c9",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def commit_at(directory):
    return subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()


def select_features(train):
    """Predeclared selector: Amount + top four training-only V correlations.

    Absolute Pearson correlation with training Class; lexical tie break. Label
    usage in this supervised feature selection is explicit. Class is never a
    binned feature or rule antecedent. No validation/test selection takes place.
    """
    names = [f"V{i}" for i in range(1, 29)]
    correlations = train[names].corrwith(train["Class"]).abs()
    if not np.isfinite(correlations.to_numpy()).all():
        raise ValueError("Training feature selection has undefined correlations.")
    ranked = sorted(names, key=lambda name: (-float(correlations[name]), name))
    return ["Amount", *ranked[:4]], {name: float(correlations[name]) for name in ranked}


def public_metrics(metrics):
    return {key: value for key, value in metrics.items() if key != "covered_row_ids"}


def reliable(metrics):
    return (metrics["coverage_count"] >= 20 and metrics["fraud_count"] >= 5
            and metrics["lift"] is not None and metrics["lift"] > 1)


def run(args):
    source = args.preprocessing_code.resolve() / "preprocessing.py"
    spec = importlib.util.spec_from_file_location("task_a_preprocessing", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "ULB_SHA256"):
        raise ValueError("A3 is not available here; supply A's implemented code directory.")
    data_hash = module.data_sha256(args.data)
    if data_hash != module.ULB_SHA256:
        raise ValueError("Raw dataset SHA-256 does not match A's pinned version.")
    parts = module.split_data(module.load_data(args.data))
    partition_summary = {}
    for name, frame in zip(EXPECTED_FINGERPRINTS, parts, strict=True):
        fingerprint = hashlib.sha256("".join(f"{i}\n" for i in frame.row_id).encode("ascii")).hexdigest()
        if fingerprint != EXPECTED_FINGERPRINTS[name]:
            raise ValueError(f"A's ordered row-ID fingerprint mismatch: {name}")
        partition_summary[name] = {"rows": len(frame), "frauds": int(frame.Class.sum()),
                                   "row_ids_sha256": fingerprint}
    train, validation, test = parts
    del test, parts  # no rule scoring or rule selection using the test partition
    features, correlations = select_features(train)
    rows, schemes, rule_sets = [], {}, {}
    for strategy in ["equal_width", "equal_frequency"]:
        binner, rules = mine_rules(train, features, bins=3, max_length=3, binning=strategy)
        scores = validate_rules(validation, binner, rules)
        kept = [rule for rule, result in zip(rules, scores, strict=True) if reliable(result)]
        out_of_range = {}
        for feature in features:
            fit = binner["features"][feature]
            out_of_range[feature] = {
                "below_train_min": int((validation[feature] < fit["train_min"]).sum()),
                "above_train_max": int((validation[feature] > fit["train_max"]).sum()),
            }
        schemes[strategy] = {"binner": binner, "candidate_count": len(rules),
                             "retained_rule_ids": [r["rule_id"] for r in kept],
                             "validation_out_of_range": out_of_range}
        rule_sets[strategy] = kept
        for rule, result in zip(rules, scores, strict=True):
            before = rule["train"]
            rows.append({"scheme": strategy, "rule_id": rule["rule_id"],
                         "antecedent": json.dumps(rule["antecedent"], sort_keys=True),
                         "validation_pass": reliable(result),
                         "confidence_change": result["confidence"] - before["confidence"]
                         if result["confidence"] is not None else None,
                         "lift_change": result["lift"] - before["lift"]
                         if result["lift"] is not None else None,
                         **{f"train_{k}": v for k, v in public_metrics(before).items()},
                         **{f"validation_{k}": v for k, v in public_metrics(result).items()}})
    # Predeclared choice: more validation-reliable rules; fixed equal-frequency
    # preference on a tie. Keep every passing rule of the chosen scheme; no
    # post-hoc cherry-picking of individual test or validation examples.
    selected = max(["equal_frequency", "equal_width"],
                   key=lambda s: len(schemes[s]["retained_rule_ids"]))
    frozen_rules = [{"rule_id": r["rule_id"], "antecedent": r["antecedent"],
                     "consequent": r["consequent"]} for r in rule_sets[selected]]
    manifest = {
        "status": "D_candidate_frozen_pending_A3_merge_and_G2",
        "test_rule_evaluation": False,
        "data_sha256": data_hash,
        "preprocessing_commit": commit_at(source.parent),
        "d_commit": commit_at(ROOT),
        "source_sha256": {"preprocessing.py": digest(source),
                          "pattern_mining.py": digest(ROOT / "code/pattern_mining.py"),
                          "run_d_validation.py": digest(__file__)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__},
        "command": [sys.executable, *sys.argv],
        "partitions": partition_summary,
        "feature_selection": "Amount plus four V features by absolute training Pearson correlation with Class; lexical ties",
        "feature_columns": features,
        "training_abs_correlations": correlations,
        "validation_gate": {"min_coverage": 20, "min_fraud_coverage": 5, "lift_exclusive": 1.0},
        "scheme_selection": "most passing validation rules; equal_frequency on tie; retain all passing rules",
        "schemes": schemes, "selected_scheme": selected,
        "frozen_binner": schemes[selected]["binner"], "frozen_rules": frozen_rules,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output / "train_validation_rules.csv", index=False)
    print(json.dumps({"features": features, "schemes": {
        s: {"candidates": v["candidate_count"], "retained": len(v["retained_rule_ids"])}
        for s, v in schemes.items()}, "selected": selected,
        "output": str(args.output)}, indent=2))


def run_extensions(args):
    """E04 only: fixed-rule diagnostics; recreate two fixed scorers if needed."""
    import math
    from threadpoolctl import threadpool_limits
    from pattern_mining import analyze_rule_overlap
    sys.path.insert(0, str(args.preprocessing_code.resolve()))
    from preprocessing import load_data, split_data, data_sha256, ULB_SHA256, get_features
    from classification import train_model, score_model
    from unsupervised import fit_isolation_forest, score_isolation_forest
    from evaluation import make_prediction_frame, evaluate_scores
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frozen = json.loads((ROOT / "results/tables/d_freeze.json").read_text(encoding="utf-8"))
    if data_sha256(args.data) != ULB_SHA256:
        raise ValueError("Raw data identity mismatch.")
    train, validation, unused_test = split_data(load_data(args.data))
    del unused_test
    for name, frame in [("train", train), ("validation", validation)]:
        actual = hashlib.sha256("".join(f"{i}\n" for i in frame.row_id).encode("ascii")).hexdigest()
        if actual != EXPECTED_FINGERPRINTS[name]:
            raise ValueError("Shared split identity mismatch.")
    table = pd.read_csv(ROOT / "report/d_validation/train_validation_rules.csv")
    metrics = {row.rule_id: {"lift": row.train_lift, "confidence": row.train_confidence}
               for row in table[table.scheme == "equal_frequency"].itertuples()}
    rules, binner = frozen["frozen_rules"], frozen["frozen_binner"]
    summary = analyze_rule_overlap(validation, binner, rules, metrics)
    coverage = {r["rule_id"]: set(r["covered_row_ids"])
                for r in validate_rules(validation, binner, rules)}
    all_covered = set().union(*coverage.values())
    ids = validation.row_id.to_numpy()
    labels = validation.Class.to_numpy()
    k = math.ceil(len(validation) * .01)
    # One fixed re-fit each, not the original candidate searches. Predictions
    # are recreated here because other members' private files are not in Git.
    with threadpool_limits(limits=1):
        lr = train_model(get_features(train), train.Class, "logistic_regression", strategy="class_weight", seed=42)
        forest = fit_isolation_forest(get_features(train), seed=42, n_estimators=100, max_samples=4096)
        scores = {"classifier": score_model(lr, get_features(validation)),
                  "IF": score_isolation_forest(forest, get_features(validation))}
    args.output.mkdir(parents=True, exist_ok=True)
    predictions, selections, ranks, model_metrics = [], {}, {}, {}
    for name, values in scores.items():
        prediction = make_prediction_frame(validation, values, name)
        predictions.append(prediction)
        order = np.lexsort((ids, values * -1))
        selections[name] = set(ids[order[:k]])
        ranks[name] = {int(ids[position]): rank for rank, position in enumerate(order, start=1)}
        evaluated = evaluate_scores(labels, values, (.01,), row_ids=ids)
        model_metrics[name] = {"AP": evaluated["AP"], "budgets": evaluated["budgets"]}
    if [model_metrics[n]["budgets"][0]["TP"] for n in ["classifier", "IF"]] != [49, 24]:
        raise ValueError("Fixed model replay differs from published validation TP; review source/environment.")
    pd.concat(predictions).to_csv(args.output / "predictions.csv", index=False)
    positives = set(validation.loc[validation.Class.eq(1), "row_id"])
    negatives = set(ids) - positives
    errors = {}
    for name, selected in selections.items():
        errors[name + "_FN"] = positives - selected
        errors[name + "_FP"] = negatives & selected
    summary["error_associations"] = {name: {"total": len(values),
        "covered_by_any_rule": len(values & all_covered)} for name, values in errors.items()}
    summary["per_rule_errors"] = {rule: {name: len(values & covered) for name, values in errors.items()}
                                   for rule, covered in coverage.items()}
    cases, used = [], set()
    for kind, values in errors.items():
        # Predeclared: first two covered examples by row_id, skipping prior cases.
        candidates = sorted(values & all_covered - used)[:2]
        for row_id in candidates:
            used.add(row_id)
            cases.append({"case_id": f"D-E04-{len(cases)+1}", "source_row_id": int(row_id),
                          "category": kind, "Class": int(row_id in positives),
                          "classifier_rank": ranks["classifier"][int(row_id)],
                          "IF_rank": ranks["IF"][int(row_id)],
                          "covering_rules": [r for r in summary["rule_order"] if row_id in coverage[r]]})
    summary["cases"] = cases
    fig, ax = plt.subplots(figsize=(10, 8))
    mat = np.array(summary["jaccard"], dtype=float)
    im = ax.imshow(np.ma.masked_invalid(mat), vmin=0, vmax=1, cmap="viridis")
    names = summary["rule_order"]
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Validation: fixed-rule transaction coverage Jaccard")
    fig.colorbar(im, ax=ax, label="Intersection / union; N/A if union is empty")
    fig.tight_layout()
    figures = ROOT / "results/figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "d_validation_rule_overlap.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    steps = summary["cumulative"]
    for ax, key in zip(axes, ["transactions", "frauds"], strict=True):
        x = np.arange(1, len(steps)+1)
        ax.plot(x, [r[key] for r in steps], "o-", label="Unique cumulative")
        ax.bar(x, [r["increment_"+key] for r in steps], alpha=.35, label="New at this rule")
        ax.set(xlabel="Rule position (fixed training order)", ylabel=key.title(),
               title="Validation coverage: " + key, xticks=x)
        ax.tick_params(axis="x", labelrotation=90)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "d_validation_rule_cumulative.png", dpi=180)
    plt.close(fig)
    result = {"phase": "validation", "group_G2_complete": False, "test_scored": False,
              "source_commit": commit_at(ROOT), "dependency_commit": commit_at(args.preprocessing_code),
              "source_sha256": {p.name: digest(p) for p in [Path(__file__), ROOT/"code/pattern_mining.py",
                  args.preprocessing_code/"classification.py", args.preprocessing_code/"unsupervised.py",
                  args.preprocessing_code/"evaluation.py"]},
              "python": platform.python_version(), "data_sha256": ULB_SHA256,
              "partitions": frozen["partitions"],
              "command": [sys.executable, *sys.argv],
              "versions": {name: __import__("importlib.metadata", fromlist=["version"]).version(name)
                           for name in ["numpy", "pandas", "scikit-learn", "matplotlib"]},
              "config": {"rule_order": "train lift desc, confidence desc, length asc, ID asc",
                  "high_overlap_threshold": .9, "display": "first five positive incremental fraud rules in fixed order",
                  "case_selection": "first two covered errors by row_id per category; skip earlier cases",
                  "primary_budget": .01, "ranking": "score descending; row_id ascending; ceil(N*q)",
                  "fixed_classifier": "class-weighted LR seed42", "fixed_IF": "100 trees, sample4096, seed42",
                  "test_set": "all original 21 rules, not only display subset"},
              "predictions_sha256": digest(args.output/"predictions.csv"),
              "freeze_sha256": digest(ROOT/"results/tables/d_freeze.json"),
              "model_validation": model_metrics, "E04": summary}
    destination = ROOT/"results/tables/d_extensions.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "unique_coverage": summary["covered_transactions"],
                      "unique_frauds": summary["covered_frauds"], "errors": summary["error_associations"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/creditcard.csv")
    parser.add_argument("--preprocessing-code", type=Path, default=ROOT / "code")
    parser.add_argument("--output", type=Path, default=ROOT / "report/d_validation")
    parser.add_argument("--extensions", action="store_true", help="Run E04 fixed-rule coverage and error diagnostics.")
    args = parser.parse_args()
    if args.extensions:
        if args.output == ROOT / "report/d_validation":
            args.output = ROOT / "data/d_extensions"
        run_extensions(args)
    else:
        run(args)
