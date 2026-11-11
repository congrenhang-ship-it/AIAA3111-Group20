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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/creditcard.csv")
    parser.add_argument("--preprocessing-code", type=Path, default=ROOT / "code")
    parser.add_argument("--output", type=Path, default=ROOT / "report/d_validation")
    run(parser.parse_args())
