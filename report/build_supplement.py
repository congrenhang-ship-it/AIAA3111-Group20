"""Run supplemental review-list analyses without changing the frozen protocol."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from classification import train_model, score_model
from handbook import CONTEXT_COLUMNS, SPLIT_DAYS
from temporal import DAY_SECONDS, handbook_history_builder

EXPECTED_HANDBOOK_SHA = "cf359432b7ca090e4d8614793e100a59c8fd0bb18c9cf11f414fc61644ae4ab7"
DAILY_BUDGET = 100
EXPLORATION = 10


def fitted(features, labels, rows):
    return train_model(features.iloc[rows], labels[rows], "logistic_regression", "class_weight", seed=42)


def rank(scores, stable_ids):
    return np.lexsort((stable_ids, -scores))


def detection_summary(labels, scores, ids):
    k = int(np.ceil(.01 * len(labels)))
    selected = rank(scores, ids)[:k]
    return {
        "reviews": k,
        "found": int(labels[selected].sum()),
        "total_frauds": int(labels.sum()),
        "AP": float(average_precision_score(labels, scores)),
    }


def workload():
    source = ROOT / "results/tables/ulb_test.json"
    records = json.loads(source.read_text())["metrics"]["logistic_regression_class_weight"]["budgets"]
    result = []
    previous_reviews = previous_found = previous_normal = 0
    for row in records:
        result.append({
            "budget": row["review_fraction"], "reviews": row["K"],
            "found": row["TP"], "normal_reviews": row["FP"],
            "amount_coverage": row["amount_coverage"],
            "extra_reviews": row["K"] - previous_reviews,
            "extra_frauds": row["TP"] - previous_found,
            "extra_normal_reviews": row["FP"] - previous_normal,
        })
        previous_reviews, previous_found, previous_normal = row["K"], row["TP"], row["FP"]
    return result


def replay(features, labels, days, ids, initial, policy, seed=42):
    """Rank closed daily batches; reveal day d review labels at the start of d+2."""
    rng = np.random.default_rng(seed)
    reviewed = {}
    model = fitted(features, labels, initial)
    records = []
    for day in range(SPLIT_DAYS[0], SPLIT_DAYS[2]):
        if policy != "static" and day > 30 and (day - 30) % 3 == 0:
            observed = [rows for when, rows in reviewed.items() if when + 2 <= day]
            if observed:
                model = fitted(features, labels, np.concatenate([initial, *observed]))
        rows = np.flatnonzero(days == day)
        scores = score_model(model, features.iloc[rows])
        order = rank(scores, ids[rows])
        top = order[:DAILY_BUDGET]
        selected = top
        if policy in {"random", "uncertainty"}:
            exploit, pool = order[:DAILY_BUDGET - EXPLORATION], order[DAILY_BUDGET - EXPLORATION:]
            if policy == "random":
                query = rng.choice(pool, EXPLORATION, replace=False)
            else:
                query = pool[np.lexsort((ids[rows[pool]], np.abs(scores[pool] - .5)))[:EXPLORATION]]
            selected = np.r_[exploit, query]
        reviewed[day] = rows[selected]
        assert len(np.unique(reviewed[day])) == DAILY_BUDGET
        records.append({
            "day": day, "reviews": len(selected),
            "policy_found": int(labels[reviewed[day]].sum()),
            "ranking_found": int(labels[rows[top]].sum()),
            "available_fraud": int(labels[rows].sum()),
        })
    test = [r for r in records if r["day"] >= SPLIT_DAYS[1]]
    return {
        "policy": policy, "seed": seed,
        "test_totals": {key: sum(r[key] for r in test) for key in
                        ["reviews", "policy_found", "ranking_found", "available_fraud"]},
        "daily": records,
    }


def sequence_features(data, amount_cut):
    """Encode the last three strictly earlier customer transactions in a 7-day window."""
    encoded = np.full(len(data), -1, dtype=int)
    order = np.lexsort((data.row_id.to_numpy(), data.Time.to_numpy(), data.CUSTOMER_ID.to_numpy()))
    customers = data.CUSTOMER_ID.to_numpy()[order]
    times = data.Time.to_numpy()
    high = data.Amount.to_numpy() >= amount_cut
    cuts = np.flatnonzero(customers[1:] != customers[:-1]) + 1
    for positions in np.split(order, cuts):
        timestamps = times[positions]
        prior = np.searchsorted(timestamps, timestamps, side="left")
        keep = np.flatnonzero(prior >= 3)
        end = prior[keep]
        window = timestamps[keep] - timestamps[end - 3] <= 7 * DAY_SECONDS
        keep, end = keep[window], end[window]
        # When earlier events share a timestamp, row ID resolves their relative order.
        encoded[positions[keep]] = 4 * high[positions[end - 3]].astype(int) + 2 * high[positions[end - 2]] + high[positions[end - 1]]
    bits = np.array([code.bit_count() for code in range(8)])
    count = np.full(len(data), -1, dtype=int)
    count[encoded >= 0] = bits[encoded[encoded >= 0]]
    ordered = pd.DataFrame({f"history_order_{i:03b}": encoded == i for i in range(8)}, index=data.row_id)
    unordered = pd.DataFrame({f"history_high_count_{i}": count == i for i in range(4)}, index=data.row_id)
    return ordered, unordered, encoded >= 0


def sequence_experiment(data, history, labels, train, test):
    cut = float(data.Amount.iloc[train].median())
    ordered, unordered, eligible = sequence_features(data, cut)
    results = {}
    for name, extra in (("history", None), ("unordered", unordered), ("ordered", ordered)):
        features = history if extra is None else pd.concat([history, extra], axis=1)
        model = fitted(features, labels, train)
        results[name] = detection_summary(labels[test], score_model(model, features.iloc[test]), data.row_id.to_numpy()[test])
    return {"training_amount_cut": cut, "eligible_test_rows": int(eligible[test].sum()),
            "eligible_test_frauds": int(labels[test][eligible[test]].sum()), "results": results}


def context_anomaly(views, labels, test, ids):
    """Rank increases over a customer's strictly earlier seven-day mean."""
    history = views["customer_history"].iloc[test]
    amounts = history["Amount"].to_numpy()
    eligible = history["customer_amount_ratio_unavailable_7d"].to_numpy() == 0
    # Amount ratios are nonnegative. The sentinel puts unscorable rows last.
    ratios = np.where(eligible, history["customer_amount_ratio_7d"], -1.0)
    y, row_ids = labels[test], ids[test]
    summary = {
        name: detection_summary(y, scores, row_ids)
        for name, scores in (("amount", amounts), ("customer_ratio", ratios))
    }
    k = summary["amount"]["reviews"]
    amount_rows = rank(amounts, row_ids)[:k]
    ratio_rows = rank(ratios, row_ids)[:k]
    shared = np.intersect1d(amount_rows, ratio_rows)
    amount_only = np.setdiff1d(amount_rows, ratio_rows)
    ratio_only = np.setdiff1d(ratio_rows, amount_rows)
    official = json.loads((ROOT / "results/tables/handbook_test.json").read_text())
    reference = official["metrics"]["amount_descending"]
    reference_budget = next(r for r in reference["budgets"] if r["review_fraction"] == .01)
    assert (k, summary["amount"]["found"]) == (reference_budget["K"], reference_budget["TP"])
    assert np.isclose(summary["amount"]["AP"], reference["AP"])
    return {
        "eligible_rows": int(eligible.sum()),
        "total_rows": len(test),
        "eligible_frauds": int(y[eligible].sum()),
        "ratio_reviews_without_history": int((~eligible[ratio_rows]).sum()),
        "methods": summary,
        "overlap": {"shared_reviews": len(shared), "shared_frauds": int(y[shared].sum()),
                    "amount_only_frauds": int(y[amount_only].sum()),
                    "ratio_only_frauds": int(y[ratio_only].sum())},
    }


def cluster_diagnostics(history, labels, train, test, ids):
    columns = ["Amount", "customer_count_1d", "terminal_count_1d"]
    values = np.log1p(history[columns].to_numpy())
    scaler = StandardScaler().fit(values[train])
    model = GaussianMixture(n_components=3, covariance_type="diag", n_init=3, random_state=42)
    model.fit(scaler.transform(values[train]))
    assert model.converged_
    posteriors = model.predict_proba(scaler.transform(values[test]))
    # Name components by training mean amount, independent of test labels.
    ordered = np.argsort(model.means_[:, 0])
    assignment = posteriors[:, ordered].argmax(axis=1)
    lr = fitted(history, labels, train)
    scores = score_model(lr, history.iloc[test])
    ranked = rank(scores, ids[test])
    reviewed = np.zeros(len(test), dtype=bool)
    reviewed[ranked[:int(np.ceil(.01 * len(test)))]] = True

    def summarize(mask):
        fraud = labels[test] == 1
        return {"rows": int(mask.sum()), "frauds": int((mask & fraud).sum()),
                "reviews": int((mask & reviewed).sum()), "found": int((mask & reviewed & fraud).sum())}

    return {
        "features": columns,
        "groups": {f"component_{i + 1}": summarize(assignment == i) for i in range(3)},
        "membership": {
            "less_than_0.8": summarize(posteriors.max(axis=1) < .8),
            "at_least_0.8": summarize(posteriors.max(axis=1) >= .8),
        },
        "fixed_global_list": detection_summary(labels[test], scores, ids[test]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("handbook_csv", type=Path, help="The pinned 60-day Handbook transactions CSV")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/review_supplement.json")
    args = parser.parse_args()
    digest = hashlib.sha256(args.handbook_csv.read_bytes()).hexdigest()
    if digest != EXPECTED_HANDBOOK_SHA:
        parser.error("CSV does not match the frozen Handbook dataset.")
    data = pd.read_csv(args.handbook_csv)
    labels = data.Class.to_numpy()
    ids = data.row_id.to_numpy()
    days = data.Time.to_numpy() // DAY_SECONDS
    train, test = np.flatnonzero(days < 30), np.flatnonzero(days >= 45)
    views = handbook_history_builder(data[CONTEXT_COLUMNS])
    # Replay starts with the historical training set fully labeled.
    initial = train
    runs = [replay(views["transaction_only"], labels, days, ids, initial, policy, seed)
            for policy, seed in [("static", 42), ("top_risk", 42), ("random", 42),
                                 ("random", 43), ("random", 44), ("uncertainty", 42)]]
    result = {
        "scope": "Supplemental controlled analyses on the existing chronological split",
        "handbook_sha256": digest, "ulb_workload": workload(),
        "review_feedback": {"daily_budget": DAILY_BUDGET, "exploration_slots": EXPLORATION,
                            "initial_labeled_days": [0, 29], "first_replay_day": 30, "test_days": [45, 59],
                            "label_arrival_days_after_review": 2, "refit_interval_days": 3, "runs": runs},
        "context_anomaly": context_anomaly(views, labels, test, ids),
        "sequence": sequence_experiment(data, views["both_history"], labels, train, test),
        "probabilistic_groups": cluster_diagnostics(views["both_history"], labels, train, test, ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
