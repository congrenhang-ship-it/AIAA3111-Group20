"""D4: compare a common-environment replay to the previously fixed rule set.

No rule selection or test scoring occurs here. Fail on changed identities,
counts, rules, dependencies or numerical drift exceeding the recorded tolerance.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


def verify(reference, replay, lock_path, output):
    old = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    new = json.loads((replay / "manifest.json").read_text(encoding="utf-8"))
    for field in ["data_sha256", "partitions", "feature_columns", "selected_scheme", "frozen_rules"]:
        if old[field] != new[field]:
            raise ValueError(f"Candidate freeze changed: {field}; review rather than reselect.")
    # Cut points are checked separately with a strict, explicit float tolerance.
    max_boundary_difference = 0.0
    for scheme in old["schemes"]:
        left, right = old["schemes"][scheme], new["schemes"][scheme]
        if left["retained_rule_ids"] != right["retained_rule_ids"]:
            raise ValueError("Retained rule IDs changed.")
        for feature in old["feature_columns"]:
            x, y = left["binner"]["features"][feature], right["binner"]["features"][feature]
            if x["effective_bins"] != y["effective_bins"]:
                raise ValueError("Effective bin count changed.")
            a, b = np.array(x["cut_points"]), np.array(y["cut_points"])
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
            if len(a):
                max_boundary_difference = max(max_boundary_difference, float(np.abs(a-b).max()))
    a = pd.read_csv(reference / "train_validation_rules.csv")
    b = pd.read_csv(replay / "train_validation_rules.csv")
    pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-12, atol=1e-12)
    installed, mismatches = {}, {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        package, expected = line.strip().split("==")
        installed[package] = version(package)
        if installed[package] != expected:
            mismatches[package] = {"expected": expected, "actual": installed[package]}
    if mismatches or platform.python_version() != "3.12.14":
        raise ValueError(f"Common environment mismatch: {platform.python_version()}, {mismatches}")
    record = {"status": "D4_common_environment_verified", "group_G2_complete": False,
              "test_rule_evaluation": False, "python": platform.python_version(),
              "packages": installed, "dependency_mismatches": mismatches,
              "reference_implementation_commit": old["d_commit"],
              "preprocessing_commit": new["preprocessing_commit"],
              "replay_source_sha256": new["source_sha256"],
              "data_sha256": new["data_sha256"], "partitions": new["partitions"],
              "candidate_rows_compared": len(a), "retained_rules": len(new["frozen_rules"]),
              "max_boundary_absolute_difference": max_boundary_difference,
              "float_tolerance": {"rtol": 1e-12, "atol": 1e-12},
              "frozen_binner": old["frozen_binner"], "frozen_rules": old["frozen_rules"],
              "note": "Original bins/rules preserved; no reselection; A confirms group G2 separately."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Verified {len(a)} candidate records and {len(new['frozen_rules'])} frozen rules: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("report/d_validation"))
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/tables/d_freeze.json"))
    args = parser.parse_args()
    verify(args.reference, args.replay, args.lock, args.output)
