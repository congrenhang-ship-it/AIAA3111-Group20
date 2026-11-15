"""A-N3 PaySim source audit and a bounded E16 validation run."""

import hashlib
import tempfile
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from classification import score_model, train_model
from evaluation import evaluate_amounts, fit_amount_bins, make_prediction_frame, partition_description
from handbook import write_experiment_json
from preprocessing import data_sha256
from unsupervised import fit_isolation_forest, score_isolation_forest

PAYSIM_PAGE = "https://www.kaggle.com/datasets/ealaxi/paysim1"
PAYSIM_METADATA = "https://www.kaggle.com/api/v1/datasets/view/ealaxi/paysim1"
PAYSIM_DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1?datasetVersionNumber=2"
PAYSIM_SHA256 = "f7eef9ffad5cfa64a034143a5c9b30491d189420b273d5ad5723ca40b596613d"
PAYSIM_FILE = "PS_20174392719_1491204439457_log.csv"
PAYSIM_HEADER = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest",
    "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]
READ_FIELDS = ["step", "type", "amount", "isFraud"]
SEED = 42
FRACTION = 0.1


def load_paysim_sample(archive, *, chunk_size=200_000):
    """Use a PCG64 Bernoulli mask over source rows and split on full hour steps."""
    archive = Path(archive)
    if data_sha256(archive) != PAYSIM_SHA256:
        raise ValueError("PaySim archive SHA-256 differs from the pinned author distribution")
    rng = np.random.default_rng(SEED)
    mask_digest = hashlib.sha256()
    offset, positives = 0, 0
    sampled = []
    with ZipFile(archive) as source:
        if source.namelist() != [PAYSIM_FILE]:
            raise ValueError("Unexpected PaySim archive member")
        member = source.getinfo(PAYSIM_FILE)
        with source.open(member) as csv:
            if csv.readline().decode("ascii").strip().split(",") != PAYSIM_HEADER:
                raise ValueError("Unexpected PaySim v2 CSV header")
            for rows in pd.read_csv(csv, names=PAYSIM_HEADER, usecols=READ_FIELDS, chunksize=chunk_size,
                                    dtype={"step": "int64", "isFraud": "int64", "amount": "float64", "type": "str"}):
                if (rows.isna().any().any() or not rows.isFraud.isin([0, 1]).all()
                        or not rows.step.between(1, 744).all()
                        or not np.isfinite(rows.amount).all() or (rows.amount < 0).any()):
                    raise ValueError("Unexpected PaySim source values")
                keep = rng.random(len(rows)) < FRACTION
                mask_digest.update(keep.astype(np.uint8).tobytes())
                retained = rows.loc[keep].copy()
                retained.insert(0, "row_id", offset + np.flatnonzero(keep))
                sampled.append(retained)
                offset += len(rows)
                positives += int(rows.isFraud.sum())
    data = pd.concat(sampled, ignore_index=True).rename(columns={"amount": "Amount", "isFraud": "Class"})
    data = data.sort_values(["step", "row_id"], ignore_index=True)
    data["Time"] = data.step * 3600
    cuts = [int(data.step.iloc[int(len(data) * q)]) for q in (0.6, 0.8)]
    first, last = data.step.searchsorted(cuts, side="left")
    parts = dict(zip(("train", "validation", "test"), (data.iloc[:first], data.iloc[first:last], data.iloc[last:])))
    if any(part.empty for part in parts.values()):
        raise ValueError("PaySim step boundaries must give three nonempty partitions")
    provenance = {
        "archive_sha256": PAYSIM_SHA256, "archive_bytes": archive.stat().st_size,
        "csv_member": PAYSIM_FILE, "csv_bytes": member.file_size, "csv_crc32": f"{member.CRC:08x}",
        "source_rows": offset, "source_frauds": positives, "sample_rows": len(data),
        "sample": {"algorithm": "PCG64 default_rng, one uniform draw per original CSV row",
                   "seed": SEED, "keep_probability": FRACTION, "mask_encoding": "one byte per source row (0/1)",
                   "mask_sha256": mask_digest.hexdigest()},
        "split_boundary_steps": cuts, "partitions": {name: partition_description(part) for name, part in parts.items()},
    }
    return parts, provenance


def _view(rows, encoder):
    categorical = pd.DataFrame(
        encoder.transform(rows[["type"]]), columns=encoder.get_feature_names_out(), index=rows.index,
    )
    return pd.concat([rows[["Amount"]], categorical], axis=1)


def _scores(rows, bundle):
    features = _view(rows, bundle["encoder"])
    return {
        "amount": rows.Amount.to_numpy(),
        "weighted_lr": score_model(bundle["lr"], features),
        "fixed_if": score_isolation_forest(bundle["if"], features),
    }


def run_paysim_validation(archive, output):
    """Record source eligibility and score validation on the fixed E16 protocol."""
    archive, output = Path(archive), Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_id": "paysim", "experiment": "E16", "phase": "source_audit",
        "source": {"author": "Edgar Lopez-Rojas", "simulator": "https://github.com/EdgarLopezPhD/PaySim",
                   "dataset": PAYSIM_PAGE, "dataset_metadata": PAYSIM_METADATA,
                   "download": PAYSIM_DOWNLOAD, "version": 2, "release_date": "2017-04-03",
                   "dataset_license": "CC BY-SA 4.0", "license": "https://creativecommons.org/licenses/by-sa/4.0/"},
        "time_unit": "one hour per step",
        "decision_time": "request received before approval, with requested type and amount",
        "model_inputs": ["type", "amount"],
        "resource_policy": "Read the ZIP member in chunks of 200000 rows; retain only sampled rows in memory.",
        "excluded_fields": {
            "balances": "Author dataset description explicitly excludes all four balance columns due to fraud cancellation",
            "nameOrig/nameDest": "entity IDs", "step": "chronology metadata",
            "isFlaggedFraud": "system decision marker", "isFraud": "simulated ground truth",
        },
        "test_scored": False, "group_G2_complete": False,
    }
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "paysim"
        stage.mkdir()
        if not archive.is_file():
            report.update(status="archive_unavailable", pending=["sample", "validation", "frozen_test"])
        else:
            parts, provenance = load_paysim_sample(archive)
            report["integrity_and_split"] = provenance
            train, validation = (parts[name] for name in ("train", "validation"))
            if set(train.Class) != {0, 1}:
                report.update(status="single_class_train", pending=["validation", "frozen_test"])
            else:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[["type"]])
                X = _view(train, encoder)
                bundle = {
                    "encoder": encoder,
                    "lr": train_model(X, train.Class, "logistic_regression", "class_weight", SEED),
                    "if": fit_isolation_forest(X, seed=SEED, n_estimators=100, max_samples=256, contamination="auto"),
                    "amount_bins": fit_amount_bins(train),
                }
                path = stage / "models.joblib"
                joblib.dump(bundle, path)
                original = _scores(validation, bundle)
                replay = _scores(validation, joblib.load(path))
                if any(not np.array_equal(scores, replay[name]) for name, scores in original.items()):
                    raise RuntimeError("Saved E16 models do not replay the validation scores")
                report.update(
                    status="validation_complete", phase="validation", pending=["frozen_test"],
                    fitted_feature_columns=list(X.columns), category_encoding="one-hot fitted on train, unknown all zero",
                    fitted_model_sha256=data_sha256(path),
                    models={"amount": "descending", "weighted_lr": "B fixed class-weight LR, seed 42",
                            "fixed_if": {"trees": 100, "max_samples": 256, "contamination": "auto", "seed": 42}},
                    metrics={name: evaluate_amounts(make_prediction_frame(validation, values, name),
                                                   validation, bundle["amount_bins"])
                             for name, values in original.items()},
                )
        write_experiment_json(stage / "paysim_a.json", report)
        stage.rename(output)
    return report
