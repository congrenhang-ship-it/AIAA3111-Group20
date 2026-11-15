# A-N1 data preparation. Upstream GPL notebook is downloaded at a pinned revision
# and kept in data/. This module orchestrates the published simulator functions.

import ast
import hashlib
import json
import platform
import random
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from evaluation import partition_description
from preprocessing import data_sha256
from temporal import DAY_SECONDS, handbook_history_builder

SOURCE_COMMIT = "81cf7d1714bb7b2f5b496407d9055d91dc68dc25"
SOURCE_REPO = "https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook"
SOURCE_PATH = "Chapter_3_GettingStarted/SimulatedDataset.ipynb"
NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/Fraud-Detection-Handbook/"
    f"fraud-detection-handbook/{SOURCE_COMMIT}/{SOURCE_PATH}"
)
NOTEBOOK_SHA256 = "3906c03740e37ecf1606f548fa937a2e8664127f76a54edad43d1bd64e9a731b"
CONTEXT_COLUMNS = ["row_id", "Time", "CUSTOMER_ID", "TERMINAL_ID", "Amount"]
METADATA_COLUMNS = [*CONTEXT_COLUMNS, "Class", "TX_FRAUD_SCENARIO"]
SPLIT_DAYS = (30, 45, 60)
PARAMETERS = {
    "days": 60, "customers": 5000, "terminals": 10000, "seed": 42,
    "customer_profiles_seed": 42, "terminal_profiles_seed": 43,
    "transaction_rng": "upstream per customer ID", "fraud_rng": "upstream per day",
    "radius": 5, "start_date": "2018-04-01", "amount_threshold": 220,
    "compromised_terminals_per_day": 2, "terminal_days": 28,
    "compromised_customers_per_day": 3, "customer_days": 14,
    "customer_fraction": "floor(one third)", "customer_amount_multiplier": 5,
}
SIMULATOR_FUNCTIONS = {
    "generate_customer_profiles_table", "generate_terminal_profiles_table",
    "get_list_terminals_within_radius", "generate_transactions_table", "add_frauds",
}


def _simulator(notebook_path):
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        notebook_path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(NOTEBOOK_URL, timeout=45) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
            raise ValueError("Source notebook SHA-256 mismatch")
        with notebook_path.open("xb") as target:
            target.write(raw)
    raw = notebook_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError("Cached source notebook SHA-256 mismatch")
    definitions = []
    for cell in json.loads(raw)["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith("def "):
            definition = ast.parse(source).body[0]
            if isinstance(definition, ast.FunctionDef) and definition.name in SIMULATOR_FUNCTIONS:
                definitions.append(definition)
    if {definition.name for definition in definitions} != SIMULATOR_FUNCTIONS or len(definitions) != len(SIMULATOR_FUNCTIONS):
        raise ValueError("Pinned notebook does not contain exactly the expected generator definitions")
    # Run only named definitions from the hash-verified upstream notebook.
    namespace = {"np": np, "pd": pd, "random": random}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), NOTEBOOK_URL, "exec"), namespace)
    return namespace


def _generate(notebook_path, parameters):
    simulator = _simulator(notebook_path)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        customers = simulator["generate_customer_profiles_table"](
            parameters["customers"], random_state=parameters["customer_profiles_seed"]
        )
        terminals = simulator["generate_terminal_profiles_table"](
            parameters["terminals"], random_state=parameters["terminal_profiles_seed"]
        )
        coordinates = terminals[["x_terminal_id", "y_terminal_id"]].to_numpy()
        customers["available_terminals"] = customers.apply(
            simulator["get_list_terminals_within_radius"],
            axis=1, args=(coordinates, parameters["radius"]),
        )
        transactions = []
        for profile in customers.itertuples(index=False):
            frame = simulator["generate_transactions_table"](
                profile, start_date=parameters["start_date"], nb_days=parameters["days"]
            )
            if not frame.empty:
                transactions.append(frame)
        raw = pd.concat(transactions, ignore_index=True)
        # Stable time ties retain the upstream customer and transaction iteration order.
        raw = raw.sort_values("TX_TIME_SECONDS", kind="stable").reset_index(drop=True)
        raw = simulator["add_frauds"](customers, terminals, raw)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    data = raw.rename(columns={"TX_TIME_SECONDS": "Time", "TX_AMOUNT": "Amount", "TX_FRAUD": "Class"})
    data.insert(0, "row_id", np.arange(len(data), dtype=np.int64))
    # Preserve source cent amounts after the simulator's x5 fraud multiplier.
    data["Amount"] = np.rint(data.Amount.to_numpy(float) * 100).astype(np.int64) / 100
    return data.loc[:, METADATA_COLUMNS]


def write_experiment_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _source():
    return {
        "notebook": NOTEBOOK_URL, "commit": SOURCE_COMMIT, "notebook_sha256": NOTEBOOK_SHA256,
        "code_license": "GPL-3.0",
        "license_url": f"{SOURCE_REPO}/blob/{SOURCE_COMMIT}/LICENSE",
        "reference": "Le Borgne, Siblini, Lebichot and Bontempi (2022), Transaction data simulator",
    }


def _partitions(data):
    first, second, end = (day * DAY_SECONDS for day in SPLIT_DAYS)
    if data.Time.min() < 0 or data.Time.max() >= end:
        raise ValueError("Handbook time must be inside the fixed 60-day interval")
    cuts = data.Time.searchsorted([first, second], side="left")
    return dict(zip(("train", "validation", "test"), (data.iloc[:cuts[0]], data.iloc[cuts[0]:cuts[1]], data.iloc[cuts[1]:])))


def prepare_handbook(output, *, pilot=False, notebook_path=None):
    """Generate the declared simulator data into a new private directory."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook_path = notebook_path or (Path(__file__).resolve().parents[1] / "data/handbook_source/SimulatedDataset.ipynb")
    parameters = {**PARAMETERS, "days": 1} if pilot else PARAMETERS.copy()
    data = _generate(notebook_path, parameters)
    if data.isna().any().any() or data.empty or not data.Class.eq(data.TX_FRAUD_SCENARIO.ne(0)).all():
        raise ValueError("Unexpected simulator transaction fields or labels")
    parts = {"pilot": data} if pilot else _partitions(data)
    if any(part.empty for part in parts.values()):
        raise ValueError("Expected nonempty chronological partitions")
    development = data if pilot else pd.concat([parts["train"], parts["validation"]])
    views = handbook_history_builder(development[CONTEXT_COLUMNS])
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "handbook"
        stage.mkdir()
        csv = stage / "transactions.csv"
        data.to_csv(csv, index=False, float_format="%.2f", lineterminator="\n")
        manifest = {
            "dataset_id": "handbook",
            "evidence_kind": "resource_pilot" if pilot else "handbook_validation",
            "data_sha256": data_sha256(csv),
            "data_bytes": csv.stat().st_size,
            "generator_source": _source(),
            "generation_parameters": parameters,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            "partitions": {name: partition_description(part) for name, part in parts.items()},
            "partition_days": None if pilot else {"train": [0, 30], "validation": [30, 45], "test": [45, 60]},
            "metadata_columns": METADATA_COLUMNS,
            "feature_columns": {name: list(frame.columns) for name, frame in views.items()},
            "row_id_policy": "0-based chronological order; ties retain upstream emission order",
            "duplicate_policy": "preserve distinct simulator events",
            "time_unit": "seconds since start_date",
            "amount_unit": "simulated currency",
            "fraud_scenario": "separate metadata; upstream later scenarios overwrite earlier scenario codes",
            "test_scored": False,
        }
        write_experiment_json(stage / "manifest.json", manifest)
        stage.rename(output)
    return manifest


def load_handbook_development(dataset_dir):
    """Return manifest, train/validation views and an unlabeled development stream."""
    dataset_dir = Path(dataset_dir)
    campaign = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    csv = dataset_dir / "transactions.csv"
    if (campaign.get("dataset_id") != "handbook" or campaign.get("evidence_kind") != "handbook_validation"
            or campaign.get("generation_parameters") != PARAMETERS or campaign.get("generator_source") != _source()
            or data_sha256(csv) != campaign.get("data_sha256")):
        raise ValueError("Expected verified A-N1 Handbook production data; rerun preparation")
    count = sum(campaign["partitions"][name]["rows"] for name in ("train", "validation"))
    development = pd.read_csv(csv, nrows=count, float_precision="round_trip")
    if (list(development.columns) != METADATA_COLUMNS or not development.row_id.is_unique
            or not development.Time.is_monotonic_increasing or development.Time.min() < 0
            or development.Time.max() >= SPLIT_DAYS[1] * DAY_SECONDS):
        raise ValueError("Handbook development shape or chronology differs")
    cut = development.Time.searchsorted(SPLIT_DAYS[0] * DAY_SECONDS, side="left")
    train, validation = development.iloc[:cut], development.iloc[cut:]
    for name, frame in (("train", train), ("validation", validation)):
        if frame.empty or partition_description(frame) != campaign["partitions"][name]:
            raise ValueError(f"Handbook {name} partition differs from A's manifest")
    context = development[CONTEXT_COLUMNS]
    views = handbook_history_builder(context)
    if {name: list(frame.columns) for name, frame in views.items()} != campaign["feature_columns"]:
        raise ValueError("Handbook feature columns differ from A's manifest")
    return {
        "manifest": campaign, "train": train, "validation": validation, "context": context,
        "train_views": {name: frame.loc[train.row_id] for name, frame in views.items()},
        "validation_views": {name: frame.loc[validation.row_id] for name, frame in views.items()},
    }
