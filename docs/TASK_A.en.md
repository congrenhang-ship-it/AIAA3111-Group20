# Task A Data and Experiment Handoff

[中文](TASK_A.zh-CN.md) · [Task checklist](../TASKS.md) · [Data source and expected fingerprints](../data/README.md)

Task A established the common data source, cleaning policy and chronological split for Group 20. It also provides Python interfaces, the G1 validation entry point and relative-time window analysis. It carries forward the original ULB version, the first-occurrence deduplication policy and stable source row IDs. Group 20 splits approximately 60/20/20 by row count and time order. The earlier review replay used fixed 24/12/12-hour segments, so results should identify their respective protocols.

## File map

| Path | Change | Purpose |
|---|---|---|
| `code/download_data.py` | Added | Download the pinned Zenodo CSV and verify it before installing it locally |
| `code/preprocessing.py` | Updated | Define the source version, read and audit data, deduplicate, split and select features |
| `code/main.py` | Updated | Run the audit, report split fingerprints and route the G1 validation baseline |
| `code/temporal.py` | Updated | Define window boundaries and counts, and call C's shared evaluator |
| `tests/test_task_a.py` | Added | Check download integrity, split fingerprints, time windows and the B/C handoff |
| `docs/TASK_A.zh-CN.md`, `docs/TASK_A.en.md` | Added | Document the handoff, reproduction steps and acceptance status |
| `requirements.txt`, `README.md`, `data/README.md`, `TASKS.md` | Updated | Record dependency versions, commands, data evidence and progress |

Each teammate creates `data/creditcard.csv` and `.venv/` locally. Git ignores both paths.

## Reproduce from a fresh clone

1. Create and activate a Python 3.12 virtual environment in the repository root. Local data verification used CPython 3.12.14.

   ```bash
   python -m venv .venv
   ```

   Use `./.venv/Scripts/Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS and Linux. You can also invoke the virtual environment's Python directly.

2. Fetch the data, install the pinned dependencies and run the checks.

   ```bash
   python code/download_data.py
   python -m pip install -r requirements.txt
   python -m pip check
   python code/main.py --check-data
   python code/main.py --prepare-data
   python -m unittest discover -s tests -v
   ```

   The downloader uses the Python standard library. It verifies and reuses a valid `data/creditcard.csv`. If the existing file fails verification, the script preserves it and exits. Move that file aside and retry, or use `--output` to choose another path and pass the same path to the main command with `--data`. If the transfer is slow, use the manual link in the [data instructions](../data/README.md) and run the same verification.

3. Compare the source SHA-256 and the three `row_ids_sha256` values with the [expected fingerprints](../data/README.md). The full-data test should pass. A skip indicates that the CSV was absent from the default path. When sharing a reproduction, include `git rev-parse HEAD`, `git status --short`, `python --version` and the `--prepare-data` output.

## Interfaces for B, C and D

Put the example in a script under `code/` and run that script directly with Python.

```python
from preprocessing import DEFAULT_DATA_PATH, get_features, load_data, split_data

train, validation, test = split_data(load_data(DEFAULT_DATA_PATH))
X_train, y_train = get_features(train), train["Class"]
X_validation = get_features(validation)
```

- **Data and features.** `split_data` returns three Pandas DataFrames with the original `row_id`, `Time`, `Class` and feature columns. `get_features` selects Time, V1–V28 and Amount in a fixed order. B and C fit models and transformations on training data. D fits bins and discovers rules on training data. Score the test split after G2 freezes the choices.
- **B → A.** `train_model(X_train, y_train, "logistic_regression", strategy="original", seed=42)` returns a fitted pipeline. `score_model(model, X_validation)` returns a one-dimensional score array in validation row order, with higher scores indicating higher risk.
- **C → A.** `make_prediction_frame(validation, scores, method)` returns `row_id, Class, Time, method, score` in the input row order. `evaluate_scores(y_true, scores, review_fractions, row_ids=..., threshold=...)` provides shared overall and window metrics. Use ordinary dicts, lists, scalars and `None` for the JSON output.
- **Time windows.** `summarize_time_windows(predictions, window_seconds=3600, origin=0)` returns counts and status. `evaluate_time_windows` also calls C's evaluator for each nonempty window. Methods share the same transactions, windows and budget fractions. Windows include the start and exclude the end. An empty window has `evaluation=None`. C marks AP and recall undefined for a nonempty window with no positives. Interpret per-window budgets separately from overall top-K.
- **G1.** Run `python code/main.py --run-validation` once B2 and C2 are ready. This entry point fits on training data and scores validation only, using review fractions of 0.1%, 0.5%, 1% and 2%. It exits with a dependency message while the necessary modules are pending.

## Acceptance status for A1–A9

Local reproduction record (2026-11-10). On the same Windows computer, a clean project directory and a fresh Python environment verified download, checksum and reuse over local HTTP with the complete CSV. All five tests passed. Two `--prepare-data` runs produced byte-identical JSON, and the three row-ID fingerprints matched an independent CSV check. Small logistic regression, K-means, Isolation Forest, random oversampling and Apriori examples verified the pinned environment. The remote Zenodo transfer stalled partway through this check. A teammate still needs to verify the complete remote download independently.

| Item | Current status and remaining acceptance criteria |
|---|---|
| A1 | The source, checksum and dependencies are pinned, and the local environment passed its rerun. A teammate still needs to download the data independently and verify the environment. |
| A2 | The audit and cleaning decisions were checked against the complete CSV. |
| A3 | The shared split, feature interface and full-data fingerprints were verified locally. The changes await the team's merge process. |
| A4 | Commands and G1 routing were checked with synthetic inputs. Missing B2/C2 dependencies produce explicit errors. |
| A5 | Synthetic checks cover window boundaries, empty and no-positive windows, and evaluator delegation. Real window metrics await C2. |
| A6 | Awaiting B2/C2 and reproduction of G1 on a teammate's computer. Integration of the remaining classifiers, anomaly methods and rules follows. |
| A7 | Awaiting B5, C6, D4 and team confirmation of the frozen settings, budgets, windows and revision. |
| A8 | Run the frozen test plan and reconcile the evidence after G2. |
| A9 | Data and reproduction instructions are ready. The final report, time analysis, second-machine check and submission package follow the formal results. |

## Handbook and PaySim (A-N1 / A-N2 / A-N3)

Use Python 3.12 with the pinned requirements from the repository root. Use a new directory for each run. Generated data, model files, paired threshold files and row-level predictions remain under data/. To replay the recorded numerical results, set OPENBLAS_NUM_THREADS=1 before the commands.

~~~bash
python code/main.py --prepare-handbook --handbook-pilot --output data/handbook/pilot
python code/main.py --prepare-handbook --output data/handbook/v1
python code/main.py --run-handbook-c --output data/handbook/c_validation
python code/main.py --run-handbook-rolling --output data/handbook/e11_validation
curl -fL "https://www.kaggle.com/api/v1/datasets/download/ealaxi/paysim1?datasetVersionNumber=2" -o data/paysim_v2.zip
python code/main.py --run-paysim-validation --output data/paysim_validation
~~~

On Windows PowerShell, use curl.exe and $env:OPENBLAS_NUM_THREADS = "1". The PaySim v2 ZIP is also available through the [author's Kaggle release](https://www.kaggle.com/datasets/ealaxi/paysim1). Use --paysim-archive for a different path. The runner checks its SHA-256. The Handbook command retrieves and verifies the pinned simulator automatically. To replay Handbook, prepare to a new directory and pass it with --handbook-dir.

| File | Responsibility |
|---|---|
| code/handbook.py | Pinned simulator orchestration, dataset manifest, development-only loader |
| code/temporal.py | Row-keyed as-of history, E14 rebuild callback, E11 rolling validation |
| code/paysim.py | Source audit, streaming sample, chronological split, three validation references |
| code/main.py | Public data preparation and validation commands |
| [Handbook evidence](../results/tables/handbook_a.json), [E11 curve](../results/figures/handbook_a_rolling.png), [PaySim evidence](../results/tables/paysim_a.json) | Aggregate evidence and fingerprints |

Add code/ to PYTHONPATH and call load_handbook_development("data/handbook/v1") from handbook. The returned dictionary has manifest, train, validation, train_views, validation_views and context. B takes matching numeric views and Class, using metadata row_id for alignment. C can run the CLI or call run_handbook_c with the returned objects and temporal.handbook_history_builder. The callback takes the five unlabeled context fields and a boolean Series indexed by source row_id. It returns all current rows using [t-window,t) for earlier visible events. D joins row_id and Class to the four E15 fields in both_history. Use mine_rules(..., dataset_id="handbook") and keep TX_FRAUD_SCENARIO for post-hoc analysis.

The 60-day configuration uses 5,000 customers, 10,000 terminals, customer/terminal profile seeds 42/43, and the published per-customer and per-day seed rules. A pilot ran first. Two production generations returned identical CSV and split fingerprints. E11 updates on days 30, 37 and 44 and pairs each final model with a 1% threshold calibrated on the same untouched final validation day. E16 uses amount and requested transaction type under the source author's balance-field guidance. Held-out scoring follows A7/G2 confirmation and A8.
