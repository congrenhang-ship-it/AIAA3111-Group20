"""B2-B5: bounded classification comparison using A's splits and C's metrics.

Only training and validation are scored. Full predictions/models stay in data/.
Run: py -3.12 code/run_b_validation.py
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from classification import MODEL_NAMES, score_model, train_model
from evaluation import DEFAULT_BUDGETS, PRIMARY_REVIEW_FRACTION, evaluate_scores, make_prediction_frame, save_figures
from main import split_summary
from preprocessing import DEFAULT_DATA_PATH, FEATURE_COLUMNS, ULB_SHA256, data_sha256, get_features, load_data, split_data, source_commit, source_dirty

ROOT = Path(__file__).resolve().parents[1]
# Declared before any validation run. No grid search or test evaluation.
EXPERIMENTS = [(name, "original") for name in MODEL_NAMES] + [
    ("logistic_regression", "class_weight"),
    ("logistic_regression", "random_undersample"),
]
SELECTION_RULE = "maximum validation TP at 1%; then AP; then method name ascending"


def select_method(results):
    def key(record):
        metric = record["metrics"]
        budget = next(row for row in metric["budgets"] if row["review_fraction"] == PRIMARY_REVIEW_FRACTION)
        return (-budget["TP"], -metric["AP"], record["method"])
    return min(results, key=key)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run(data_path=DEFAULT_DATA_PATH):
    digest = data_sha256(data_path)
    if digest != ULB_SHA256:
        raise ValueError(f"ULB SHA-256 mismatch: {digest}")
    train, validation, test = split_data(load_data(data_path))
    partitions = split_summary((train, validation, test))
    # No test scores, test-based selection or train+validation refitting.
    del test
    X_train, X_validation = get_features(train), get_features(validation)
    local = ROOT / "data" / "b_validation"
    local.mkdir(parents=True, exist_ok=True)
    tables = ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    records, rows, predictions, models = [], [], {}, {}
    for name, strategy in EXPERIMENTS:
        method = f"{name}_{strategy}"
        start = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with threadpool_limits(limits=1):
                model = train_model(X_train, train.Class, name, strategy=strategy, seed=42)
                train_seconds = perf_counter() - start
                scores = score_model(model, X_validation)
        frame = make_prediction_frame(validation, scores, method)
        frame.to_csv(local / f"{method}.csv", index=False, float_format="%.17g")
        metric = evaluate_scores(frame.Class, frame.score, DEFAULT_BUDGETS, row_ids=frame.row_id)
        # PR arrays are large; preserve full metrics locally, compact evidence in Git.
        write_json(local / f"{method}_metrics.json", metric)
        compact = {key: value for key, value in metric.items() if key != "pr_curve"}
        record = dict(method=method, model_name=name, strategy=strategy,
                      train_seconds=train_seconds, seed=42,
                      classifier_parameters=model.named_steps["classifier"].get_params(),
                      fitted_class_counts=model.training_class_counts_, metrics=compact)
        records.append(record)
        predictions[method] = frame
        models[method] = model
        for budget in metric["budgets"]:
            rows.append(dict(method=method, AP=metric["AP"], train_seconds=train_seconds, **budget))
        print(f"{method}: AP={metric['AP']:.6f}, train={train_seconds:.2f}s", flush=True)

    selected = select_method(records)
    metrics = {record["method"]: json.loads((local / f"{record['method']}_metrics.json").read_text(encoding="utf-8")) for record in records}
    for figure in save_figures({"methods": metrics, "label": "B classification - validation only"}, local / "figures"):
        path = Path(figure)
        target = ROOT / "results" / "figures" / ("b_" + path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    selected_frame = predictions[selected["method"]]
    selected_path = local / "selected_validation_predictions.csv"
    selected_frame.to_csv(selected_path, index=False, float_format="%.17g")
    model_path = local / "selected_model.joblib"
    joblib.dump(models[selected["method"]], model_path)
    with threadpool_limits(limits=1):
        replay_scores = score_model(joblib.load(model_path), X_validation)
    np.testing.assert_allclose(replay_scores, selected_frame.score.to_numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(
        np.lexsort((validation.row_id.to_numpy(), -replay_scores)),
        np.lexsort((validation.row_id.to_numpy(), -selected_frame.score.to_numpy())),
    )
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")}
    locked = dict(line.split("==", 1) for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if "==" in line and not line.startswith("#"))
    mismatches = {name: dict(actual=version, required=locked.get(name)) for name, version in versions.items() if locked.get(name) != version}
    code_files = ["code/classification.py", "code/run_b_validation.py", "code/preprocessing.py", "code/evaluation.py"]
    metadata = dict(
        owner="B", phase="validation", dataset_sha256=digest, partitions=partitions,
        python=platform.python_version(), versions=versions,
        dependency_lock_mismatches=mismatches,
        source_commit=source_commit(ROOT),
        source_dirty=source_dirty(ROOT),
        code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in code_files},
        command="py -3.12 code/run_b_validation.py",
    )
    write_json(tables / "b_validation.json", dict(metadata=metadata, selection_rule=SELECTION_RULE, experiments=records, selected=selected["method"]))
    pd.DataFrame(rows).to_csv(tables / "b_validation.csv", index=False)
    freeze = dict(
        schema_version=1, owner="B", status="B_validation_selection_pending_group_review",
        group_G2_complete=False, test_evaluated=False, dataset_sha256=digest,
        partitions=partitions, feature_columns=FEATURE_COLUMNS, seed=42,
        method=selected["method"], model_name=selected["model_name"], strategy=selected["strategy"],
        classifier_parameters=selected["classifier_parameters"],
        preprocessing="StandardScaler for logistic_regression/gaussian_nb; passthrough for trees",
        resampling="training only; seeded 1:1 random undersampling without replacement when selected",
        selection_rule=SELECTION_RULE, primary_review_fraction=PRIMARY_REVIEW_FRACTION,
        review_fractions=DEFAULT_BUDGETS, ranking="score descending; row_id ascending; K=ceil(N*q)",
        threshold=None, threshold_reason="Budget ranking defines decisions; no fixed probability threshold selected.",
        refit_policy="training partition only, never train+validation",
        predictions_path="data/b_validation/selected_validation_predictions.csv",
        predictions_sha256=data_sha256(selected_path), provenance=metadata,
        fitted_model_path="data/b_validation/selected_model.joblib",
    )
    write_json(tables / "b_freeze.json", freeze)
    print(f"Selected: {selected['method']}; evidence: results/tables/b_validation.csv", flush=True)
    return freeze


def extension_provenance():
    return dict(
        phase="validation", test_scored=False, group_G2_complete=False,
        source_commit=source_commit(ROOT),
        code_sha256={name: data_sha256(ROOT / name) for name in (
            "code/classification.py", "code/run_b_validation.py", "code/main.py", "code/evaluation.py",
            "code/handbook.py", "code/temporal.py", "code/unsupervised.py")},
        python=platform.python_version(),
        versions={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")},
    )


def compact_metrics(frame):
    metric = evaluate_scores(frame.Class, frame.score, DEFAULT_BUDGETS, row_ids=frame.row_id)
    metric.pop("pr_curve", None)
    return metric


def run_ulb_extensions(data_path, output):
    """Standalone E03/E05 command, sharing the team runner's implementation."""
    if data_sha256(data_path) != ULB_SHA256:
        raise ValueError("ULB file differs from A's pinned source")
    train, validation, test = split_data(load_data(data_path))
    partitions = split_summary((train, validation, test))
    return run_ulb_extensions_on_split(train, validation, partitions, output,
                                       results=ROOT / "results")


def run_ulb_extensions_on_split(train, validation, partitions, output, *, results):
    """E03/E05 on A's supplied split; no loading/repartitioning or test scoring.

    The team runner supplies private and aggregate staging directories. The
    standalone command supplies the existing shared results directory.
    """
    from classification import LABEL_FRACTIONS, LABEL_SEEDS, nested_label_masks, fixed_error_cases
    from unsupervised import fit_isolation_forest, score_isolation_forest
    from sklearn.inspection import permutation_importance
    from evaluation import partition_description

    for name, frame in (("train", train), ("validation", validation)):
        if partition_description(frame) != partitions[name]:
            raise ValueError(f"B extensions received a different {name} split")
    output, results = Path(output).resolve(), Path(results).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (results / "tables").mkdir(parents=True, exist_ok=True)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    X, V = get_features(train), get_features(validation)
    rows, cached = [], {}
    full_model = None
    for seed in LABEL_SEEDS:
        for fraction, positions in nested_label_masks(train.Class, seed).items():
            mask_hash = hashlib.sha256(train.row_id.to_numpy(dtype="<i8")[positions].tobytes()).hexdigest()
            reused = mask_hash in cached
            if not reused:
                start = perf_counter()
                model = train_model(X.iloc[positions], train.Class.iloc[positions], "logistic_regression", "class_weight", seed=42)
                seconds = perf_counter() - start
                start = perf_counter()
                frame = make_prediction_frame(validation, score_model(model, V), f"lr_labels_{fraction:g}_{seed}")
                score_seconds = perf_counter() - start
                cached[mask_hash] = (compact_metrics(frame), seconds, score_seconds)
                joblib.dump(model, output / f"ulb_labels_{fraction:g}_{seed}.joblib")
                if fraction == 1:
                    full_model = model
            metric, seconds, score_seconds = cached[mask_hash]
            np.save(output / f"ulb_mask_{fraction:g}_{seed}.npy", positions)
            rows.append(dict(fraction=fraction, mask_seed=seed, model_seed=42,
                             labeled_rows=len(positions), labeled_frauds=int(train.Class.iloc[positions].sum()),
                             mask_sha256=mask_hash, reuse_full_fit=reused,
                             train_seconds=seconds, score_seconds=score_seconds, metrics=metric))
            print(f"E03 fraction={fraction} seed={seed} AP={metric['AP']:.6f}", flush=True)
    start = perf_counter()
    if_model = fit_isolation_forest(X, seed=42)
    if_seconds = perf_counter() - start
    start = perf_counter()
    reference = make_prediction_frame(validation, score_isolation_forest(if_model, V), "IF_fixed256")
    if_score_seconds = perf_counter() - start
    joblib.dump(if_model, output / "ulb_if256.joblib")
    frame = make_prediction_frame(validation, score_model(full_model, V), "logistic_regression_class_weight")
    start = perf_counter()
    importance = permutation_importance(full_model, V, validation.Class, scoring="average_precision",
                                       n_repeats=5, random_state=42, n_jobs=1)
    importance_rows = [dict(feature=name, mean=float(importance.importances_mean[i]),
                            std=float(importance.importances_std[i]), repeats=importance.importances[i].tolist())
                       for i, name in enumerate(V.columns)]
    k = int(np.ceil(len(validation) * PRIMARY_REVIEW_FRACTION))
    cases = fixed_error_cases(frame, k)
    cases.merge(validation, on=["row_id", "Class", "Time"], validate="many_to_one").to_csv(output / "ulb_error_cases.csv", index=False)
    result = dict(dataset_id="ulb", metadata=extension_provenance(), partitions=partitions,
                  dataset_sha256=ULB_SHA256,
                  E03=dict(fractions=list(LABEL_FRACTIONS), seeds=list(LABEL_SEEDS),
                           mask_hash_rule="SHA256 of selected row_ids in training order, little-endian int64",
                           sampling="class-stratified nested permutations; ceil per class; labels used only to simulate annotation",
                           fitting="LR sees only labeled rows; IF sees all unlabeled training features; hidden labels never become 0",
                           pipeline_parameters=full_model.named_steps['classifier'].get_params(),
                           conditions=rows, IF_reference=dict(metrics=compact_metrics(reference), train_seconds=if_seconds,
                               score_seconds=if_score_seconds, parameters=if_model.named_steps['model'].get_params())),
                  E05=dict(scoring="average_precision", seed=42, repeats=5, fitting="no refit, all original features",
                           elapsed_seconds=perf_counter()-start, importances=importance_rows,
                           case_rule="two FP closest inside K; two FN closest outside K; two highest-ranked FP; score desc,row_id asc; categories may overlap",
                           case_counts={str(key):int(value) for key,value in cases.category.value_counts().items()},
                           private_cases=str((output / 'ulb_error_cases.csv').relative_to(ROOT))))
    write_json(output / "ulb_b_extensions.json", result)
    write_json(results / "tables/b_extensions.json", result)
    save_b_extension_figure(result, results / "figures/b_extensions.png")
    print("E03/E05 completed", flush=True)
    return result


def save_b_extension_figure(result, path, *, phase="validation"):
    """Plot existing aggregate E03/E05 evidence; no training or selection."""
    rows = result["E03"]["conditions"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for field, label in (("AP", "AP"), ("recall_at_k", "Recall@1%")):
        matrix = np.array([[next(r for r in rows if r['fraction']==f and r['mask_seed']==s)['metrics']['AP']
                            if field=='AP' else next(b['recall_at_k'] for b in next(r for r in rows if r['fraction']==f and r['mask_seed']==s)['metrics']['budgets'] if b['review_fraction']==.01)
                            for s in result['E03']['seeds']] for f in result['E03']['fractions']])
        x = np.array(result['E03']['fractions'])*100
        axes[0].plot(x, matrix.mean(axis=1), marker='o', label=label)
        axes[0].fill_between(x, matrix.min(axis=1), matrix.max(axis=1), alpha=.15)
    axes[0].axhline(result['E03']['IF_reference']['metrics']['AP'], ls='--', label='IF AP (fixed 256)')
    axes[0].set(xlabel="Labeled training rows (%)", ylabel="Validation metric", title="Mean and three-seed range (not CI)", ylim=(0,1))
    axes[0].legend(fontsize=8)
    top = sorted(result['E05']['importances'], key=lambda r:r['mean'])[-10:]
    axes[1].barh([r['feature'] for r in top], [r['mean'] for r in top], xerr=[r['std'] for r in top])
    axes[1].set(xlabel="AP decrease after permutation (5 repeats)", title="Fixed weighted LR: model reliance")
    fig.suptitle(f"ULB {phase} - E03 / E05")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)



def run_handbook_b(dataset_dir, output, *, results=None):
    """E10/E13: four fixed LR fits using A's views; no test or view selection."""
    from handbook import load_handbook_development
    from classification import cold_start_groups
    from unsupervised import _checked_handbook_views, _handbook_view_fingerprints, HANDBOOK_FINGERPRINT_VERSION
    from evaluation import evaluate_amounts, evaluate_groups, fit_amount_bins
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = Path(results) if results is not None else ROOT / "results"
    (results / "tables").mkdir(parents=True, exist_ok=True)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    development = load_handbook_development(dataset_dir)
    expected = json.loads((ROOT / 'results/tables/handbook_a.json').read_text(encoding='utf-8'))['A_N1']['production']
    if development['manifest']['data_sha256'] != expected['data_sha256']:
        raise ValueError("Generated Handbook bytes differ from A's delivered dataset; do not publish unmatched records")
    train, validation = development['train'], development['validation']
    views = _checked_handbook_views(train, development['train_views'])
    valid_views = _checked_handbook_views(validation, development['validation_views'])
    groups = cold_start_groups(train, validation)
    group_names = ['both_seen', 'new_customer_only', 'new_terminal_only', 'both_new']
    bins, conditions = fit_amount_bins(train), {}
    for name, X in views.items():
        start = perf_counter()
        model = train_model(X, train.Class, 'logistic_regression', 'class_weight', seed=42)
        seconds = perf_counter()-start
        start = perf_counter()
        scores = score_model(model, valid_views[name])
        score_seconds = perf_counter()-start
        frame = make_prediction_frame(validation, scores, f'lr_{name}')
        joblib.dump(model, output / f'handbook_lr_{name}.joblib')
        frame.to_csv(output / f'handbook_lr_{name}.csv', index=False)
        grouping = evaluate_groups(frame, groups, DEFAULT_BUDGETS, group_values=group_names)
        for record in grouping['groups']:
            group = record['cluster_id']
            mask = groups == group
            # Only global allocations are used; discard independent local quotas.
            record['within_group'] = {key:value for key,value in record['within_group'].items() if key in ('status','n_samples','n_positive','AP','prevalence')}
            record['history_availability'] = {entity:float((valid_views['both_history'].loc[mask, f'{entity}_no_history_7d']==0).mean()) if mask.any() else None for entity in ('customer','terminal')}
        metrics = compact_metrics(frame)
        for i, budget in enumerate(metrics['budgets']):
            assert sum(r['global_budget_allocation'][i]['reviewed_in_group'] for r in grouping['groups']) == budget['K']
            assert sum(r['global_budget_allocation'][i]['frauds_detected_in_group'] for r in grouping['groups']) == budget['TP']
        conditions[name] = dict(metrics=metrics, amounts=evaluate_amounts(frame, validation, bins),
                                cold_start=grouping, train_seconds=seconds, score_seconds=score_seconds,
                                features=list(X.columns), parameters=model.named_steps['classifier'].get_params())
        print(f"Handbook {name}: AP={metrics['AP']:.6f}", flush=True)
    result = dict(dataset_id='handbook', metadata=extension_provenance(), evidence_kind='handbook_validation',
                  manifest=development['manifest'], feature_fingerprints=dict(train=_handbook_view_fingerprints(views), validation=_handbook_view_fingerprints(valid_views)),
                  feature_fingerprint_version=HANDBOOK_FINGERPRINT_VERSION,
                  fixed_reference='both_history; all four views retained, no selection', cold_start_rule='initial training entities, not development updates',
                  amount_bins=bins, conditions=conditions)
    write_json(output / 'handbook_b.json', result)
    write_json(results / 'tables/handbook_b.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12,4.5))
    names = list(conditions)
    for ax, field in zip(axes, ('AP','recall_at_k')):
        values = [conditions[n]['metrics']['AP'] if field=='AP' else next(b['recall_at_k'] for b in conditions[n]['metrics']['budgets'] if b['review_fraction']==.01) for n in names]
        ax.bar(range(len(names)), values)
        ax.set_xticks(range(len(names)), [n.replace('_','\n') for n in names], fontsize=8)
        ax.set(title=field, ylim=(0,1))
    fig.suptitle('Handbook validation - fixed weighted LR history ablation')
    fig.tight_layout()
    fig.savefig(results / 'figures/handbook_b_ablation.png', dpi=160)
    plt.close(fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--extensions", action="store_true", help="Run ULB E03/E05 only; no baseline reselection.")
    parser.add_argument("--handbook", action="store_true", help="Run Handbook E10/E13 on A's verified views.")
    parser.add_argument("--handbook-dir", type=Path, default=ROOT / "data/handbook/v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.extensions or args.handbook:
        if args.output is None or not args.output.resolve().is_relative_to((ROOT / "data").resolve()) or args.output.exists():
            parser.error("Extensions require a new --output directory inside data/")
        args.output = args.output.resolve()
        args.output.mkdir(parents=True)
        with threadpool_limits(limits=1):
            if args.extensions:
                run_ulb_extensions(args.data, args.output)
            if args.handbook:
                run_handbook_b(args.handbook_dir, args.output)
    else:
        run(args.data)
