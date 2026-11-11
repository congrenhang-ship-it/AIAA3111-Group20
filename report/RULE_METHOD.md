# D rule module: methods and handoff (development draft)

Update 2026-11-11: the dependency-branch training/validation run is complete using
A's commit `26c1c32`. See D_VALIDATION_PROTOCOL.md for the predeclared selector
and D_VALIDATION_RESULTS.md for actual evidence. The selected features are Amount,
V17, V14, V12 and V3; the candidate freeze contains 21 equal-frequency rules.
This is pending A3 integration and group G2 confirmation; no test rules were scored.
The following original preparation notes describe the baseline before that run.

This is a methods draft, not a results section. A3 is not implemented in baseline
commit `5bde9fc`. No real feature set, validation choice or G2 rule set is frozen.

## Existing interface, now implemented for synthetic development

```python
from preprocessing import load_data, split_data
from pattern_mining import mine_rules, validate_rules

# Call only once A has delivered the unique split and fixed data provenance.
train, validation, test = split_data(load_data(data_path))
# feature_columns must be an explicit, recorded training-only choice (<=5).
binner, rules = mine_rules(train, feature_columns, bins=3, max_length=3,
                           binning="equal_frequency")
validation_metrics = validate_rules(validation, binner, rules)
# Compare equal_width with the SAME training-selected features.
# Freeze selected strategy, bins, limits and rule IDs at G2 before A uses test.
```

The two public signatures are unchanged. No changes were made to A/B/C files.
Only numpy/pandas are required; the small bounded Apriori implementation does
not add dependencies. This module does not define a competing common evaluator.

Return values:

- `binner`: JSON-compatible dictionary with schema version, feature list,
  method, requested/effective bins, finite interior cut points, observed training
  min/max, interval/missing policies and candidate limits.
- `rules`: deterministic rule IDs, feature/bin antecedents, `Class=1` consequent,
  training metrics and covered row IDs. IDs are local to a fitted candidate set:
  archive the matching binner, strategy and rules together, never merge on ID alone.
- `validate_rules`: same-order metric records with the same IDs and covered
  held-out row IDs. No fitting, selection or mutation occurs in validation.

## Transaction and search design

An item is `feature=bin_index`, bin indices start at zero. Use one or two distinct
feature items in an antecedent, with `Class=1` only as consequent; `max_length=3`
counts the consequent. `Class` and `row_id` cannot be supplied as feature inputs.
Feature choice is supplied explicitly and must use training information only.
The implementation supports Time, but the proposed small rule experiment should
focus on Amount and a recorded small subset of V1–V28; no subset is selected yet.

Equal-width bins use training min/max; equal-frequency bins use training
quantiles (numpy default linear interpolation). Repeated/endpoint quantiles are
removed; constant columns have one effective bin. This can produce fewer bins
than requested, which must be recorded, not corrected by random jitter.
Interior intervals are left-closed/right-open; a value equal to a cut goes into
the higher bin. Outer bins are unbounded so held-out extremes use the same end
bins, with no refit. Missing/nonfinite values fail explicitly; A must document
any upstream processing decision before real experiments.

Apriori first retains feature items covering at least 20 transactions, joins
frequent itemsets and prunes candidates with infrequent subsets. This is safe
because joint antecedent coverage cannot exceed subset coverage. Itemsets never
contain two bins of the same feature. Fraud-associated output additionally
requires at least 5 covered frauds and training lift strictly greater than 1.
At most 50 candidates survive, ranked by descending training lift, descending
fraud count, shorter length, then lexical items. These limits are development
defaults, not a statistical guarantee or an already selected final protocol.

## Metrics and zero-count policy

Let N be split size, F total fraud, A antecedent coverage and J covered fraud.

| Field | Definition |
|---|---|
| antecedent_support | A/N |
| support | J/N (joint support, not A/N) |
| confidence | J/A |
| lift | (J/A)/(F/N) |
| fraud_recall | J/F |
| kulczynski | ((J/A) + (J/F))/2 |
| normal_count | A − J |

Zero denominators are `None`/JSON null. If F=0, lift, fraud recall and Kulczynski
are not applicable. If A=0, confidence/lift/Kulczynski are not applicable, while
support is zero for a nonempty split. Empty splits retain zero counts with null
metrics and an explicit `empty` status. Neither NaN nor infinite lift is exported.

Hand example: N=12, F=4, A=6, J=4 gives support=1/3, confidence=2/3, lift=2,
Kulczynski=5/6, normal_count=2. These are synthetic arithmetic checks only.

## D2–D4 work after A3

1. Record A's data SHA-256, audit decisions, split sample/fraud counts and commit;
   call A's split exactly once. Do not construct a local alternative split.
2. Record the training-only feature selection and justification before inspecting
   validation labels. Compare only the two planned binning schemes initially,
   using three requested bins and rule length at most three.
3. Report every candidate's training/validation coverage, normal/fraud counts,
   all above metrics, change in confidence/lift and effective bins. Out-of-range
   feature counts should be included in the experiment summary for context.
4. Proposed validation reliability gate (not yet G2 frozen): require >=20 covered
   transactions, >=5 covered frauds and lift >1 on validation as well. Record all
   rejections and metrics. Do not lower thresholds just to retain a rule. If none
   qualify, the frozen rule set may be empty; report that finding.
5. Supply A a manifest with selected features, exact cut points, limits, surviving
   rule IDs/antecedents, validation criteria, commands, dependency versions and
   code/data provenance. Freeze the complete set; only then evaluate on test.

The candidate support gate is not a multiple-testing correction; searching many
rules and using small fraud counts can exaggerate lift. Validation describes
short-term stability; it does not establish business meaning or causality.

## Later error linkage / output handling

Join covered row IDs to C's common prediction table at the frozen review budget.
Separate covered true positives, false negatives and normal false positives;
do not select a new rule from test errors. Reuse C's stable ranking/K convention.
Rules are diagnostic associations, not a new risk-ranking method by default.

Keep full covered-ID lists local. Commit only aggregate metrics, frozen bins,
short antecedents and selected small case summaries. Do not commit raw data or
full prediction tables. Include split names and method/strategy in output names.

## Verification command (no dataset needed)

```bash
python -m unittest discover -s tests -p test_pattern_mining.py -v
```

Tests cover a hand-calculated example, an independent exhaustive short-itemset
oracle, both binning schemes, fixed held-out boundaries, input/ID integrity,
constant features, minimum coverage, no fraud, no matches and empty holdouts.
