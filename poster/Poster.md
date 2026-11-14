# Credit Card Fraud Detection under a Limited Review Budget

Group 20 · Content editing draft · 2026-11-14

**Validation preview.** Final A8 evidence, course layout dimensions and author
details remain outstanding; this is not the submission poster.

## Question

When only a small fraction of transactions can be reviewed, do anomaly detection
and classification find complementary fraud? Do short feature rules offer
stable diagnostic associations?

## One shared protocol

- ULB/Worldline data; fixed CSV SHA-256 and stable transaction IDs.
- Chronological train/validation/test: 170,235 / 56,745 / 56,746 transactions.
- Primary capacity: 1% of validation, exactly 568 reviews; 57 validation frauds.
- Rank by score descending, row_id ascending. Fit preprocessing on training.
- Select on development/validation; final test awaits the group G2 freeze.

## Main panel: same capacity, honest comparison

| Validation method | Reviews | Fraud found | False positives | Recall |
|---|---:|---:|---:|---:|
| Class-weighted logistic regression | 568 | 49 | 519 | 85.96% |
| Selected Isolation Forest | 568 | 24 | 544 | 42.11% |
| Selected quota endpoint: pure classifier | 568 | 49 | 519 | 85.96% |

All 24 IF detections are already in the classifier's top-568 list. The declared
quota/tie policy chooses the pure-classifier endpoint. At 2% there is one
anomaly-exclusive fraud, but the evaluated exactly-K policies still have no net
gain. **An exclusive discovery is not automatically an equal-budget improvement.**

Visual allocation: make the shared review-budget and exactly-K complementarity
plots the largest panels. Current source figures are C's validation artifacts;
replace them with C7 formal test figures only after A8.

## Reliability and diagnostic rules

The same K-means groups have different recall: classifier 13/14 versus 36/43,
IF 9/14 versus 15/43. A third group is empty. Hourly validation windows have
small fraud counts, so short-term variation requires cautious interpretation.

Training-only short-rule mining compares equal-width and equal-frequency bins.
The declared gates retain 15/38 and 21/25 candidates respectively. The chosen
equal-frequency candidate set contains 21 rules. One example covers 4,556
validation transactions including 39 frauds (confidence 0.856%, lift 8.52):
association alone does not make an efficient review list. Rules may overlap.

## Takeaway and limits

E04 now adds fixed-rule overlap and cumulative coverage: all 21 rules cover
56,055/56,745 transactions (98.78%), including all 57 frauds. They cover all
eight classifier misses but also most model false positives. Fifteen rules add
no new fraud at their position. A broad diagnostic union is not evidence of
better review efficiency. Use the two-panel validation cumulative figure as a
compact illustration; the original 21-rule test set remains unchanged.

C has now delivered validation amount/reference/cost/interval evidence: at K=568,
classifier fraud-amount coverage is 73.48%, IF 37.78%, Amount-descending 28.30%
(only two frauds). Classifier recall's paired row interval is [76.92%,94.65%];
it conditions on the fixed model/current sample, not all model-selection
uncertainty. Label-budget, importance and threshold-migration evidence still
awaits B/A. Formal test evidence for all extensions remains pending A8; no
second dataset is part of this round.

Validation supports the selected classifier at the stated capacity and shows
no combination gain under the examined policy. Final generalization and rule
coverage of prediction errors remain pending. Two-day data, anonymous features,
small fraud counts and development choices limit conclusions. Anomaly is not
fraud; association is not causality.

Evidence: B `55ee151`, C `a936f29`, D's rule manifest and common-environment
record. Dataset attribution: ULB/Worldline, Zenodo 7395559. Source and AIGC
records must be checked against the final report before export. D7 will set
final dimensions, typography, figure labels and PDF checks after G3.
