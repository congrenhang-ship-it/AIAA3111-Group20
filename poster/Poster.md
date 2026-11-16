# Credit Card Fraud Detection under a Limited Review Budget

Group 20 · Content editing draft · 2026-11-16

**Validation preview.** Final A8 evidence, course layout dimensions and author
details and Handbook rule/mechanism analysis remain outstanding; this is not
the submission poster.

## Question

When only a small fraction of transactions can be reviewed, do anomaly detection
and classification find complementary fraud? Do short feature rules offer
stable diagnostic associations? How do labels and customer/terminal context
affect detection at the same review capacity?

## Shared review task and three datasets

- ULB is the main real-data comparison; fixed CSV SHA-256 and stable IDs.
- ULB train/validation/test: 170,235 / 56,745 / 56,746 transactions.
- Handbook: 583,193 simulated transactions over 60 days, split 30/15/15 days.
- PaySim: fixed approximately 10% sample, 635,359 transactions; lightweight reference.
- Primary capacity: 1%; ULB validation has K=568 and 57 frauds.
- Datasets are modeled separately; histories use only strictly earlier transactions.
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
uncertainty.

## Labels, model reliance and contextual validation

E03/E05 and E08 are now delivered and included in the shared ULB validation
entry. With 5% labeled training data, mean AP is 0.3799 (three-seed range
0.2562–0.6259), compared with full-label AP 0.7727. Amount has the largest
permutation AP decrease (0.580432); this is fixed-model reliance, not causality.
The label experiment's IF256 reference is distinct from the selected IF4096.
Fixed validation thresholds show varying hourly review rates; they are
validation diagnostics, not independent held-out results.

On Handbook, both-history LR detects 342/1,291 frauds at K=1,454, versus
331 for transaction-only LR; AP falls from 0.255353 to 0.208312. Adding
relationship features reduces two-history LR detections to 334 and IF from
124 to 82. History therefore does not uniformly improve the fixed models.
With 30% missing history, fixed IF detects 74–82 frauds across seeds. The
19 new-customer transactions have no frauds, so cold-start detection is not
estimable. Rolling/label-delay and full stress evidence are available in the
Handbook summaries; semantic-rule/mechanism analysis remains pending D-N1.

PaySim's K=1,274 lists detect 6/64 frauds for weighted LR and amount ranking,
and 4/64 for IF256. This limited-field simulated reference does not establish
strong real-world transfer. All findings shown here are validation evidence;
formal test results for the three data lines remain pending A7/A8.

Validation supports the selected classifier at the stated capacity and shows
no combination gain under the examined policy. Final generalization and rule
coverage of prediction errors remain pending. ULB's two-day observation, anonymous
features, small fraud counts, simulated mechanisms and development choices
limit conclusions. Anomaly is not
fraud; association is not causality.

Evidence: [shared ULB validation](../results/tables/a_team_validation.json),
[B labels/importance](../results/tables/b_extensions.json),
[Handbook A](../results/tables/handbook_a.json),
[B](../results/tables/handbook_b.json), [C](../results/tables/handbook_c.json),
[PaySim](../results/tables/paysim_a.json), and
[D rule findings](../report/D_VALIDATION_RESULTS.md). B's report and presentation
material from b73df08 is integrated in main. Attribution: ULB/Worldline,
Fraud Detection Handbook and the author's PaySim release, with pinned sources
and licenses in [data/README.md](../data/README.md). Source and AIGC records
remain part of the final report review. D7 sets dimensions, typography, figure
labels and PDF checks after G3.
