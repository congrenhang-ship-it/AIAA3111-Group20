# Credit Card Fraud Detection under a Limited Review Budget

AIAA 3111 · Group 20 · Integrated working draft · 2026-11-13

**Editorial status:** methods and validation evidence only. Formal test results,
final error cases, all authors' final disclosures and G3 approval are outstanding.
This is not the final submission. The planned typeset main text remains 6–10 pages;
pagination is not established by this Markdown draft.

## Abstract — validation-stage draft

Credit-card fraud detection is evaluated as a ranking task with limited human
review capacity. Using the ULB transaction dataset and a shared chronological
split, we compare four classification baselines, bounded class-imbalance
strategies, K-means distance and Isolation Forest. All methods use the same
validation transactions and deterministic top-K policy. At the primary 1%
budget, class-weighted logistic regression detects 49 of 57 validation frauds
from 568 reviews, while the selected Isolation Forest detects 24. The latter
detections are all covered by the classifier's list, so the predeclared quota
search selects the pure-classifier endpoint. Training-discovered short rules
provide additional diagnostic associations; 21 equal-frequency candidates pass
the declared validation count/lift gates. These observations motivate explicit
capacity constraints and honest reporting of absent combination gains. Final
test generalization remains to be assessed after the group configuration freeze.

## 1. Research question and review task

A ranking is useful only in relation to the number of transactions that can be
reviewed. We therefore ask how supervised classification and anomaly detection
compare at identical review capacities, whether their fraud discoveries are
complementary, and whether performance varies across transaction groups and
relative-time windows. We also ask whether short discretized feature patterns
remain associated with fraud on later data. Rules supplement the diagnostic
analysis; they do not provide causal explanations of model decisions.

The narrative follows the shared review task rather than treating each algorithm
as a separate application. Positive findings, failed changes and a combination
that yields no additional detected fraud are all retained in the evidence.

## 2. Data and experimental boundaries — A contribution

The pinned ULB/Worldline CSV is the version archived in Zenodo record 7395559.
Its SHA-256 is `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.
There are 284,807 original transactions, including 492 frauds. Time is elapsed
seconds; V1–V28 are anonymous PCA features; Amount is a monetary quantity. The
data do not support customer-history or local-clock interpretations.

A's audit finds no missing values. Exact duplicates are defined using all 31
original fields, including Class, retaining the earliest source row. This removes
1,081 additional rows, including 19 frauds; it is a reproducible record policy,
not proof that identical anonymized records represent the same real transaction.
Extreme amounts and fraud labels alone are not reasons for removing records.

| Partition | Transactions | Fraud | Relative Time range, seconds |
|---|---:|---:|---:|
| Training | 170,235 | 342 | 0–120,392 |
| Validation | 56,745 | 57 | 120,393–145,233 |
| Held-out test | 56,746 | 74 | 145,234–172,792 |

The approximately 60/20/20 split is chronological after cleaning. Equal boundary
timestamps stay together on the later side. Stable original row IDs and ordered
split fingerprints permit independent alignment checks. All preprocessing is
fitted within training workflows; validation/test retain natural class prevalence.
The test counts identify the partition and are not performance measurements.

## 3. Methods and selection — B/C/D contributions

### Classification

B implements logistic regression, decision tree, Gaussian Naive Bayes and random
forest. A bounded imbalance comparison adds class weighting and random
undersampling for logistic regression. The original six candidates and selection
policy are retained: maximize validation true positives at 1%, then AP, then
method name. Feature preprocessing is part of the fitted pipeline and excludes
Class and row_id. Probability thresholds are not used to substitute for top-K.

### Groups and anomaly scores

C fits K-means and Isolation Forest to mixed training features without filtering
normal transactions by their labels. Scores increase with suspicion; neither
cluster ID nor an anomaly score is a fraud probability. Training-only K-means
structure/stability analysis retains k=3. Candidate improvements are selected
using a further chronological training-inner development split and declared
tie rules, then fitted on the original full training partition.

The retained Isolation Forest uses StandardScaler, 100 trees, max_samples=4096,
contamination=auto, seed=42 and n_jobs=1. Formal validation had already been
viewed before C's later protocol, which C explicitly discloses; the inner split
does not turn the formal validation into an untouched independent dataset.

### Diagnostic association rules

D includes Amount and selects four V features by training-only absolute Pearson
correlation with Class: V17, V14, V12 and V3. This is supervised feature selection;
Class is used only as the consequent in the resulting rules. Equal-width and
equal-frequency discretization each request three bins. Apriori considers at
most two distinct-feature antecedent items, plus consequent Class=1.

Training candidates require antecedent coverage >=20, covered fraud >=5 and
lift >1, with at most 50 per scheme. Fixed training bins are applied unchanged
to validation, including unbounded outer intervals for extreme values. The same
count/lift gate is applied there. Zero-denominator metrics are not applicable.
The predeclared scheme criterion is the number of passing diagnostic rules,
with equal-frequency preferred on ties; it is not a top-K fraud-yield objective.

## 4. Common evaluation and fair capacity comparison — C contribution

AP summarizes precision–recall ranking behavior. Budgeted outcomes are reported
at 0.1%, 0.5%, 1% and 2%: K=ceil(N×budget), ordered by score descending and row_id
ascending. For each list, TP+FP must equal K; precision is TP/K and recall TP/F.
One extra fraud changes validation recall by about 1.75 percentage points.
An expected random-review reference has precision equal to prevalence and
expected TP=K×F/N; it is an expectation, not an observed model result.

Combination lists scan the declared quota order, remove duplicates and refill
to exactly K. A union that doubles capacity is not a valid equal-budget gain.
Global-list allocation across clusters is reported separately from budgets
allocated within each cluster. AP/recall are not applicable for no-fraud groups;
empty groups/windows remain explicitly identified.

## 5. Validation evidence and implications

### Classification and imbalance strategies

| B original validation run | AP | TP at K=568 | FP |
|---|---:|---:|---:|
| Logistic regression, original | 0.642429 | 48 | 520 |
| Decision tree | 0.301922 | 38 | 530 |
| Gaussian NB | 0.033763 | 20 | 548 |
| Random forest | 0.757048 | 47 | 521 |
| Logistic regression, class weight | 0.772681 | 49 | 519 |
| Logistic regression, undersampling | 0.157422 | 48 | 520 |

The selected class-weighted logistic regression detects one more fraud than its
unweighted baseline. Random forest has higher AP than unweighted regression but
fewer frauds at the primary capacity. Undersampling does not improve primary
budget yield and substantially reduces AP. None of these findings establishes
that a strategy universally helps. In C's common-environment reproduction,
candidate TP values and the selected model agree; random forest AP is 0.755615
rather than B's original 0.757048. Preserve that version difference explicitly.

### Equal-budget anomaly comparison and complementarity

| Budget | K | Selected classifier TP | Selected IF TP | Frozen endpoint TP |
|---|---:|---:|---:|---:|
| 0.1% | 57 | 42 | 0 | 42 |
| 0.5% | 284 | 48 | 8 | 48 |
| 1% | 568 | 49 | 24 | 49 |
| 2% | 1,135 | 49 | 35 | 49 |

At 1%, the classifier reaches recall 85.96% and precision 8.63%, with eight
frauds missed and 519 normal transactions reviewed. IF detects 24 frauds with
544 false positives. All 24 are already in the classifier's list. Classification
quota candidates 0/25/50/75/100% detect 24/45/48/49/49 frauds; the fixed tie rule
prefers 100% classifier capacity. The chosen endpoint has no mixture benefit.

At 2%, IF finds one fraud outside the classifier list, but the examined equal-K
policies do not increase total yield: gaining an exclusive case can trade away
a classifier-exclusive detection. This distinction between exclusivity and net
benefit is central to the research question.

### Group and time reliability

Using the same global 568-review lists, cluster 0 contains 9,596 transactions and
14 frauds: classifier recall is 13/14, IF recall 9/14. Cluster 1 contains 47,149
transactions and 43 frauds: corresponding recall is 36/43 and 15/43. Cluster 2
is empty on validation. Cluster assignments are transaction groups, not labels.
Hourly validation windows contain only 1–12 frauds and incomplete edge windows;
these are exploratory short-term diagnostics. A must freeze the final relative
window origin/length and publish formal window evidence after G2.

### Rules as diagnostic evidence

Equal-width yields 38 training candidates, of which 15 pass the validation gate;
equal-frequency yields 25, of which 21 pass. The original rule criterion selects
the latter. Exact boundaries and all candidate results are retained in D's
manifest/table, including rejected rules.

Equal-frequency R001 combines low-bin V12 and low-bin V17. It covers 11,722
training transactions with 269 frauds and lift 11.42; validation coverage is
4,556 transactions with 39 frauds, confidence 0.856% and lift 8.52. This shows a
persistent association but a broad review burden. Some equal-width rules are
more concentrated despite fewer passing candidates: validation R023 covers
46 transactions with 31 frauds. Rule-count selection must not be presented as
proof of greater review efficiency. Overlapping rule counts are never added.

## 6. Final test and error analysis — pending A7/A8

No formal test scores are inserted into this draft. After the group freeze, A
must publish the common overall, group, combination, time-window and fixed-rule
tables. B/C/D then attach traceable cases and update this section together.
Published C validation cases may motivate the questions but cannot substitute
for final test cases or provide missing row IDs for cross-method joins.

## 7. Limitations and validation-stage conclusion

The short observation period, few frauds and one chronological split limit
generalization. Validation choices, including supervised feature selection and
label-based quota selection, must be disclosed. Cluster stability is imperfect;
anomaly scores need not prioritize known fraud. Probabilities are not calibrated.
Rule count/lift gates are not multiple-testing corrections. Anonymous features
prevent concrete business narratives; observed associations are not causal.

The current evidence favors the selected classifier under the declared review
task and provides no same-budget combination gain. Rules add associations but
do not yet explain which final-test errors they cover. These are validation
conclusions, awaiting frozen held-out evaluation.

## Sources and disclosures — outside planned main-text page count

- A: [data provenance and split](https://github.com/congrenhang-ship-it/AIAA3111-Group20/blob/26c1c32/data/README.md).
- B: [classification evidence](https://github.com/congrenhang-ship-it/AIAA3111-Group20/blob/55ee151/docs/B_VALIDATION_RESULTS.md).
- C: [methods, common-environment results, caveats and figures](https://github.com/congrenhang-ship-it/AIAA3111-Group20/blob/a936f29/report/C_IMPROVEMENTS.md).
- D: D_VALIDATION_PROTOCOL.md, D_VALIDATION_RESULTS.md and d_validation/; current
  common-environment handoff is recorded separately in D_COMMON_ENVIRONMENT.md.
- Dataset: [Zenodo archive](https://zenodo.org/records/7395559), originally
  [ULB/Worldline](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Figure plan: C's validation PR/budget figure, exactly-K complementarity figure
and group figure; final versions must come from A8/C7 with split-labelled
captions. Do not use current validation figures as final test evidence.

A/B/C/D source records disclose Codex assistance in implementation/checks and
analysis. This integration used Codex to compile their existing material; authors
must review their technical content and confirm the complete final disclosures.
No member names, student IDs or unreported collaboration details are invented.
