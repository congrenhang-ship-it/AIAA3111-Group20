# Credit Card Fraud Detection under a Limited Review Budget

AIAA 3111 · Group 20 · Integrated working draft · 2026-11-16

**Editorial status:** methods and validation evidence only. Formal test results,
final error cases, Handbook semantic rules/mechanism analysis (D-N1), all authors'
final disclosures and G3 approval are outstanding.
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
capacity constraints and honest reporting of absent combination gains.
Controlled Handbook experiments further examine supervision, entity histories,
relations, delayed labels and incomplete context; PaySim provides a lightweight
external reference. Added context does not uniformly improve the fixed models.
Final test generalization remains to be assessed after the group freeze.

## 1. Research question and review task

A ranking is useful only in relation to the number of transactions that can be
reviewed. We therefore ask how supervised classification and anomaly detection
compare at identical review capacities, whether their fraud discoveries are
complementary, and whether performance varies across transaction groups and
relative-time windows. We also ask whether short discretized feature patterns
remain associated with fraud on later data. With semantic simulated data, we
also examine whether history and relationship features help at the same capacity,
and how label availability and incomplete context affect those findings.
Rules supplement the diagnostic analysis; they do not provide causal explanations of model decisions.

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
split fingerprints permit independent alignment checks. Static-model preprocessing is
fitted on the declared training records; validation/test retain natural class
prevalence. Rolling development models use only past records whose labels are
available at their scheduled update time.
The test counts identify the partition and are not performance measurements.

### Controlled context data and external reference

Handbook uses the pinned public transaction simulator, with 5,000 customers,
10,000 terminals, 60 days and fixed seeds. The first 30 days train static models,
the next 15 validate them, and the final 15 are reserved for testing.
PaySim uses the author's v2 release: a label-independent, seed-42 random sample
retains 635,359 of 6,362,620 source rows, then splits them chronologically without
splitting equal time steps. The three datasets are trained and evaluated separately.

| Dataset | Training | Validation | Reserved test | Fraud: train / validation / test |
|---|---:|---:|---:|---|
| ULB, after deduplication | 170,235 | 56,745 | 56,746 | 342 / 57 / 74 |
| Handbook, 60 days | 292,139 | 145,315 | 145,739 | 1,752 / 1,291 / 1,280 |
| PaySim, fixed approximately 10% sample | 379,109 | 127,377 | 128,873 | 326 / 64 / 413 |

Handbook supplies current amount/time-of-day encoding and strictly past
customer/terminal histories. Entity IDs identify records and groups but are not
model inputs; labels and simulator mechanism codes are excluded from features.
Simulator mechanisms support controlled diagnostics, not claims of real-world
causality or deployment generalization. PaySim uses only requested amount and
transaction type; balance fields, entity IDs and system decision markers are
excluded. Sources and split evidence are recorded in the [data handoff](../data/README.md),
[Handbook summary](../results/tables/handbook_a.json) and
[PaySim summary](../results/tables/paysim_a.json).

## 3. Methods and selection — B/C/D contributions

### Classification

We compare logistic regression (LR), a decision tree, Gaussian Naive Bayes and
a random forest on the shared chronological ULB split. StandardScaler is fitted
within the training pipeline for LR and Gaussian Naive Bayes; trees use unscaled
features. Balanced class weights and seeded 1:1 random undersampling are the two
additional LR strategies. Undersampling retains all minority training rows and
samples an equal number of majority rows without replacement. Validation keeps
its natural distribution. The classifier is selected by validation true positives
at 1% capacity, then AP and a deterministic method-name tie-break. Pipelines
exclude Class and row_id; thresholds do not substitute for top-K.

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

### ULB extensions and shared execution

The main validation entry now runs E01–E08: amount coverage/reference rankings,
paired uncertainty, label availability, rule overlap, permutation importance,
assumed costs and fixed-threshold diagnostics. The same fixed protocols and
shared split are used throughout; model selection and test evaluation remain
separate stages.

For E03, class-specific permutations define nested 5%, 10%, 25%, 50% and 100%
labeled subsets, using mask seeds 42/43/44 and model seed 42. Each class count is
rounded upward. Weighted LR preprocessing and fitting use only the labeled
subset; hidden labels never become normal labels. Stratification uses true
training labels, so this is a controlled supervision simulation. The fixed
IF256 reference sees all unlabeled training features, changing available records
as well as supervision. It is distinct from the selected IF4096 main comparator.
E05 uses validation AP, five permutations per original feature and seed 42, with
no refitting or subsequent feature removal. Error examples follow fixed rank-based
selection rules. The [B extension summary](../results/tables/b_extensions.json)
contains all 15 label conditions and all 150 permutation values.

For E08, each fixed model's threshold was its 568th-highest score on the
56,745-transaction validation segment. The thresholds were 0.804282 for the
class-weighted logistic regression and 0.531776 for the improved Isolation
Forest. Applying each threshold with an inclusive comparison flagged 568
transactions overall and found 49 and 24 of 57 frauds, respectively. The
validation diagnostic applies each threshold unchanged within fixed one-hour
windows from elapsed time zero. The first and last observed windows are partial.
See the [validation summary](../results/tables/a_team_validation.json) for the
full-precision thresholds, window counts and fixed-capacity comparison.

### Context, update schedules and external reference

On Handbook, B fits fixed weighted LR and C fits fixed IF separately on the same
four views: current transaction, customer history, terminal history and both
histories. IF uses 100 trees, max_samples=256 and seed 42. History queries use
[t-window,t), with all simultaneous transactions queried before insertion.
E12 adds only three past-seven-day relationship features: customer degree,
terminal degree and a new-pair flag. Two-history LR/IF are the fixed references;
view choice is not reselected from the ablation. E14 rebuilds visible history
with 0/10/30% missing events and seeds 42/43/44 while retaining current queries,
labels and the full-history-trained models. E13 defines four disjoint novelty
groups using the initial training customer/terminal sets.

E11 compares static LR with scheduled development updates at days 30/37/44,
a trailing 30-day window, and 0/1/7-day label delays. Updates see only labels
available at the decision time; final snapshots and validation-calibrated
thresholds are saved for subsequent frozen testing. PaySim runs only amount
ranking, weighted LR and fixed IF256 as a lightweight external check. Handbook
semantic-rule and mechanism analysis (E15) remains pending D-N1.

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

### Limited labels and model reliance (E03/E05)

At 5% labeled training data, mean AP is 0.3799, ranging from 0.2562 to 0.6259
across the three masks; at 25%, mean AP is 0.7389. Full-label AP is 0.7727.
The seed ranges describe sensitivity to sampled labels, not confidence intervals.
The fixed IF256 reference has AP 0.016546 and Recall@1% of 17.54%, and must not
be confused with the selected IF4096. Full-label weighting's one-fraud gain also
needs interpretation against the small number of validation positives.

Amount has the largest mean permutation AP decrease (0.580432), followed by
V2 and V7. This is reliance of the fixed model, not a causal effect or a business
interpretation of anonymous features. Fixed cases include normal transactions
at ranks 568/567 and 35/37, and missed frauds at ranks 1,405/1,740. Uncalibrated
scores do not guarantee individual correctness. The [combined validation panel](../results/figures/b_extensions.png)
and [full table](../results/tables/b_extensions.json) retain all conditions.

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

### Fixed-rule overlap and validation error linkage (E04)

Ordering the unchanged 21 rules by training lift/confidence/length/ID gives a
union of 56,055 validation transactions, including all 57 frauds. This covers
98.78% of the validation population and 55,998 normal transactions. Fifteen
rules add no new fraud at their position, and R020 adds no new transactions.
There are no identical-coverage pairs or pairs with Jaccard >=0.9, illustrating
that cumulative redundancy can occur without near-identical pairs. The compact
validation display uses R001/R002/R005/R006/R009, while all 21 remain fixed for
eventual test evaluation. The display omits a later novel fraud covered by R013.

The union covers all eight classifier false negatives and all 33 IF false
negatives, but also 516/519 classifier false positives and 538/544 IF false
positives. Given its broad scope, this is diagnostic association and not an
equal-budget improvement. Validation example source row 176049 is missed at
K=568 by both models (ranks 11,505/867); normal row 170937 is ranked 321 by IF
and 56,714 by the classifier. Both match R013/R016/R021. The shared rule match
does not identify a business or causal explanation.

Figure candidates: the validation Jaccard heatmap and two-panel cumulative
transactions/fraud figure in results/figures/d_validation_rule_*.png. Full pair
counts, increments, per-rule error counts and deterministic cases are in
results/tables/d_extensions.json. Do not sum overlapping rule fraud counts.

### Amount coverage, references and assumed costs (C E01/E06/E07)

C's training-only Amount boundaries are 21.94, 200.0 and 1008.6402000000035.
At the common validation K=568, the classifier covers 73.48% of fraud amount,
IF 37.78%, cluster distance 28.80%, and the fixed Amount-descending reference
28.30%. The latter finds only two frauds, compared with 49 for the classifier
and 24 for IF. Quantity and amount therefore answer different questions.
Classifier-covered fraud amount is 8,987.38 of 12,231.09; this is coverage of
transaction amounts, not actual money recovered. The classifier's precision
improvement over validation prevalence is 85.88-fold; this ratio does not
change the absolute precision of 8.63% or the number of false positives.

With assumed loss L=FP+c_FN*FN and c_FN=10/50/100, classifier losses at 0.1%
are 165/765/1515, at 0.5% 326/686/1136, at 1% 599/919/1319 and at 2%
1166/1486/1886. These are hypothetical units. No budget was reselected from
this table; the primary review capacity remains 1%. Higher review capacity
does not necessarily minimize a particular assumed loss.

### Paired uncertainty and limits (C E02)

C uses seed 42 for 500 paired row resamples and 200 paired 1800-second block
resamples, reporting empirical 2.5/97.5 percentiles. Per-replicate N determines
K; methods share the same sample and preserve repeated transaction instances.
All observed replicates contain fraud. Classifier Recall@1% is 0.8596 with
row interval [0.7692, 0.9465] and block interval [0.7954, 0.9153]; IF recall
is 0.4211 with intervals [0.3129, 0.5569] and [0.2496, 0.5714]. The paired
classifier-minus-IF recall difference is 0.4386, with corresponding intervals
[0.3056, 0.5641] and [0.3098, 0.5882]. The pure-classifier policy has exactly
zero difference from the classifier in every replicate, as expected from its
identical ranking. These intervals condition on chosen models/current data,
exclude full selection uncertainty and do not constitute significance tests
or proof of long-term stability. C's full evidence is c_extensions.json; its
three extension figures should form a compact panel during later typesetting.

### Handbook: history value, relations and incomplete context

The four static views share 145,315 validation transactions and 1,291 frauds;
every 1% list contains K=1,454 records.

| View | LR AP | LR TP@1% | IF AP | IF TP@1% |
|---|---:|---:|---:|---:|
| Current transaction | 0.255353 | 331 | 0.153913 | 324 |
| Customer history | 0.218989 | 348 | 0.051731 | 225 |
| Terminal history | 0.235494 | 325 | 0.019456 | 42 |
| Both histories | 0.208312 | 342 | 0.027708 | 124 |

History does not uniformly improve AP: customer history raises LR's budgeted
detection from 331 to 348, while both-history LR detects 342 with lower AP
than transaction-only LR. Both-history remains the predeclared reference.
Adding relationship features reduces LR detections from 342 to 334 and IF
detections from 124 to 82. These negative increments are retained.

With 30% missing history, fixed IF detects 74–82 frauds across seeds, compared
with 124 using full history. LR AP falls to 0.178961–0.195633, while its
TP@1% is 349–351 rather than 342. Whole-ranking quality and local review yield
therefore respond differently; this does not establish that losing data improves
the model. Of all validation transactions, 145,296 involve seen customers and
terminals; only 19 involve new customers and none are fraudulent. New-terminal
groups are empty, so this experiment cannot estimate cold-start fraud detection
or support demographic-fairness claims.

For the scheduled E11 comparison, pooled validation AP is 0.208312 for static
LR, 0.212765 for weekly zero-delay updates, 0.210970 with one-day delay, and
0.204373 with seven-day delay. These are development-time diagnostics using
only legally available past labels; test-time updating is not performed.
The [B context results](../results/tables/handbook_b.json),
[C ablation/stress results](../results/tables/handbook_c.json) and
[A rolling results](../results/tables/handbook_a.json) preserve the full conditions.
Data and partitions match A's handoff; B/C use the same feature definitions.
Full Handbook rule/mechanism diagnosis remains pending D-N1.

### PaySim lightweight external validation

The fixed sample has 127,377 validation records, 64 frauds and K=1,274 at 1%.

| Reference | AP | TP@1% |
|---|---:|---:|
| Amount descending | 0.002458 | 6 |
| Weighted LR | 0.002751 | 6 |
| IF256 | 0.002367 | 4 |

These low detection counts are retained rather than interpreted as external
confirmation of strong performance. This check uses limited decision-time
fields and simulated fraud. Different class prevalences and mechanisms prevent
a pooled AP across datasets. The [PaySim audit/results](../results/tables/paysim_a.json)
record sampling, exclusions and all budgets; held-out scores remain pending.

## 6. Final test and error analysis — pending A7/A8

No formal test scores are inserted into this draft. After the group freeze, A
must publish the common overall, group, combination, time-window and fixed-rule
tables. B/C/D then attach traceable cases and update this section together.
The new E04 validation cases have traceable row IDs and fixed-model ranks;
they do not substitute for the final cases or justify selection on test errors.

## 7. Limitations and validation-stage conclusion

ULB's short observation period, few frauds and one chronological split limit
generalization. Handbook's designed mechanisms, insufficient cold-start positives,
and the sampled PaySim reference further limit transfer to real deployments.
Validation choices, including supervised feature selection and
label-based quota selection, must be disclosed. Cluster stability is imperfect;
anomaly scores need not prioritize known fraud. Probabilities are not calibrated.
Rule count/lift gates are not multiple-testing corrections. Anonymous features
prevent concrete business narratives; observed associations are not causal.

The ULB evidence favors the selected classifier under the declared review
task and provides no same-budget combination gain. Context and relationship
features show negative as well as positive local changes on Handbook, and the
PaySim check does not establish strong external detection. Rules add associations but
do not yet explain which final-test errors they cover. These are validation
conclusions, awaiting frozen held-out evaluation.

## Sources and disclosures — outside planned main-text page count

- A: [data provenance/splits](../data/README.md), [shared validation](../results/tables/a_team_validation.json), [Handbook/E11](../results/tables/handbook_a.json), [PaySim](../results/tables/paysim_a.json).
- B: [classification methods, validation prose and review notes](B_CLASSIFICATION.md), [ULB extensions](../results/tables/b_extensions.json), [Handbook classification](../results/tables/handbook_b.json).
- C: [methods, caveats and figures](C_IMPROVEMENTS.md), [ULB reliability](../results/tables/c_extensions.json), [Handbook ablation/stress](../results/tables/handbook_c.json).
- D: [methods, protocol and validation findings](D_VALIDATION_RESULTS.md), [rule manifest](d_validation/manifest.json), [common-environment freeze](../results/tables/d_freeze.json), [baseline reproduction](../results/tables/d_g1_reproduction.json).
- Data attribution: [ULB/Worldline Zenodo archive](https://zenodo.org/records/7395559), [Fraud Detection Handbook simulator](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html), [author's PaySim release](https://www.kaggle.com/datasets/ealaxi/paysim1). Pinned versions and licenses are recorded in the linked data/results handoffs.

Figure plan: C's validation PR/budget figure, exactly-K complementarity figure
and group figure; final versions must come from A8/C7 with split-labelled
captions. Do not use current validation figures as final test evidence.

A/B/C/D source records disclose Codex assistance in implementation/checks and
analysis. This integration used Codex to compile their existing material; authors
must review their technical content and confirm the complete final disclosures.
No member names, student IDs or unreported collaboration details are invented.
