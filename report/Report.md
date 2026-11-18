---
documentclass: article
fontsize: 11pt
papersize: a4
geometry:
  - margin=20mm
mainfont: texgyretermes
mainfontoptions:
  - Extension=.otf
  - UprightFont=*-regular
  - BoldFont=*-bold
  - ItalicFont=*-italic
  - BoldItalicFont=*-bolditalic
sansfont: texgyreheros-regular.otf
monofont: lmmono10-regular.otf
mathfont: latinmodern-math.otf
linestretch: 1.04
colorlinks: true
linkcolor: black
urlcolor: black
header-includes: |
  \usepackage{float}
  \floatplacement{figure}{H}
  \usepackage{caption}
  \captionsetup{font=small,labelfont=bf}
  \setlength{\parskip}{4pt}
  \setlength{\emergencystretch}{3em}
  \setcounter{secnumdepth}{0}
---

\begin{center}
{\Large\bfseries Credit Card Fraud Detection under a Limited Review Budget\par}
\vspace{4pt}
AIAA 3111 Course Project · Group 20
\end{center}

| Group member | Student ID |
|---|---|
| __________________________ | __________________ |
| __________________________ | __________________ |
| __________________________ | __________________ |
| __________________________ | __________________ |

**Abstract.** We study fraud detection as a ranking problem with limited review capacity. On the chronological ULB test set, the preselected weighted logistic regression detects 64 of 74 frauds from 568 reviews; the selected Isolation Forest detects 41. We then examine label availability, transaction amounts, uncertainty, time, entity histories and interpretable rules. Historical and relationship features reduce performance in the fixed Handbook models, while missing history affects the two model families differently. A PaySim comparison further shows that AP and fixed-budget detection can rank methods differently. The results connect model performance to review capacity, available information and the structure of the data.

# 1. Introduction

Fraud detection systems produce more alerts than reviewers can investigate. A useful comparison therefore needs an explicit capacity: detecting more fraud by reviewing more transactions is a different decision from improving detection at the same workload. We use a common top-K policy to compare supervised classifiers, clustering distances and Isolation Forest.

Our project asks four questions: which methods work at a fixed budget; whether their discoveries complement one another; how labels and contextual information change performance; and how rules and cases explain the results. ULB provides the real anonymous-data experiment, the Handbook simulator provides identifiable histories and mechanisms, and PaySim provides a separate transaction domain [1–4].

The study contributes an equal-capacity comparison, controlled changes to available information, and a rule-based explanation of model errors. It applies association mining, classifier evaluation, cluster validation and richer-data feature engineering from Lectures 3–7 [8–12].

# 2. Problem formulation

Each transaction has features $x_i$, a label $y_i\in\{0,1\}$ and a stable row ID. A method returns a risk score $s_i$. For $N$ transactions and budget $q$, we review $K=\lceil Nq\rceil$ records, ordered by decreasing score and then increasing row ID. The primary budget is 1%; 0.1%, 0.5% and 2% show the workload trade-off. Outputs comprise ranked reviews, comparative metrics, group/time diagnostics and short fraud-associated rules.

\clearpage

# 3. Methodology

## 3.1 Data acquisition and preprocessing

We use the fixed ULB/Worldline CSV archived on Zenodo [1]. It contains 284,807 transactions and 492 frauds. Time records elapsed seconds; V1–V28 are anonymous PCA components. The data have no missing values. Exact duplicates across the 31 original fields are merged, retaining the first source row: 1,081 extra records, including 19 frauds, are removed. This record-based policy produces 283,726 transactions and 473 frauds. Extreme amounts are retained.

Transactions are sorted by Time and original row ID. Boundaries target 60/20/20 by row count; equal timestamps enter the later partition together. The ULB ranges are 0–120,392 seconds for training, 120,393–145,233 for validation, and 145,234–172,792 for test. The 30 numerical inputs exclude the label and row ID. Scaling is fitted within each model's training data.

**Table 1. Chronological partitions; each cell is transactions / frauds.**

| Dataset | Training | Validation | Test |
|---|---:|---:|---:|
| ULB, cleaned | 170,235 / 342 | 56,745 / 57 | 56,746 / 74 |
| Handbook | 292,139 / 1,752 | 145,315 / 1,291 | 145,739 / 1,280 |
| PaySim sample | 379,109 / 326 | 127,377 / 64 | 128,873 / 413 |

The Handbook data are generated with the pinned public simulator [2]: 5,000 customers, 10,000 terminals, 60 days and fixed seeds. Days 0–29 train the static models, days 30–44 support development, and days 45–59 form the test. Current features are Amount and sine/cosine time encodings. Customer and terminal histories use preceding 1-day and 7-day counts, mean amounts and availability indicators; customer features also compare the current amount with previous spending. The four views have 3, 13, 9 and 19 features. Histories use $[t-w,t)$, so transactions at the same timestamp cannot enter one another's history. IDs support joins and grouping; they are excluded from model features.

PaySim is a synthetic mobile-money dataset [3,4]. We stream the author's v2 ZIP and retain a fixed, label-independent approximately 10% sample: 635,359 of 6,362,620 source records. Time steps remain intact across the chronological split. Models use requested Amount and one-hot transaction type; balance fields, IDs and the system flag are excluded. The three datasets are fitted and evaluated separately.

## 3.2 Experimental sequence

The pipeline is acquisition and audit → chronological partitioning → training and bounded development → frozen models and policies → test scoring → interpretation. Configuration files record the source data, seeds, parameters, feature views, quotas, rule bins and thresholds. Static models keep their original training fit. Development-only rolling models are saved before testing; test transactions update past-event context, while labels remain outside model fitting.

\clearpage

## 3.3 Classification, clustering and anomaly detection

**Classification.** Following Lecture 5 [10], LR uses $p_i=\sigma(\beta^\top z_i+b)$ and regularized log loss. Balanced weights $w_c=N/(2N_c)$ raise the contribution of the rare fraud class. Gaussian naive Bayes assumes conditional independence and Gaussian feature likelihoods. Trees choose splits by Gini reduction, $G=1-\sum_c p_c^2$; the 100-tree random forest combines bootstrap sampling with random feature subsets. LR and naive Bayes use StandardScaler; trees use unscaled inputs. Original, weighted and 1:1 undersampled LR compare ordinary fitting, cost-sensitive learning and data-level balancing. Undersampling retains 342 frauds and 342 normal records, fitting its scaler on those 684 rows. Seeds are 42. Validation TP at 1%, then AP, then method name selects weighted LR.

**Clustering and anomaly scores.** K-means minimizes $\mathrm{SSE}=\sum_i\|z_i-\mu_{c_i}\|_2^2$ on standardized features. It ranks distance to the assigned centroid. Because SSE favors larger k, training silhouette selects $k=3$ from 3/5/8 [12]. Ten initializations reduce local-optimum sensitivity. Isolation Forest ranks $-\mathrm{score\_samples}(x)$: shorter isolation paths imply greater abnormality [6]. A bounded second round compares six K-means representations with raw and within-cluster relative distances, and eight IF settings. An inner chronological 80/20 training split with seeds 42/43/44 selects family representatives by mean TP at 1%, then AP. The retained IF has 100 trees and 4,096 training samples per tree. All 20 scores remain in the test output.

**Review combinations.** Quotas are 0/25/50/75/100%. The first ranking contributes $\lfloor K\alpha\rfloor$ transactions; the second fills the list while skipping duplicates. Inner training data choose the anomaly-pair quota. Validation chooses the classifier/IF quota, preferring a pure method on a tie. The frozen policies are pure IF for the anomaly pair and pure weighted LR for the classifier comparison.

## 3.4 Context, rules and evaluation

Handbook applies the richer-data approach in Lecture 6 [11]: summarize a customer’s transaction sequence into 1/7-day features, extract degree and new-pair features from customer–terminal edges, then fit the same LR and IF256. Development LR updates on days 30/37/44 use a 30-day sliding window and 0/1/7-day label delays [5]. Saved snapshots stay fixed in test. History stress tests omit 0/10/30% of source events from later histories with seeds 42/43/44; current transactions and labels stay present.

Each transaction contains one bin item per selected feature. Apriori joins antecedents only when their subsets cover at least 20 rows: support is downward-closed, while confidence and lift are assessed after counting [8]. We compare three-bin equal-width/equal-frequency discretization and at most two antecedents followed by Class=1. Rules require coverage at least 20, at least 5 frauds and lift above 1 in both training and validation. The scheme retaining more rules wins; ties prefer equal-frequency. ULB uses Amount and four V features chosen by a training-correlation filter [11]; Handbook uses amount, relative amount and 7-day counts. The retained sets contain 21 and 19 rules.

With $F=\sum_i y_i$, evaluation uses $P@K=TP/K$, $R@K=TP/F$, non-interpolated AP [7], fraud-amount coverage and loss $L=FP+rFN$ for $r=10,50,100$. We use 500 paired row draws and 200 draws of 30-minute blocks; intervals condition on the fitted models. For antecedent count A and covered-fraud count J, rule support is $J/N$, confidence $J/A$, lift $JN/(AF)$ and Kulczynski $\tfrac12(J/A+J/F)$ [8,9]. Overlapping coverage is counted once.

\clearpage

# 4. Experiments and results

## 4.1 ULB: detection at the same review capacity

**Table 2. Formal ULB test, 56,746 transactions and 74 frauds; K=568.**

| Method | AP | TP | FP | Recall (%) | Precision (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LR | 0.6909 | 64 | 504 | 86.49 | 11.27 |
| Decision tree | 0.3012 | 45 | 523 | 60.81 | 7.92 |
| Gaussian NB | 0.0578 | 44 | 524 | 59.46 | 7.75 |
| Random forest | 0.7845 | 64 | 504 | 86.49 | 11.27 |
| Weighted LR (selected) | 0.7475 | 64 | 504 | 86.49 | 11.27 |
| Undersampled LR | 0.3545 | 63 | 505 | 85.14 | 11.09 |
| Cluster distance | 0.0460 | 37 | 531 | 50.00 | 6.51 |
| IF4096 (selected) | 0.0660 | 41 | 527 | 55.41 | 7.22 |

Weighted LR detects 64 frauds, leaving 10 missed frauds and 504 normal transactions in the review list. IF4096 detects 41; centroid distance detects 37. Relative to the original IF256 test result of 26 detections, the frozen larger-sample IF gains 15 at this budget. Classification retains the larger advantage: the weighted LR recall is 86.49%, compared with 55.41% for IF4096.

Random forest has the highest test AP in this bounded comparison, but ties weighted LR in TP at 1%. We retain the classifier selected before test. The original LR also reaches 64 detections; class weighting improves AP from 0.6909 to 0.7475 without improving TP at this capacity. Undersampling loses one detection and substantially lowers AP. Thus the imbalance strategy affects the ranking beyond the single operating point.

![ULB test recall and precision across the four fixed budgets. The methods and models are frozen; each curve uses the same transactions at each budget.](../results/figures/report_budget.pdf){width=100%}

The smallest budget separates the methods sharply: weighted LR finds 50 frauds within 57 reviews, whereas IF4096 finds none. At 0.5%, LR already finds 63; increasing to 1% adds one fraud at 284 additional reviews. An always-normal classifier has 99.87% accuracy and zero recall. Following Lecture 5’s positive-class evaluation [10], $F_1=2TP/(K+F)$ gives 0.1994 for LR and 0.1277 for IF at K=568; with the same K and F, this is the TP ordering. A random list has only 0.74 expected frauds.

\clearpage

## 4.2 Complementarity, groups and time

At 1%, the classifier and IF lists share 100 transactions and 40 detected frauds. The classifier contributes 24 additional frauds and IF contributes one. Their union would detect 65 frauds while reviewing 1,036 transactions. The frozen 100% classifier quota reviews 568 and detects 64. This distinguishes useful complementary information from an improvement already achieved under the same capacity. The anomaly-only policy likewise remains the preselected pure IF endpoint.

K-means groups provide a second view of the ranking. Table 3 allocates each method's single global list across the same groups; its review counts sum to 568 for each method.

**Table 3. ULB test groups and allocation of the global 1% review list.**

| Cluster | Rows | Frauds | LR reviews / TP | IF reviews / TP |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8411 | 16 | 54 / 15 | 208 / 13 |
| 1 | 48335 | 58 | 514 / 49 | 360 / 28 |
| 2 | 0 | 0 | 0 / 0 | 0 / 0 |

Training k=3 had mean silhouette 0.1537 and seed-agreement ARI 0.5558 on the fixed 1,000-row sample; k=5 gave 0.0669 and 0.1530. Silhouette measures cohesion/separation, while this ARI measures agreement between seeded partitions [12]. These favor k=3 within the tested choices, with limited separation. In test cluster 0, LR finds 15/16 frauds in 54 reviews versus IF’s 13 in 208; in cluster 1, it finds 49/58 versus 28. The third test group is empty. Scaling also changes the ranking: raw RobustScaler distance detects 10 frauds at 1%, versus 37 with StandardScaler [12]. These groups supply geometric diagnostics, while fraud detection remains a separate evaluation.

A separate question is how many alerts a fixed validation threshold produces later. Table 4 applies the original thresholds without changing them.

**Table 4. Frozen validation thresholds on the ULB test segment.**

| Method | Frozen threshold | Reviews | TP | Review rate (%) |
| --- | ---: | ---: | ---: | ---: |
| Weighted LR | 0.8043 | 453 | 63 | 0.80 |
| IF4096 | 0.5318 | 374 | 36 | 0.66 |

The LR threshold produces 453 reviews and detects 63 frauds, compared with 568 and 64 under test top-K. The IF threshold produces 374 reviews and detects 36, compared with 568 and 41. Both review rates fall below 1%. A score threshold therefore carries a different workload interpretation from a capacity rule, even when calibrated to 1% on validation.

Hourly diagnostics use the same relative-time origin and frozen models. Each window has its own top-K capacity; the whole-segment list allocates reviews across hours. The final ULB segment spans fewer than eight hours, so the windows describe short-run variation.

\clearpage

## 4.3 Label availability and model interpretation

Lecture 6 distinguishes learning settings by which labels are available [11]. Here LR is supervised training on reduced labeled subsets, with a separate unlabeled-data IF reference. Within each training class, random permutations give nested 5/10/25/50/100% subsets using ceiling counts. The masks use the original labels to control class composition; hidden labels are excluded from fitting. LR sees its labeled subset, while fixed IF256 sees all training features. Table 5 retains every predeclared condition.

**Table 5. ULB label experiment on the formal test set; three mask seeds.**

| Labels visible | Mean test AP | Three-seed AP range | TP range at 1% |
| --- | ---: | ---: | ---: |
| 5% | 0.5370 | 0.4397–0.7059 | 55–58 |
| 10% | 0.5958 | 0.5339–0.6786 | 55–60 |
| 25% | 0.7744 | 0.7613–0.7814 | 60–63 |
| 50% | 0.7792 | 0.7524–0.7928 | 63–64 |
| 100% | 0.7475 | 0.7475–0.7475 | 64–64 |

![ULB formal test: label-availability curves and the fixed LR's eight largest permutation effects. Shading is the three-seed range; error bars on importance are standard deviations over five permutations.](../results/figures/report_supervision.pdf){width=100%}

The 5% setting has an AP range of 0.4397–0.7059, showing sensitivity to the few sampled frauds. By 25%, all three fits have AP above 0.76, and TP ranges from 60 to 63. Full labels recover 64 frauds, while the mean AP at 25% and 50% exceeds the full-label fit on this segment. More labels increase TP here, while AP is not monotonic across the fitted subsets.

Amount has the largest permutation effect: shuffling it lowers test AP by 0.3695 on average; V14 and V2 follow at approximately 0.1503 and 0.0879. This identifies the fitted model's dependence on inputs. The deterministic error samples include a false positive at rank 568 (score 0.7573), a missed fraud at rank 783 (0.6875), and a normal transaction ranked third (score approximately 1.0000). Raising capacity could include the rank-783 fraud, while the high-ranked false positive remains a confident error. Scores are used for ranking; this experiment does not calibrate them as event probabilities.

\clearpage

## 4.4 Fraud counts, amounts, costs and uncertainty

At 1%, weighted LR covers 86.49% of fraud transactions and 69.59% of their amount. IF4096 covers 55.41% and 47.51%; cluster distance covers 50.00% and 10.71%. The descending-Amount reference finds only 3 frauds but covers 46.56% of fraud amount. The test fraud amount totals 7,727.67 units. Amount coverage therefore changes the interpretation of a count-based ranking without replacing the transaction-review task.

![ULB formal test: count and amount coverage at 1%, and loss across budgets for the fixed FN:FP cost ratio of 50:1.](../results/figures/report_amount.pdf){width=100%}

Under the 50:1 cost scenario, weighted LR has loss 771 at a 0.5% budget and 1,004 at 1%. The larger list gains one detected fraud but adds many normal reviews. With a different assumed error cost, the balance changes. These calculations compare the declared scenarios; the primary experimental budget stays at 1%. Fraud amount is a transaction quantity rather than a measured recovery amount.

**Table 6. Paired test differences, weighted LR minus IF4096 at 1%.**

| Resampling | AP gap | 95% interval | Recall gap (pp) | 95% interval (pp) |
| --- | ---: | ---: | ---: | ---: |
| Paired rows (500) | 0.6815 | [0.5763, 0.7710] | 31.08 | [20.83, 41.78] |
| Time blocks (200) | 0.6815 | [0.5469, 0.7926] | 31.08 | [18.69, 44.24] |

Both resampling schemes retain a positive LR advantage in AP and recall. The block analysis gives wider recall limits, consistent with less independent information when transactions are grouped in time. The intervals describe this fixed-model comparison on the observed test segment. The small test fraud count also makes individual detections visible: one of 74 frauds corresponds to about 1.35 recall percentage points.

The frozen classifier/IF policy equals LR, so its paired AP and recall differences are zero in every resampling draw. The extra anomaly scores leave the selected review list unchanged.

\clearpage

## 4.5 Historical and relationship context

The Handbook test compares identical model families across controlled input views. Each method reviews 1,458 of 145,739 transactions, containing 1,280 frauds. The first four views add histories to current Amount and cyclical time; the fifth adds 7-day customer/terminal degrees and a new-pair flag.

**Table 7. Handbook formal test, static models and the same 1% budget.**

| Input view | LR AP | LR TP | IF AP | IF TP |
| --- | ---: | ---: | ---: | ---: |
| Current transaction | 0.2692 | 346 | 0.1596 | 334 |
| + Customer history | 0.2233 | 339 | 0.0529 | 222 |
| + Terminal history | 0.2500 | 338 | 0.0205 | 55 |
| + Both histories | 0.2130 | 334 | 0.0294 | 139 |
| Both + relations | 0.2064 | 333 | 0.0229 | 104 |

![Handbook formal test: AP and recall for the same LR and IF settings across feature views. “+Relation” extends the both-history view.](../results/figures/report_context.pdf){width=100%}

Current-transaction LR has AP 0.2692 and detects 346 frauds. Adding both histories lowers AP to 0.2130 and detections to 334. The relational extension lowers them further to 0.2064 and 333. Customer-only and terminal-only LR also remain below the current-only model. The descending-Amount reference detects the same 346 frauds as current LR, suggesting a strong amount-driven component in this simulator setting.

The IF results change more sharply. Current-only IF detects 334 frauds, customer-history IF 222, terminal-history IF 55 and both-history IF 139. Additional relational features reduce both-history IF to 104. Adding dimensions alters what is unusual: atypical entity activity need not coincide with the labeled fraud patterns. Historical counts and amounts describe behavior, while some simulator fraud comes from compromised terminals whose activity may otherwise look ordinary.

The controlled comparison supports a practical finding: the value of context depends on the information it encodes and the model using it. Here, a compact current-transaction view preserves useful amount structure; the tested historical and relational summaries do not improve detection. The mechanism breakdown in Section 4.7 helps explain which fraud types remain missed.

\clearpage

## 4.6 Reliability: delayed labels, missing history and new entities

Weekly LR models are fitted during development using only labels available at each cutoff. The saved day-44 snapshots then score the test without further updates. Table 8 compares them with the static both-history LR.

**Table 8. Handbook test using pre-test rolling snapshots, K=1,458.**

| History LR snapshot | AP | TP at 1% | Recall (%) |
| --- | ---: | ---: | ---: |
| Static | 0.2130 | 334 | 26.09 |
| Weekly, delay 0 days | 0.2134 | 341 | 26.64 |
| Weekly, delay 1 day | 0.2138 | 339 | 26.48 |
| Weekly, delay 7 days | 0.2156 | 340 | 26.56 |

The 0-day model adds seven detections over static, the 1-day model adds five and the 7-day model adds six. AP changes are small and are not ordered monotonically by delay. The tested update policy offers a modest improvement over static history LR, but it still detects fewer than the 346 frauds found by current-only LR.

![Handbook formal test under missing event histories. Each condition rebuilds past-only features for the same test transactions and scores the same frozen models. Shading is the range across seeds 42/43/44.](../results/figures/report_stress.pdf){width=100%}

At 30% missing history, both-history IF falls from 139 detections to 77–82; AP falls from 0.0294 to approximately 0.0204–0.0209. LR AP also falls, from 0.2130 to 0.1927–0.1989, while its fixed-budget detections remain 334–337. AP and Recall@K thus respond differently to the same information loss. The IF ordering is more sensitive to incomplete event histories in this experiment.

The cold-start partition is defined by whether a customer and terminal appeared in initial training. Of the test transactions, 145,713 involve entities seen in training; all 1,280 frauds occur there. Only 26 transactions involve new customers, with no fraud, and the new-terminal groups are empty. The group output therefore documents exposure and history availability, but provides no fraud-positive new-entity test. A fraud-positive new-entity dataset is the next useful cold-start test.

\clearpage

## 4.7 Rules, mechanism diagnostics and PaySim

The ULB rule union covers 56,053/56,746 test rows and all 74 frauds, including all 10 LR false negatives and 504 false positives. The first training-ordered rule, R001, covers 3,846 rows and 52 frauds: support 0.0916%, confidence 1.35%, lift 10.37 and Kulczynski 0.3581. High lift coexists with low conditional precision because fraud prevalence is only 0.1304%. This illustrates the base-rate issue in Lectures 3–4 [8,9]; the union’s broad coverage does not specify a 568-review ranking.

Handbook’s 19 rules cover 120,751 transactions and 1,122 frauds. R001 combines Amount at least 65.22 with spending at least about 1.20 times the customer’s preceding 7-day mean. It covers 28,997 rows and 514 frauds: confidence 1.77%, lift 2.02 and Kulczynski 0.2096. This is the rule coverage/accuracy distinction in Lecture 6 [11]. Table 9 uses the simulator’s final mutually exclusive mechanism codes.

**Table 9. Handbook fraud mechanisms; TP from the global 1% history-model lists.**

| Final scenario | Frauds | History LR TP | History IF TP | Covered by rules |
| --- | ---: | ---: | ---: | ---: |
| 1 | 73 | 70 | 7 | 73 |
| 2 | 800 | 4 | 9 | 650 |
| 3 | 407 | 260 | 123 | 399 |

History LR finds 70/73 scenario-1 frauds but only 4/800 in scenario 2; history IF finds 7 and 9, respectively. In scenario 3, LR finds 260/407 and IF 123. The main blind spot is the terminal-compromise scenario, which the selected unlabeled activity summaries capture poorly. Mechanism labels enter this analysis after scoring.

PaySim uses three separately fitted, fixed comparisons on its declared sample. Its 128,873 test transactions contain 413 frauds; K=1,289.

**Table 10. PaySim formal test.**

| PaySim method | AP | TP | Precision (%) | Recall (%) |
| --- | ---: | ---: | ---: | ---: |
| Amount | 0.0798 | 94 | 7.29 | 22.76 |
| Weighted LR | 0.0634 | 80 | 6.21 | 19.37 |
| Fixed IF | 0.0447 | 102 | 7.91 | 24.70 |

The amount ranking leads AP, while IF detects the most fraud at 1%. LR detects fewer than either. This repeats the broader lesson that whole-ranking quality and the chosen review capacity answer different questions across data sources.

# 5. Conclusion and limitations

The strongest ULB fixed-budget result comes from supervised ranking: weighted LR detects 64/74 frauds at 1%. Its amount coverage is lower than its count recall, and its development-selected combination yields no additional same-capacity detections. On Handbook, the tested histories and relations reduce static performance; weekly updating gives a small recovery, and incomplete history harms IF more strongly. Rules reveal broad associations and mechanism-specific blind spots.

The evidence comes from one chronological split with only 74 ULB test frauds and from two simulated sources. Label masks use known class proportions, costs are assumed, and the new-entity groups contain no test fraud. These settings define the scope of the findings. Future work should prioritize fraud-positive new-entity evaluation, calibrated review costs and additional time periods.

\label{maintextend}
\clearpage

# 6. Disclosures

The project used OpenAI Codex/ChatGPT assistance for implementation and refactoring, automated checks, experiment execution, numerical verification, figure generation, repository organization and preparation of the report and poster. The report's tables and plots are based on the stored experiment outputs. Assistance also included locating and checking the data and documentation references listed below.

Each member should complete the name/ID fields and confirm their individual AIGC use and any external collaborators before submission. Sources for the datasets, simulator and software are credited below.

# References

[1] L. Liu. *Credit Card Fraud Detection*, Zenodo, v1, 2022. Archive of the ULB/Worldline dataset. DOI: [10.5281/zenodo.7395559](https://doi.org/10.5281/zenodo.7395559).

[2] Fraud Detection Handbook authors. *Transaction data simulator*, in *Reproducible Machine Learning for Credit Card Fraud Detection—Practical Handbook*. [Online chapter](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html).

[3] E. A. Lopez-Rojas, A. Elmir and S. Axelsson. “PaySim: A financial mobile money simulator for fraud detection.” *28th European Modeling and Simulation Symposium*, 2016. Citation and simulator provided by the [author's repository](https://github.com/EdgarLopezPhD/PaySim).

[4] E. Lopez-Rojas. *Synthetic Financial Datasets for Fraud Detection*, PaySim1 v2, Kaggle. [Dataset page](https://www.kaggle.com/datasets/ealaxi/paysim1).

[5] Fraud Detection Handbook authors. *Validation strategies*. [Online chapter](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html).

[6] Scikit-learn developers. *IsolationForest API documentation*. [Software documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

[7] Scikit-learn developers. *average_precision_score API documentation*. [Software documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

[8] Z. Liu. *Lecture 3: Pattern Mining—Basic Concepts and Methods*. AIAA 3111, course slides updated 17 September 2026.

[9] Z. Liu. *Lecture 4: Pattern Mining—Advanced Methods*. AIAA 3111, course slides updated 27 September 2026.

[10] Z. Liu. *Lecture 5: Classification—Basic Concepts*. AIAA 3111, course slides updated 10 October 2026.

[11] Z. Liu. *Lecture 6: Classification—Advanced Methods*. AIAA 3111, course slides updated 19 October 2026.

[12] Z. Liu. *Lecture 7: Cluster Analysis—Basic Concepts and Methods*. AIAA 3111, course slides updated 29 October 2026.

Online sources checked 18 November 2026. Main-text pages: 1–\pageref{maintextend}. Disclosures, references and appendices are outside the main-text page count.

\clearpage

# Appendix A. Parameters and numerical sources

## A.1 Reproduction

The environment is Python 3.12.14 with the exact dependency pins in requirements.txt. A fresh clone can download ULB, generate the pinned Handbook data, obtain the author's PaySim v2 ZIP, and run:

```bash
python code/main.py --reproduce-final --output data/reproduction
```

The independent Linux run rebuilds all snapshots and compares the published tests. The supplementary ULB random forest has AP 0.7840486 on Linux versus 0.7845347 in the original run; eight amount-group allocation entries differ by 1–5 reviews, while overall budget/TP/FP and amount coverage match. The output records those differences. Following the structural refactor and directory reorganization, local scoring of the original snapshots reproduced all eight aggregate tables numerically, all 48 prediction/case CSVs byte-for-byte, and all 17 test PNGs byte-for-byte. The 98 public checks and 11 classification checks pass.

The first formal run stopped at a cold-start aggregation call. The missing budget argument was corrected and the same frozen models were used for the successful run. The ULB prediction files match across those runs. These records are retained with the freeze and reproduction evidence.

## A.2 Bounded settings

| Component | Fixed settings and selection |
|---|---|
| ULB LR | max_iter=1000, C=1, lbfgs; original / balanced / 1:1 undersampling; seed 42 |
| ULB tree / naive Bayes | DecisionTreeClassifier seed 42; GaussianNB defaults |
| ULB random forest | 100 trees, seed 42, one job |
| K-means structure | k=3/5/8, seeds 42/43, 1,000-row silhouette/ARI sample; selected k=3, n_init=10 |
| K-means representations | Standard / no Time / log Amount / both changes / Robust / both changes + Robust; raw and training-IQR relative distances |
| IF variants | Base; no Time; log Amount; both changes; samples 1,024/4,096; 300 trees; combined transform + 300 trees + 1,024 samples |
| IF selection | Inner chronological train split; seeds 42/43/44; retained base transform, 100 trees, 4,096 samples, contamination=auto |
| E03 labels | 5/10/25/50/100%; class-stratified nested masks, seeds 42/43/44; LR seed 42, fixed IF256 reference |
| E05 permutation | Fixed full-label weighted LR, AP, 30 features × 5 repeats, seed 42 |
| Rules | Three bins; maximum 2 antecedents; coverage >=20, frauds >=5, lift >1 in training and validation; at most 50 candidates per scheme |
| Handbook | Fixed LR and 100-tree IF256; four views plus 3 relation features; seed 42 |
| E11 / E14 | Updates every 7 days in development, 30-day window, 0/1/7-day delay; missing history 0/10/30%, three seeds |
| PaySim | Fixed label-independent sample, one-hot type and Amount, weighted LR and IF256; separate amount ranking |

\clearpage

## A.3 Result-to-source map

| Report content | Canonical source in results/tables/ |
|---|---|
| Dataset versions, parameters, frozen thresholds | group_freeze.json; handbook_a.json; paysim_a.json |
| ULB methods and fixed-threshold test | ulb_test.json |
| Cluster allocation and same-budget overlap | c_test.json |
| All label masks and test importance repeats | b_extensions_test.json |
| Amount, costs, 500 row / 200 block draws | c_extensions_test.json |
| History, relation, rolling, cold-start and missing-history tests | handbook_test.json |
| Semantic rule intervals, union and mechanism counts | handbook_d_test.json |
| PaySim formal comparison | paysim_test.json |
| Test consistency and independent reconstruction | project_test.json; reproduction.json |

Figures 1–5 are rebuilt directly from these formal JSON files by report/build_report.py. Their PDF versions are stored in results/figures/ with the prefix report_. The underlying scientific outputs are unchanged. Every reported method uses the original split and its predeclared fit policy; all conditions, including unsuccessful changes, remain available in the JSON tables.

The main text emphasizes the primary comparison and the most informative controlled contrasts. Full 20-score ULB ablations, all budgets, three-seed conditions, time windows, individual rule metrics and error associations are supplied by the same source files. Row-level predictions and fitted models remain in the local data/ directory; a fresh clone rebuilds them from the documented sources.

The original ULB validation had been examined before the second C protocol. Its inner training split limits further use of that validation segment for the anomaly selection, and this development history remains part of the method record.


\clearpage

# Appendix B. Supporting calculations

## B.1 Why SSE alone does not select k

**Table B1. Original training-structure experiment; mean across seeds 42/43.**

| k | Mean training SSE | Mean sample silhouette | Seed ARI |
| --- | ---: | ---: | ---: |
| 3 | 4,699,495 | 0.1537 | 0.5558 |
| 5 | 4,409,757 | 0.0669 | 0.1530 |
| 8 | 3,994,957 | 0.0614 | 0.1317 |

SSE falls as k grows, but silhouette and agreement deteriorate. The selected k=3 is supported by cohesion/separation and initialization stability within the declared range. Here ARI compares the two seeded cluster assignments on the same unlabeled sample.

## B.2 Reading an association rule under a rare consequent

**Table B2. Individual R001 test metrics, derived from the frozen rules’ stored counts.**

| Rule | Coverage (%) | Support (%) | Confidence (%) | Lift | Kulczynski |
| --- | ---: | ---: | ---: | ---: | ---: |
| ULB R001 | 6.7776 | 0.0916 | 1.3521 | 10.3681 | 0.3581 |
| Handbook R001 | 19.8965 | 0.3527 | 1.7726 | 2.0183 | 0.2096 |

Coverage is A/N; rule support is J/N, where A is the antecedent count and J the covered-fraud count. The two differ because many covered transactions are normal. Confidence J/A answers how often a fired rule is correct, whereas J/F measures the share of frauds it covers. Their average is Kulczynski. Each rule in Table B2 is first in the fixed training order, so its individual counts equal the first cumulative step.

Lecture 4’s null-invariance point is visible algebraically: holding A, J and F fixed while adding transactions with neither antecedent nor fraud leaves confidence and Kulczynski unchanged. Lift JN/(AF) rises with N. This explains why lift should be read alongside conditional precision and coverage in this rare-class task.

## B.3 Comparing the declared discretizations

**Table B3. Development-only rule comparison. “Retained” means passing the same validation count/lift gates.**

| Dataset | Binning | Training candidates | Validation retained |
| --- | ---: | ---: | ---: |
| ULB | equal-width | 38 | 15 |
| ULB | equal-frequency | 25 | 21 |
| Handbook | equal-width | 16 | 6 |
| Handbook | equal-frequency | 24 | 19 |

Equal-frequency retains more rules for both datasets under the fixed gates, so it supplies the frozen 21-rule ULB and 19-rule Handbook sets. Equal-width and equal-frequency create different interval occupancies from the same training values. For ULB, 21/25 equal-frequency candidates (84.0%) survive validation, versus 15/38 equal-width candidates (39.5%). This makes the representation comparison concrete: fewer training candidates can yield more patterns that persist into the next time segment. The formal test then applies the selected bins and rules.
