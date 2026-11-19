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
| Jingyang YI | 50027058 |
| Shuhan SHEN | 50025347 |
| Congren HANG | 50025402 |
| Zhanlin CHEN | 50024707 |

**Abstract.** Fraud teams must decide which transactions to inspect when they can review only a fraction of all alerts. We compare the lists produced by different rankings at the same review capacity. On the chronological ULB test set, weighted logistic regression finds 64 of 74 frauds from 568 reviews, leaving 10 frauds outside the list. Doubling the weighted model’s reviews from 284 to 568 finds one additional fraud. In Handbook, the tested histories reduce detections and missing history affects Isolation Forest more strongly. In PaySim, the method with the most detections at 1% differs from the method with the highest average precision. These results motivate choosing a ranking at the available review capacity and retesting it when the available information or transaction setting changes.

# 1. Introduction

Missing fraud can mean financial losses, and manually checking alerts takes staff time. A review team therefore needs a shortlist. Suppose B and E are the two frauds among six transactions and staff can check only two. A ranking that puts B and E first finds both. A ranking that puts B and C first finds one, even if E is third. We use known labels after ranking to measure what the team would find.

The experiments follow the decisions a review team faces:

- **Build and compare the lists (Sections 3 and 4.1).** Fit the models on earlier transactions and rank later ones. On ULB [1], give each method the same review capacity and count the frauds its list finds.
- **Check when the choice changes (Sections 4.2–4.6).** Compare overlap and vary review capacity, access to labels and the information available at review time. Handbook [2] tests customer and terminal histories under controlled changes in the inputs.
- **Explain misses and check another setting (Section 4.7).** Use rules and fraud mechanisms to investigate missed cases. On PaySim [3,4], ask whether the same method would lead in a different transaction setting.

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

**Classification.** LR uses $p_i=\sigma(\beta^\top z_i+b)$ and regularized log loss. Balanced weights $w_c=N/(2N_c)$ raise the contribution of the rare fraud class. Gaussian naive Bayes assumes conditional independence and Gaussian feature likelihoods. Trees choose splits by Gini reduction, $G=1-\sum_c p_c^2$; the 100-tree random forest combines bootstrap sampling with random feature subsets. LR and naive Bayes use StandardScaler; trees use unscaled inputs. Original, weighted and 1:1 undersampled LR compare ordinary fitting, cost-sensitive learning and data-level balancing. Undersampling retains 342 frauds and 342 normal records, fitting its scaler on those 684 rows. Seeds are 42. Validation TP at 1%, then AP, then method name selects weighted LR.

**Clustering and anomaly scores.** K-means minimizes $\mathrm{SSE}=\sum_i\|z_i-\mu_{c_i}\|_2^2$ on standardized features. It ranks distance to the assigned centroid. Because SSE favors larger k, training silhouette selects $k=3$ from 3/5/8. Ten initializations reduce local-optimum sensitivity. Isolation Forest ranks $-\mathrm{score\_samples}(x)$: shorter isolation paths imply greater abnormality [6]. A bounded second round compares six K-means representations with raw and within-cluster relative distances, and eight IF settings. An inner chronological 80/20 training split with seeds 42/43/44 selects family representatives by mean TP at 1%, then AP. The retained IF has 100 trees and 4,096 training samples per tree. All 20 scores remain in the test output.

**Review combinations.** Quotas are 0/25/50/75/100%. The first ranking contributes $\lfloor K\alpha\rfloor$ transactions; the second fills the list while skipping duplicates. Inner training data choose the anomaly-pair quota. Validation chooses the classifier/IF quota, preferring a pure method on a tie. The frozen policies are pure IF for the anomaly pair and pure weighted LR for the classifier comparison.

## 3.4 Context, rules and evaluation

For Handbook, we summarize each customer’s transaction sequence into 1- and 7-day features, extract degree and new-pair features from customer–terminal edges, and fit the same LR and IF256. Development LR updates on days 30/37/44 use a 30-day sliding window and 0/1/7-day label delays [5]. Saved snapshots stay fixed in test. History stress tests omit 0/10/30% of source events from later histories with seeds 42/43/44; current transactions and labels stay present.

Each transaction contains one bin item per selected feature. Apriori joins antecedents only when their subsets cover at least 20 rows: support is downward-closed, while confidence and lift are assessed after counting. We compare three-bin equal-width/equal-frequency discretization and at most two antecedents followed by Class=1. Rules require coverage at least 20, at least 5 frauds and lift above 1 in both training and validation. The scheme retaining more rules wins; ties prefer equal-frequency. ULB uses Amount and four V features chosen by a training-correlation filter; Handbook uses amount, relative amount and 7-day counts. The retained sets contain 21 and 19 rules.

With $F=\sum_i y_i$, evaluation uses $P@K=TP/K$, $R@K=TP/F$, non-interpolated AP [7], fraud-amount coverage and loss $L=FP+rFN$ for $r=10,50,100$. We use 500 paired row draws and 200 draws of 30-minute blocks; intervals condition on the fitted models. For antecedent count A and covered-fraud count J, rule support is $J/N$, confidence $J/A$, lift $JN/(AF)$ and Kulczynski $\tfrac12(J/A+J/F)$. Overlapping coverage is counted once.

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

The smallest budget separates the methods sharply: weighted LR finds 50 frauds within 57 reviews, whereas IF4096 finds none. At 0.5%, LR already finds 63; increasing to 1% adds one fraud at 284 additional reviews. An always-normal classifier has 99.87% accuracy and zero recall. The positive-class score $F_1=2TP/(K+F)$ gives 0.1994 for LR and 0.1277 for IF at K=568; with the same K and F, this is the TP ordering. A random list has only 0.74 expected frauds.

\clearpage

## 4.2 Complementarity, groups and time

To see whether anomaly scores can improve the shortlist, we first examine which frauds the rankings find in common. At 1%, the classifier and IF lists share 100 transactions and 40 detected frauds. The classifier contributes 24 additional frauds and IF contributes one. Their union would detect 65 frauds while reviewing 1,036 transactions. The frozen 100% classifier quota reviews 568 and detects 64. This distinguishes useful complementary information from an improvement already achieved under the same capacity. The anomaly-only policy likewise remains the preselected pure IF endpoint.

K-means groups provide a second view of the ranking. Table 3 allocates each method's single global list across the same groups; its review counts sum to 568 for each method.

**Table 3. ULB test groups and allocation of the global 1% review list.**

| Cluster | Rows | Frauds | LR reviews / TP | IF reviews / TP |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8411 | 16 | 54 / 15 | 208 / 13 |
| 1 | 48335 | 58 | 514 / 49 | 360 / 28 |
| 2 | 0 | 0 | 0 / 0 | 0 / 0 |

The shared list reaches the two occupied groups unevenly. Appendix C gives the training cluster diagnostics. Appendix B tests an alternative grouping with probabilistic membership.

A separate question is how many alerts a fixed validation threshold produces later. Table 4 applies the original thresholds without changing them.

**Table 4. Frozen validation thresholds on the ULB test segment.**

| Method | Frozen threshold | Reviews | TP | Review rate (%) |
| --- | ---: | ---: | ---: | ---: |
| Weighted LR | 0.8043 | 453 | 63 | 0.80 |
| IF4096 | 0.5318 | 374 | 36 | 0.66 |

The fixed LR threshold uses 453 reviews to find 63 frauds. A threshold fixes the score cutoff, so workload can change over time. The top-$K$ rule fixes the number of reviews. Hourly outputs distinguish each hour’s own top-$K$ list from the portion of the global list allocated to that hour.

\clearpage

## 4.3 Label availability and model interpretation

To see whether the supervised shortlist remains useful with fewer labels, we fit LR on reduced labeled subsets and use a separate IF reference fitted without labels. Within each training class, random permutations give nested 5/10/25/50/100% subsets using ceiling counts. The masks use the original labels to control class composition; hidden labels are excluded from fitting. LR sees its labeled subset, while fixed IF256 sees all training features. Table 5 retains every predeclared condition.

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

Amount has the largest permutation effect, lowering test AP by 0.3695 on average when shuffled. Appendix C gives the other feature effects and ranked error examples.

\clearpage

## 4.4 Fraud counts, amounts, costs and uncertainty

At 1%, weighted LR covers 86.49% of fraud transactions and 69.59% of their amount. IF4096 covers 55.41% and 47.51%; cluster distance covers 50.00% and 10.71%. The descending-Amount reference finds only 3 frauds but covers 46.56% of fraud amount. The test fraud amount totals 7,727.67 units. The count-based advantage therefore answers how many incidents staff find. Amount coverage answers how much of the fraudulent spend is in that same shortlist.

![ULB formal test: count and amount coverage at 1%, and loss across budgets for the fixed FN:FP cost ratio of 50:1.](../results/figures/report_amount.pdf){width=100%}

Under the 50:1 cost scenario, weighted LR has loss 771 at a 0.5% budget and 1,004 at 1%. The larger list gains one detected fraud but adds many normal reviews. With a different assumed error cost, the balance changes. These calculations compare the declared scenarios; the primary experimental budget stays at 1%. Fraud amount is a transaction quantity rather than a measured recovery amount.

**Table 6. What changes when the review budget grows? Weighted LR on ULB.**

| Budget | Reviews | Frauds found | Normal reviews | Fraud amount (%) | Added fraud / normal |
| --- | --- | --- | --- | --- | --- |
| 0.1% | 57 | 50 | 7 | 57.27 | — |
| 0.5% | 284 | 63 | 221 | 69.37 | +13 / +214 |
| 1% | 568 | 64 | 504 | 69.59 | +1 / +283 |
| 2% | 1135 | 65 | 1070 | 69.69 | +1 / +566 |

From 0.5% to 1%, the team would review 283 more normal transactions to find one additional fraud. Fraud-amount coverage rises from 69.37% to 69.59%. The paired interval estimates for the main LR comparison are in Appendix C.

\clearpage

## 4.5 Historical and relationship context

Handbook tests whether additional information improves the shortlist for the same model family. Each method reviews 1,458 of 145,739 transactions, containing 1,280 frauds. The first four views add histories to current Amount and cyclical time; the fifth adds 7-day customer/terminal degrees and a new-pair flag.

**Table 7. Handbook formal test, static models and the same 1% budget.**

| Input view | LR AP | LR TP | IF AP | IF TP |
| --- | ---: | ---: | ---: | ---: |
| Current transaction | 0.2692 | 346 | 0.1596 | 334 |
| + Customer history | 0.2233 | 339 | 0.0529 | 222 |
| + Terminal history | 0.2500 | 338 | 0.0205 | 55 |
| + Both histories | 0.2130 | 334 | 0.0294 | 139 |
| Both + relations | 0.2064 | 333 | 0.0229 | 104 |

![Handbook formal test: AP and recall for the same LR and IF settings across feature views. “+Relation” extends the both-history view.](../results/figures/report_context.pdf){width=100%}

The current-transaction LR finds 346 frauds. Adding both histories reduces that to 334, and adding relationships reduces it to 333. An amount-only ranking also finds 346, suggesting a strong amount-driven component in this simulator. The parallel IF comparison shows a larger decline with the tested histories. A supplemental ranking based only on the ratio to a customer’s past mean finds 255 frauds. It identifies 46 frauds outside the amount ranking’s list (Appendix B).

We then test whether the order of a customer's last three transactions helps beyond the same information in an unordered representation. Both representations raise the both-history LR from 334 to 351 detections. This experiment finds a gain from the recent amount pattern and no additional detections from preserving its order. Appendix B gives the matched comparison. Its probabilistic grouping also identifies a lower-amount component with 141 frauds and no detections in the original both-history LR list, a specific group to inspect for misses.

\clearpage

## 4.6 Review feedback and reliability

Once a team reviews a list, its findings can become training labels. We replay Handbook in closed daily batches of 100 reviews. All policies start with the same weighted LR. Updates use only reviewed labels after the delay. The two exploration policies reserve 10 of the 100 slots for random or uncertain transactions.

**Table 8. Supplemental replay on Handbook days 45–59.**

| Feedback policy | Reviews | Frauds found | Ranking-only top 100 |
| --- | --- | --- | --- |
| Static ranking | 1500 | 346 | 346 |
| Risk-only updates | 1500 | 345 | 345 |
| Risk + random | 1500 | 343–344 | 345 |
| Risk + uncertainty | 1500 | 344 | 345 |

The static policy finds 346 frauds, versus 343–345 for the updating policies. Even if all 100 slots were assigned by score after an update, the resulting rankings find 345. This replay gives no evidence of a gain from allocating some reviews to collect different labels. Appendix B details the timing and scoring.

![Handbook formal test under missing event histories. Each condition rebuilds past-only features for the same test transactions and scores the same frozen models. Shading is the range across seeds 42/43/44.](../results/figures/report_stress.pdf){width=100%}

Deleting 30% of past events reduces both-history IF detections from 139 to 77–82, while LR finds 334–337. The original pre-test rolling and cold-start diagnostics are in Appendix C.

\clearpage

## 4.7 Rules, mechanism diagnostics and PaySim

The ULB rule union covers 56,053/56,746 test rows and all 74 frauds, including all 10 LR false negatives and 504 false positives. The first training-ordered rule, R001, covers 3,846 rows and 52 frauds: support 0.0916%, confidence 1.35%, lift 10.37 and Kulczynski 0.3581. High lift coexists with low conditional precision because fraud prevalence is only 0.1304%. The union’s broad coverage does not specify a 568-review ranking.

Handbook’s 19 rules cover 120,751 transactions and 1,122 frauds. R001 combines Amount at least 65.22 with spending at least about 1.20 times the customer’s preceding 7-day mean. It covers 28,997 rows and 514 frauds: confidence 1.77%, lift 2.02 and Kulczynski 0.2096. Table 9 uses the simulator’s final mutually exclusive mechanism codes.

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

On PaySim, IF finds the most fraud at 1%, while the amount ranking leads AP. The ULB method choice therefore does not carry over to this setting. For a team working through the top 1% of alerts, the relevant difference is in the shortlist.

# 5. Conclusion and limitations

Set the review capacity first and compare how many frauds each shortlist finds. At the ULB 1% budget, the selected weighted LR finds 64/74, and the selected combination adds no detections at the same workload. Expanding from 284 to 568 reviews adds only one. The Handbook results show that adding or losing history can change shortlist quality. PaySim shows that a new transaction setting can change the preferred method. The daily replay finds no gain from the tested review feedback policies. The sequence comparison finds a gain from a recent amount pattern, with equal detections whether or not it preserves order. Use rules and transaction groups to diagnose misses, then retest the ranking in the new setting.

The evidence comes from one chronological split with only 74 ULB test frauds and from two simulated sources. Label masks use known class proportions, costs are assumed, and the new-entity groups contain no test fraud. These settings define the scope of the findings. Future work should prioritize fraud-positive new-entity evaluation, calibrated review costs and additional time periods.

\label{maintextend}
\clearpage

# 6. Disclosures

The project used OpenAI Codex/ChatGPT assistance for implementation and refactoring, automated checks, experiment execution, numerical verification, figure generation, repository organization and preparation of the report and poster. The report's tables and plots are based on the stored experiment outputs. Assistance also included locating and checking the data and documentation references listed below.

# References

[1] L. Liu. *Credit Card Fraud Detection*, Zenodo, v1, 2022. Archive of the ULB/Worldline dataset. DOI: [10.5281/zenodo.7395559](https://doi.org/10.5281/zenodo.7395559).

[2] Fraud Detection Handbook authors. *Transaction data simulator*, in *Reproducible Machine Learning for Credit Card Fraud Detection—Practical Handbook*. [Online chapter](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html).

[3] E. A. Lopez-Rojas, A. Elmir and S. Axelsson. “PaySim: A financial mobile money simulator for fraud detection.” *28th European Modeling and Simulation Symposium*, 2016. Citation and simulator provided by the [author's repository](https://github.com/EdgarLopezPhD/PaySim).

[4] E. Lopez-Rojas. *Synthetic Financial Datasets for Fraud Detection*, PaySim1 v2, Kaggle. [Dataset page](https://www.kaggle.com/datasets/ealaxi/paysim1).

[5] Fraud Detection Handbook authors. *Validation strategies*. [Online chapter](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_5_ModelValidationAndSelection/ValidationStrategies.html).

[6] Scikit-learn developers. *IsolationForest API documentation*. [Software documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

[7] Scikit-learn developers. *average_precision_score API documentation*. [Software documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

\clearpage

# Appendix A. Reproducing the results

Follow the repository README to prepare ULB, Handbook and PaySim with the pinned Python dependencies. From the repository root:

```bash
python code/main.py --reproduce-final --output data/reproduction
```

To rebuild the supplemental results, give the runner the pinned 60-day Handbook CSV:

```bash
python report/build_supplement.py path/to/transactions.csv
```

| Evidence | Repository source |
|---|---|
| Dataset versions, split boundaries, frozen configurations and model selection | results/tables/group_freeze.json |
| ULB rankings, fixed thresholds and rule coverage | results/tables/ulb_test.json |
| ULB training cluster geometry and test list allocation | results/tables/c_baseline.json; c_test.json |
| ULB label budgets, failure examples, amount, costs and uncertainty | results/tables/b_extensions_test.json; c_extensions_test.json |
| Handbook features, delays, missing histories and cold start | results/tables/handbook_test.json |
| Rule settings, redundancy and scenario diagnostics | report/d_validation/manifest.json; results/tables/handbook_d.json; handbook_d_test.json |
| PaySim test and full reconstruction checks | results/tables/paysim_test.json; reproduction.json |
| Supplemental daily records, sequence scores, GMM groups and list overlap | results/tables/review_supplement.json |

The figure and table build is `python report/build_report.py`. The independent Linux reconstruction and any platform differences are recorded in `reproduction.json`. The original experiment grid and implementation parameters are in `report/EXTENSION_PROPOSALS.md`.

\clearpage

# Appendix B. Supplemental review-list analyses

The Handbook additions reuse the previously examined time split and are exploratory.

**Review feedback (Table 8).** Start with labeled days 0–29 and review 100 transactions per day from day 30. Exploration assigns ten slots to a random draw or the scores closest to 0.5 outside the first 90. Labels from day $d$ become available on day $d+2$. Refitting every third day uses the initial training rows and only the available reviewed labels. Evaluate on days 45–59. The random-policy range uses three seeds.

**Ordered history.** Encode the last three strictly earlier customer transactions within seven days using the training median amount. The unordered version counts high amounts. The ordered version uses eight binary patterns. Rows with fewer than three eligible events have no pattern indicator. Both use the same model and base features.

**Table B1. Handbook static LR with and without recent amount patterns.**

| History features | Reviews | Frauds found | AP |
| --- | --- | --- | --- |
| Existing histories | 1458 | 334 | 0.2130 |
| Histories + unordered pattern | 1458 | 351 | 0.2268 |
| Histories + ordered pattern | 1458 | 351 | 0.2255 |

**Missed groups.** Fit a three-component Gaussian mixture on training log amounts and 1-day customer and terminal counts after scaling. Assign by highest membership and order components by increasing training mean log amount. All rows share the original 1% review list.

**Table B2. Global list allocation by most probable component.**

| GMM group | Transactions | Frauds | Reviews | Found |
| --- | --- | --- | --- | --- |
| Component 1 | 22516 | 141 | 2 | 0 |
| Component 2 | 53800 | 480 | 465 | 115 |
| Component 3 | 69423 | 659 | 991 | 219 |

**Customer-relative amount.** Divide current amount by the customer’s mean amount over strictly earlier transactions in the last seven days. Put rows without a positive historical mean last and break ties by row ID.

**Table B3. Absolute amount versus customer-relative amount on Handbook.**

| Ranking | Reviews | Frauds found | AP |
| --- | ---: | ---: | ---: |
| Absolute amount | 1458 | 346 | 0.2693 |
| Customer-relative amount | 1458 | 255 | 0.1034 |

The two lists share 209 frauds. Relative amount finds 46 outside the absolute-amount list. Combining them would find 392 frauds in 2,526 reviews.

\clearpage

# Appendix C. Additional formal-test diagnostics

**Cluster geometry.** On the fixed training sample, $k=3$ has silhouette 0.1537 and seed-agreement ARI 0.5558 versus 0.0669 and 0.1530 at $k=5$. SSE decreases with more clusters. The seed comparison measures agreement on the same unlabeled sample.

**Ranked errors.** Shuffling Amount, V14 and V2 produces the three largest mean AP decreases. One normal transaction ranks third, while a missed fraud ranks 783rd under the selected ULB LR. The shortlist contains ranks 1–568.

**Rule checks.** Equal-frequency binning retains 21 ULB and 19 Handbook rules, versus 15 and 6 under equal-width binning. High lift can coexist with low confidence when fraud is rare. The rule metrics and binning candidates are available in the sources listed in Appendix A.

**Table C1. Paired test differences, weighted LR minus IF4096 at 1%.**

| Resampling | AP gap | 95% interval | Recall gap (pp) | 95% interval (pp) |
| --- | ---: | ---: | ---: | ---: |
| Paired rows (500) | 0.6815 | [0.5763, 0.7710] | 31.08 | [20.83, 41.78] |
| Time blocks (200) | 0.6815 | [0.5469, 0.7926] | 31.08 | [18.69, 44.24] |

The block interval is wider for this fixed-model comparison. It accounts for some time dependence by resampling transactions together within blocks.

**Table C2. Handbook test using pre-test rolling snapshots, K=1,458.**

| History LR snapshot | AP | TP at 1% | Recall (%) |
| --- | ---: | ---: | ---: |
| Static | 0.2130 | 334 | 26.09 |
| Weekly, delay 0 days | 0.2134 | 341 | 26.64 |
| Weekly, delay 1 day | 0.2138 | 339 | 26.48 |
| Weekly, delay 7 days | 0.2156 | 340 | 26.56 |

The rolling snapshots only use labels available at their development cutoff. The initial-training cold-start groups include 26 transactions from new customers and no fraud positives, so they do not support a fraud-detection comparison on new entities.
