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

**Abstract.** When investigators can review only a small share of card payments, which transactions should they check first? We give each ranking the same review capacity. On the chronological ULB test set, the weighted logistic regression selected on validation finds 64 of 74 frauds in 568 reviews. Increasing its list from 284 to 568 adds one fraud and 283 normal transactions. Using Handbook and PaySim, we test how the ranking responds to different information and a different payment setting. We then use association rules and clustering to see what the review list misses. These comparisons help a team set a workable review capacity, choose a ranking at that capacity and check when the choice needs revisiting.

# 1. Introduction

Card issuers monitor payments for possible fraud and investigate cases that need follow-up [8]. Our task is to decide which payments staff should check first when review time is limited. Suppose B and E are the two frauds among six payments and staff can check only two. The list B, E finds both. The list B, C finds only one even if E ranks third. We use known test labels after scoring to count what each list would find.

We follow the review list from construction to diagnosis:

- **Build a fair shortlist (§3 and §4.1).** Use earlier transactions to learn a score and compare classification, clustering and anomaly detection on later ULB payments [1].
- **Find out what changes the shortlist (§4.2–§4.6).** Vary the number of reviews and the labels used for learning. On Handbook [2], vary which past activities the models can see.
- **Explain misses and recheck the choice (§4.7).** Use Apriori rules to flag patterns and clustering to find under-reviewed groups. Refit the ranking approaches on PaySim [3,4] and compare them at the same 1% review rate.

# 2. Problem formulation

Each transaction has features $x_i$, a label $y_i\in\{0,1\}$ and a stable row ID. A method returns a risk score $s_i$. For $N$ transactions and budget $q$, we review $K=\lceil Nq\rceil$ records, ordered by decreasing score and then increasing row ID. The primary budget is 1% of the test transactions. Budgets of 0.1%, 0.5% and 2% let us measure what extra reviews buy.

\clearpage

# 3. Methodology

## 3.1 Data acquisition and preprocessing

We use the fixed ULB/Worldline CSV archived on Zenodo [1]. It contains 284,807 transactions and 492 frauds. Time records elapsed seconds and V1–V28 are anonymous PCA components. The file has no missing values or customer IDs that link payments. Exact duplicates across the 31 original fields are merged, retaining the first source row: 1,081 extra records, including 19 frauds, are removed. This record-based policy produces 283,726 transactions and 473 frauds. Extreme amounts are retained.

Transactions are sorted by Time and original row ID. Boundaries target 60/20/20 by row count; equal timestamps enter the later partition together. The ULB ranges are 0–120,392 seconds for training, 120,393–145,233 for validation, and 145,234–172,792 for test. The 30 numerical inputs exclude the label and row ID. Scaling is fitted within each model's training data.

**Table 1. Chronological partitions; each cell is transactions / frauds.**

| Dataset | Training | Validation | Test |
|---|---:|---:|---:|
| ULB, cleaned | 170,235 / 342 | 56,745 / 57 | 56,746 / 74 |
| Handbook | 292,139 / 1,752 | 145,315 / 1,291 | 145,739 / 1,280 |
| PaySim sample | 379,109 / 326 | 127,377 / 64 | 128,873 / 413 |

Handbook [2] simulates 60 days with 5,000 customers, 10,000 terminals and fixed seeds. Customer and terminal IDs link earlier payments to the current one. Days 0–29 train the static models, days 30–44 support development, and days 45–59 form the test. Current features are Amount and sine/cosine time encodings. Customer and terminal histories use preceding 1-day and 7-day counts, mean amounts and availability indicators; customer features also compare the current amount with previous spending. The four views have 3, 13, 9 and 19 features. Histories use $[t-w,t)$, so transactions at the same timestamp cannot enter one another's history. IDs support joins and grouping; they are excluded from model features.

PaySim [3,4] simulates mobile-money transfers. A fixed sample independent of fraud labels retains 635,359 of 6,362,620 records from the v2 archive (about 10%). Time steps remain intact across the chronological split. Models use requested Amount and one-hot transaction type; balance fields, IDs and the system flag are excluded. The three datasets are fitted and evaluated separately.

## 3.2 Experimental sequence

Train on the earliest period, choose settings on validation, then freeze them before scoring later payments. The freeze records data, seeds, features, score cutoffs, rule bins and review quotas. In the formal Handbook test, earlier transactions can update the history features. The fitted models and policies stay fixed.

\clearpage

## 3.3 Classification, clustering and anomaly detection

**Classification.** LR uses $p_i=\sigma(\beta^\top z_i+b)$ and regularized log loss. Balanced weights $w_c=N/(2N_c)$ raise the contribution of the rare fraud class. Gaussian naive Bayes assumes conditional independence and Gaussian feature likelihoods. Trees choose splits by Gini reduction, $G=1-\sum_c p_c^2$; the 100-tree random forest combines bootstrap sampling with random feature subsets. LR and naive Bayes use StandardScaler; trees use unscaled inputs. Original, weighted and 1:1 undersampled LR compare ordinary fitting, cost-sensitive learning and data-level balancing. Undersampling retains 342 frauds and 342 normal records, fitting its scaler on those 684 rows. Seeds are 42. Validation TP at 1%, then AP, then method name selects weighted LR.

**Clustering and anomaly scores.** These methods can fit without fraud labels. K-means minimizes $\mathrm{SSE}=\sum_i\|z_i-\mu_{c_i}\|_2^2$ on standardized features. It ranks distance to the assigned centroid. Because SSE favors larger k, training silhouette selects $k=3$ from 3/5/8. Ten initializations reduce local-optimum sensitivity. Isolation Forest (IF) gives high scores to payments that are easy to separate from others in random trees [6]. It ranks $-\mathrm{score\_samples}(x)$. A bounded second round compares six K-means representations with raw and within-cluster relative distances, and eight IF settings. An inner chronological 80/20 training split with seeds 42/43/44 selects family representatives by mean TP at 1%, then AP. The retained IF has 100 trees and 4,096 training samples per tree. All 20 scores remain in the test output.

**Review combinations.** Quotas are 0/25/50/75/100%. The first ranking contributes $\lfloor K\alpha\rfloor$ transactions; the second fills the list while skipping duplicates. Inner training data choose the anomaly-pair quota. Validation chooses the classifier/IF quota, preferring a pure method on a tie. The frozen policies are pure IF for the anomaly pair and pure weighted LR for the classifier comparison.

## 3.4 Context, rules and evaluation

Handbook lets us add past activity to the current payment. We use 1- and 7-day summaries and links between customers and terminals. We compare LR, which learns from labels, with IF256, which can fit without them. Development LR updates on days 30/37/44 use a 30-day sliding window and 0/1/7-day label delays [5]. Saved snapshots stay fixed in test. History stress tests omit 0/10/30% of source events from later histories with seeds 42/43/44; current transactions and labels stay present.

For Apriori, turn each selected numeric feature into a bin. Each rule combines conditions, such as a high amount and a high amount relative to past spending. Join candidates only when their subsets cover at least 20 rows (downward closure), then calculate confidence and lift. We compare three-bin equal-width/equal-frequency discretization and at most two antecedents followed by Class=1. Rules require coverage at least 20, at least 5 frauds and lift above 1 in both training and validation. The scheme retaining more rules wins; ties prefer equal-frequency. ULB uses Amount and four V features chosen by a training-correlation filter; Handbook uses amount, relative amount and 7-day counts. The retained sets contain 21 and 19 rules.

With $F=\sum_i y_i$, evaluation uses $P@K=TP/K$, $R@K=TP/F$, non-interpolated AP [7], fraud-amount coverage and loss $L=FP+rFN$ for $r=10,50,100$. We use 500 paired row draws and 200 draws of 30-minute blocks; intervals condition on the fitted models. For antecedent count A and covered-fraud count J, rule support is $J/N$, confidence $J/A$, lift $JN/(AF)$ and Kulczynski $\tfrac12(J/A+J/F)$. Overlapping coverage is counted once.

\clearpage

# 4. Experiments and results

## 4.1 Which ranking finds more fraud at the same capacity?

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

Start with the same 568 review slots. Weighted LR finds 64 of 74 frauds, so staff would also review 504 normal payments and miss 10 frauds. The two baselines fitted without labels find fewer: 41 for IF4096 and 37 for centroid distance. IF4096 improves on the 26 found by IF256. Recall is 86.49% for weighted LR and 55.41% for IF4096.

Random forest, ordinary LR and weighted LR each find 64 frauds. Random forest has the highest AP, which measures how well the full list ranks frauds. We keep the weighted LR policy selected on validation. Class weighting raises AP from 0.6909 to 0.7475 at the same 1% count. Undersampling finds one fewer fraud and has a lower AP.

![ULB test recall and precision across the four fixed budgets. The methods and models are frozen; each curve uses the same transactions at each budget.](../results/figures/report_budget.pdf){width=100%}

The smallest budget separates the methods sharply: weighted LR finds 50 frauds within 57 reviews, whereas IF4096 finds none. At 0.5%, LR already finds 63; increasing to 1% adds one fraud at 284 additional reviews. An always-normal classifier has 99.87% accuracy and zero recall. The positive-class score $F_1=2TP/(K+F)$ gives 0.1994 for LR and 0.1277 for IF at K=568; with the same K and F, this is the TP ordering. A random list has only 0.74 expected frauds.

\clearpage

## 4.2 Can the list change without adding work?

IF trails weighted LR on ULB. It may still flag cases the classifier misses, since it scores unusual payments without fraud labels. The 1% lists share 100 payments and 40 frauds. LR finds 24 frauds unique to its list and IF finds one. Their union finds 65 in 1,036 reviews. With 568 slots, the validation-selected blend uses weighted LR alone and finds 64. The anomaly-only blend uses IF alone.

An overall detection count can hide which groups receive reviews. Table 3 shows where the two 568-slot lists spend their capacity across the same K-means groups.

**Table 3. ULB test groups and allocation of the global 1% review list.**

| Cluster | Rows | Frauds | LR reviews / TP | IF reviews / TP |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8411 | 16 | 54 / 15 | 208 / 13 |
| 1 | 48335 | 58 | 514 / 49 | 360 / 28 |
| 2 | 0 | 0 | 0 / 0 | 0 / 0 |

The shared list reaches the two occupied groups unevenly. Appendix C gives the training cluster diagnostics. Appendix B tests an alternative grouping with probabilistic membership.

Another way to build a list is to keep a fixed score cutoff. Table 4 applies the cutoff chosen on validation to later ULB payments.

**Table 4. Frozen validation thresholds on the ULB test segment.**

| Method | Frozen threshold | Reviews | TP | Review rate (%) |
| --- | ---: | ---: | ---: | ---: |
| Weighted LR | 0.8043 | 453 | 63 | 0.80 |
| IF4096 | 0.5318 | 374 | 36 | 0.66 |

The fixed LR threshold uses 453 reviews to find 63 frauds. A threshold fixes the score cutoff, so workload can change over time. The top-$K$ rule fixes the number of reviews. Hourly outputs distinguish each hour’s own top-$K$ list from the portion of the global list allocated to that hour.

\clearpage

## 4.3 How much does the ranking rely on labels and features?

Weighted LR learns from known fraud and normal examples. To test how many labels it needs, we let it see nested 5/10/25/50/100% subsets of the training set, with ceiling counts within each class. The original labels control class composition, and LR trains on the visible subset. IF256 provides a reference fitted on all training features without fraud labels. The test set and review capacity stay fixed.

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

To see which fields support the ranking, shuffle a single test column and score again with the same fitted model. Shuffling amount lowers AP by 0.3695 on average, the largest drop. Appendix C gives other feature effects and two ranked errors.

\clearpage

## 4.4 What do extra reviews buy?

A team can count the frauds it finds and the money involved in those payments. At 1%, weighted LR covers 86.49% of fraud transactions and 69.59% of their amount. IF4096 covers 55.41% and 47.51%; cluster distance covers 50.00% and 10.71%. The descending-Amount reference finds only 3 frauds but covers 46.56% of fraud amount. The test fraud amount totals 7,727.67 units. This contrast shows how the choice of ranking can depend on whether the team prioritizes the number of cases or the amount involved.

![ULB formal test: count and amount coverage at 1%, and loss across budgets for the fixed FN:FP cost ratio of 50:1.](../results/figures/report_amount.pdf){width=100%}

Under the 50:1 cost scenario, weighted LR has loss 771 at a 0.5% budget and 1,004 at 1%. The larger list gains one detected fraud but adds many normal reviews. With a different assumed error cost, the balance changes. These calculations compare the declared scenarios; the primary experimental budget stays at 1%. Fraud amount is a transaction quantity rather than a measured recovery amount.

**Table 6. What changes when the review budget grows? Weighted LR on ULB.**

| Budget | Reviews | Frauds found | Normal reviews | Fraud amount (%) | Added fraud / normal |
| --- | --- | --- | --- | --- | --- |
| 0.1% | 57 | 50 | 7 | 57.27 | — |
| 0.5% | 284 | 63 | 221 | 69.37 | +13 / +214 |
| 1% | 568 | 64 | 504 | 69.59 | +1 / +283 |
| 2% | 1135 | 65 | 1070 | 69.69 | +1 / +566 |

From 0.5% to 1%, the team reviews 283 more normal transactions and finds one additional fraud. Amount coverage rises from 69.37% to 69.59%. To check the steadiness of the comparison in Table 2, Appendix C resamples the same test rows or half-hour blocks for both fixed models. The block-based 95% interval for LR’s recall lead over IF4096 is 18.69–44.24 percentage points.

\clearpage

## 4.5 Does looking back improve the list?

The ULB file cannot link a customer’s payments. Handbook can, so we ask whether LR and IF benefit from the same additional information. Each method reviews 1,458 of 145,739 payments with 1,280 frauds. Table 7 adds customer and terminal histories to current Amount and cyclical time. The final view also adds 7-day connection counts and a flag for new customer–terminal pairs.

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

The previous test adds summary statistics. A separate experiment adds a pattern based on the customer's last three payments. With or without their order, the pattern raises both-history LR from 334 to 351 detections. Appendix B reports that comparison and a clustering check on which groups the original list misses.

\clearpage

## 4.6 What if labels arrive late or payment history is missing?

Section 4.3 varies how many training labels are visible. In practice, staff learn the outcome of the cases they actually review. To test this feedback loop, we replay Handbook with 100 daily slots and a two-day delay. All policies start with the same weighted LR. Two policies use 10 slots for random or uncertain cases and update only from reviewed labels.

**Table 8. Supplemental replay on Handbook days 45–59.**

| Feedback policy | Reviews | Frauds found | Ranking-only top 100 |
| --- | --- | --- | --- |
| Static ranking | 1500 | 346 | 346 |
| Risk-only updates | 1500 | 345 | 345 |
| Risk + random | 1500 | 343–344 | 345 |
| Risk + uncertainty | 1500 | 344 | 345 |

The static policy finds 346 frauds, versus 343–345 for the updating policies. Even if all 100 slots were assigned by score after an update, the resulting rankings find 345. This replay gives no evidence of a gain from allocating some reviews to collect different labels. Appendix B details the timing and scoring.

![Handbook formal test under missing event histories. Each condition rebuilds past-only features for the same test transactions and scores the same frozen models. Shading is the range across seeds 42/43/44.](../results/figures/report_stress.pdf){width=100%}

A separate test changes the information used to score each payment. With 30% of past events missing, both-history IF falls from 139 to 77–82 detections. LR finds 334–337. The test payments and review capacity stay fixed. Appendix C reports the pre-test label-delay snapshots and the checks for new customers and terminals.

\clearpage

## 4.7 What does the list miss, and does the choice change?

Apriori flags combinations of conditions linked to fraud. On ULB, the first rule flags 3,846 payments including 52 frauds (0.0916% support, 1.35% confidence, 10.37 lift, 0.3581 Kulczynski). The 21-rule union covers all 74 frauds, including the 10 LR misses, but also all 504 normal payments in its shortlist. It flags 56,053 of 56,746 payments, far beyond the 568 slots. The flags are clues for investigation.

On Handbook, a rule flags a payment if its amount is at least 65.22 and at least about 1.20 times the customer’s 7-day average. It flags 28,997 payments including 514 frauds (1.77% confidence, 2.02 lift, 0.2096 Kulczynski). All 19 rules cover 120,751 payments and 1,122 frauds. Table 9 compares that coverage with the history models’ fixed-budget detections by simulator scenario.

**Table 9. Handbook fraud mechanisms; TP from the global 1% history-model lists.**

| Final scenario | Frauds | History LR TP | History IF TP | Covered by rules |
| --- | ---: | ---: | ---: | ---: |
| 1 | 73 | 70 | 7 | 73 |
| 2 | 800 | 4 | 9 | 650 |
| 3 | 407 | 260 | 123 | 399 |

History LR finds 70/73 frauds in scenario 1, 4/800 in the terminal-compromise scenario and 260/407 in scenario 3. IF finds 7, 9 and 123. These known scenario labels identify misses after scoring. A complementary Gaussian mixture model (GMM) groups similar payments by amount and recent activity. The low-amount group has 141 frauds, but receives only two reviews from the same LR list, both normal (Table B2).

The Handbook checks explain where a list misses fraud. PaySim asks whether to choose a different ranking for mobile money. Fit each method separately on PaySim and give it 1,289 reviews out of 128,873 test transactions, including 413 frauds.

**Table 10. PaySim formal test.**

| PaySim method | AP | TP | Precision (%) | Recall (%) |
| --- | ---: | ---: | ---: | ---: |
| Amount | 0.0798 | 94 | 7.29 | 22.76 |
| Weighted LR | 0.0634 | 80 | 6.21 | 19.37 |
| Fixed IF | 0.0447 | 102 | 7.91 | 24.70 |

IF finds the most fraud in 1,289 reviews. The amount ranking leads in AP. The method to use depends on the payment setting and on how far staff can go down the list.

# 5. Conclusion and limitations

Set the review capacity from staffing and compare rankings at that workload. On ULB, the selected weighted LR finds 64/74 frauds in 568 reviews. Doubling the list from 284 reviews adds one fraud and 283 normal payments. On Handbook, check which kinds of fraud receive few reviews and how missing history changes the list. PaySim shows why a team should repeat the fixed-capacity comparison when its payment setting changes. Rules can help investigate individual flags, and grouped results show where more investigation is needed.

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

[8] Hong Kong Monetary Authority. *Supervisory Policy Manual CR-S-5: Credit Card Business*, §5.4, 2006. [Fraud control](https://www.hkma.gov.hk/media/eng/img/key-functions/banking-stability/supervisory-policy-manual/CR-S-5.pdf#page=32).

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
