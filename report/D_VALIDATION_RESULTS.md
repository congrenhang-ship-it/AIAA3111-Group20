# 规则方法与开发验证证据

整理：2026-11-17。本文合并原规则方法、预声明协议和验证发现。以下带日期的记录属于历史开发证据；正式冻结和测试现已完成，见 [G2](../results/tables/group_freeze.json)、[ULB 正式规则诊断](../results/tables/ulb_test.json)、[Handbook 正式规则与机制](../results/tables/handbook_d_test.json)及 [独立复现](../results/tables/reproduction.json)。历史段落的“待 G2/A8”不是当前任务状态。

## 原预声明协议（2026-11-11）

Selection, declared before observing validation rule performance:

1. Verify A's raw SHA-256 and all ordered split fingerprints. Test counts/IDs are
   checked for dataset identity only; do not apply/evaluate any rule on test.
2. Always include Amount. Select four V features by descending absolute Pearson
   correlation with Class in training only; lexical name tie break. This is
   supervised training feature selection, not unsupervised selection. Class is
   never discretized as an input or placed in an antecedent.
3. Compare only equal-width vs equal-frequency, three requested bins each,
   maximum rule length three including Class=1. Reuse exactly the same features.
4. Keep the existing training limits: antecedent coverage >=20, fraud coverage
   >=5, lift >1, at most 50 rules per scheme in deterministic training order.
5. Validate every candidate with its fixed training bins. Retain rules with
   validation coverage >=20, fraud coverage >=5 and lift >1. No threshold changes
   after seeing results; retain zero rules if none pass.
6. Choose the scheme with more validation-passing candidates, preferring
   equal-frequency on a tie. Freeze all passing rules from that scheme. This
   criterion targets diagnostic rule availability; redundant rules may overlap
   and their counts must not be interpreted as additional detected fraud.
7. Export all candidates' aggregate training/validation metrics and the matching
   bins, configuration and surviving rule IDs. Keep full covered-ID lists local.
   A must confirm this candidate freeze for G2; final test evaluation remains A's.

## 方法与统计定义

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

# D training / validation analysis — 2026-11-11

## E04 extension update — 2026-11-14

The original 21 fixed rules and bin boundaries are unchanged. Under the
2026-11-14 extension protocol, training lift/confidence/length/ID fixes their
order as R001–R021. A paired coverage analysis produces
`results/tables/d_extensions.json` plus the validation-labelled overlap heatmap
and cumulative coverage figure in results/figures/. Python 3.12.14 and the
common dependency environment were used. Validation has already been viewed;
this is development-stage diagnostic evidence, not a fresh held-out experiment.

| Sequential checkpoint | Unique transactions | Unique frauds |
|---|---:|---:|
| R001 | 4,556 | 39 |
| Through R002 | 20,807 | 53 |
| All 21 rules | 56,055 | 57 |

The full union covers **98.78% of all validation transactions**, including
55,998 normal transactions. Covering every validation fraud therefore does not
establish useful limited-budget detection. There are 15 rules with zero new
frauds at their position and R020 adds no transactions beyond the previous
union. No pair has identical coverage or Jaccard >=0.9, yet cumulative redundancy
still occurs: pairwise similarity and redundancy relative to a union differ.
There are 29 antecedent-containment pairs. All pairs and increments are retained.

The predeclared compact display is R001/R002/R005/R006/R009. It omits R013,
which later adds another fraud, so it must not be described as preserving the
entire full-set fraud coverage. All 21 original rules remain the test set;
the five-rule subset is only an editorial display selected on validation.

### Link to fixed model errors at K=568

Other members' private predictions are not in Git. The existing public B/C
interfaces therefore rebuilt only the fixed class-weighted LR and improved IF
(100 trees, sample4096, seed42); no candidate/parameter/quota search was rerun.
Their primary-budget TP values reproduce 49/24 respectively. Full combined
predictions remain local in data/d_extensions/predictions.csv; the public
summary records its SHA-256 and source code/environment/partition provenance.

| Error group | All model errors | Covered by any fixed rule |
|---|---:|---:|
| Classifier false negatives | 8 | 8 |
| Classifier false positives | 519 | 516 |
| IF false negatives | 33 | 33 |
| IF false positives | 544 | 538 |

These are union counts, not sums over rules. The broad union includes both
missed frauds and most reviewed normal transactions. It supports descriptive
error linkage, not an efficient repair of either model's review list.

Eight validation examples follow the predeclared row-ID selection order and
are in the shared JSON with source row IDs, ranks and covering rules. Example
D-E04-1 (source row 176049) is fraud ranked 11,505 by the classifier and 867 by
IF, covered by R013/R016/R021. Neither model reviews it at K=568; IF would reach
it at the supplementary 2% capacity. Example D-E04-7 (source row 170937) is a
normal transaction ranked 321 by IF and 56,714 by the classifier, also covered
by R013/R016/R021. The same rule combination can cover missed fraud and normal
anomalies. Anonymous features do not justify assigning a business cause.

### Reproduce and hand off to A7

```powershell
.venv-d-common/Scripts/python.exe -X utf8 code/run_d_validation.py --extensions --preprocessing-code ../AIAA3111-C-check/code
.venv-d-common/Scripts/python.exe -m unittest discover -s tests -p test_pattern_mining.py -v
```

The dependency worktree is the same A/B/C source used for D4, commit a936f29;
source hashes identify exact modules. E04 code/protocol precedes the real result
commits. Ten D checks pass (three new hand/empty/training-order checks). Both
figures were rendered and visually inspected for labels and readability.
A7 should record the rule order, >=0.9 overlap mark, first-five-positive-increment
display policy and deterministic two-case-per-error-category selection. The
same frozen-rule coverage helper can be applied to A's test data ONLY after G2;
the runner's E04 CLI intentionally uses training/validation only. Do not run a
new search or choose a new display set from test results. No test scores were
generated. A6/full final reproduction remains D5 work, not this validation run.

Codex assisted with fixed-rule diagnostics, checks, fixed model replay, figures,
case linking and drafting; retain the actual assistance in final disclosures.

## Earlier D2–D4 record

## What can continue now

A's unified data preparation is available in `Task_A` at
`26c1c32e3b8dad50525298842c4448a78a9ac99c`. Main is still `5bde9fc`, so this is
an explicit dependency-branch integration run. No A/B/C code or main was changed.
Protocol and runner were committed before running the experiment (`8b6e689`).

The Zenodo CSV downloaded independently and matched A's SHA-256
`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.
A's unmodified split removed 1,081 exact duplicate records. All three ordered
row-ID fingerprints matched A's published values.

| Split | Transactions | Fraud | Usage here |
|---|---:|---:|---|
| Training | 170,235 | 342 | feature selection, bin fitting, rule discovery |
| Validation | 56,745 | 57 | fixed-rule checks and candidate-scheme selection |
| Test | 56,746 | 74 | identity/count check only; no rule scoring |

This independently reproduces the data preparation, not the full G1/G3 pipeline
or A's pinned Python 3.12 environment. Actual versions: Python 3.13.5,
numpy 2.1.3, pandas 2.2.3. A's five checks passed, including its full-data test;
D's seven checks passed. The aggregate export was also checked for rule-set
consistency and absence of full covered-row-ID lists.

## Rule protocol and results

Training-only selection chose Amount plus V17, V14, V12 and V3 by the declared
absolute Pearson correlation criterion. Training labels are used in this
supervised selector; Class is never a rule antecedent or numeric input item.
Both schemes used the same features, three requested bins and one or two
antecedent items. See 本文件的“原预声明协议” for all thresholds and tie breaks.

| Scheme | Training candidates | Pass validation gate | Rejected |
|---|---:|---:|---:|
| Equal-width | 38 | 15 | 23 |
| Equal-frequency | 25 | 21 | 4 |

The validation gate requires at least 20 covered transactions, at least 5 covered
frauds and lift greater than 1. The predeclared selection criterion was the
number of passing rules, so equal-frequency is the candidate frozen scheme.
The exact bins and all 21 passing antecedents/IDs are in
[d_validation/manifest.json](d_validation/manifest.json); all 63 candidates and
their training/validation counts, metrics and changes are in
[d_validation/train_validation_rules.csv](d_validation/train_validation_rules.csv).
No rejected rule or failed scheme was dropped from that evidence table.

## Interpretation for the report

As a concrete validation example, equal-frequency R001 is
`V12 < -0.2537695828426037 AND V17 < -0.28451140583566836`.
Its training coverage is 11,722 transactions including 269 frauds (lift 11.42).
On validation it covers 4,556 transactions including 39 frauds and 4,517 normal
transactions: confidence 0.856%, fraud recall 68.42%, lift 8.52. This is an
association under fixed bins, not an efficient review list at the project's
small budgets. The decrease in lift also shows that training strength does not
transfer unchanged to the later validation period.

Rule count alone is not evidence that equal-frequency is higher quality for
fraud review. Equal-width has fewer validation-passing rules, but some are much
more concentrated: validation example R023 covers 46 transactions including
31 frauds (confidence 67.39%). This example is descriptive validation analysis,
not a change to the declared scheme-selection criterion. Neither scheme has
been evaluated on test, and rule performance is not compared to a trained model
here. The current manifest is a diagnostic candidate freeze for G2 review.

Rules overlap and may be redundant; their fraud counts must not be added.
Five frauds is only a minimum gate, not a significance or multiple-testing
correction. Later G2 confirmation must retain a clear distinction between
diagnostic associations and any actual exactly-K ranking strategy. V features
have no inferred business meaning; no causal explanation is claimed.

## Reproduction and handoff

Use the command in 本文件的“原预声明协议” and A's pinned worktree. The manifest
records actual commits, source hashes, data identity, environment, selected
features, exact boundaries, out-of-range validation counts and surviving rules.
It contains aggregate information only. The raw CSV and full coverage IDs are
not committed. The runner's default local preprocessing is still a stub until
A3 is integrated; the explicit `--preprocessing-code` argument is required now.

Remaining: merge/confirm A3 through the group process, reproduce the pinned
environment, obtain A's G2 confirmation, then let A evaluate frozen rules on
test. D5 still needs the common final workflow and classification/anomaly errors;
D6/D7 still need the other authors' evidence, integration checks and G3.


## 实际 AIGC 使用记录

## Actual AIGC assistance record (D draft)

On 2026-11-09 Codex assisted with reading TASKS/README and existing interfaces,
implementing the rule module, writing/running synthetic checks, drafting the
methods/outline documents and recording local progress. No real experimental
results were generated. D should review the implementation and update this
record with their own actual validation and subsequent tool use before inclusion
in the report's Disclosures. Other members supply their own records.

后续验证中 Codex 协助依赖检查、下载、规则固定重放、图表、错误关联及材料整理。2026-11-17 又协助统一冻结、正式测试、独立复现与仓库整理。成员须核对并补充各自实际使用情况，不能把这些记录当作已经完成的最终报告披露。
