# D local work record — 2026-11-09

## Latest update — 2026-11-14

The new task list introduces D-E1/E04. It is now implemented and run on the
unchanged 21 fixed rules in the common environment: pairwise Jaccard,
training-ordered cumulative/unique coverage, redundancy and B/C validation
error linkage. Ten D checks pass. Shared results are in d_extensions.json,
two validation-labelled figures and the existing D_VALIDATION_RESULTS.md.
The broad union covers 56,055 transactions and 57 frauds, so error coverage
must not be claimed as a limited-budget improvement. Full predictions remain
local; no test set scores, new rule search or new model selection occurred.

Report.md and poster/Poster.md now incorporate E04 findings and distinguish
other extension protocols from completed experiments. A second 2026-11-14
main update (51e543a) confirms C-E1/E2/E3 delivery; the shared checklist retains
C's new state together with D-E1 completion. C's new amount, cost and paired
interval validation evidence has also been compiled from C commit 8818b20.
D5 still requires A6
and A8, D6 remains an incomplete content draft, and D7 awaits G3. D-E1 alone
is marked complete in the current remaining-task checklist. Codex assistance
for the new implementation/checks, figures and writing is recorded in the
existing validation analysis and must be reviewed in final disclosures.

## Previous update — 2026-11-13

D4 now passes in the exact common Python 3.12.14 / 22-package environment.
All 63 candidate records and 21 retained rules match; maximum boundary drift
is zero. See D_COMMON_ENVIRONMENT.md and results/tables/d_freeze.json for the
checkable A7 handoff. The integrated local A/B/C/D copy passes 53 checks with
PYTHONUTF8=1, and B's eight separate checks pass. The initial GBK decoding
failure and its environment-only resolution are recorded, not omitted.

D5's independent real G1 AP and four-budget comparison matches B's record.
Full A6/A8 reproduction and final cross-method cases remain pending. D6 has
integrated methods/validation content drafts in report/Report.md and
poster/Poster.md. These are Markdown content drafts, not paginated final
documents or exported PDFs. Test results, author review and G3 remain pending.
Only D4 is checked in the current remaining-task checklist. No main edit or
test performance evaluation was made. Codex assistance for this work is recorded
in the common-environment handoff and draft disclosures.

## Previous update — 2026-11-11

A3 is now implemented in A's `Task_A` commit `26c1c32`, but has not been merged
into main. D used that unmodified commit from a detached worktree, downloaded
the pinned CSV independently and matched the raw SHA-256 and all three ordered
split fingerprints. A's five tests and D's seven tests passed, including A's
full-data check. The run used Python 3.13.5/numpy 2.1.3/pandas 2.2.3, not the
Python 3.12 pinned environment; that environment check is still outstanding.

D2/D3 real training and validation are now locally completed against this
explicit dependency branch. The predeclared protocol commit is `8b6e689`.
Equal-width produced 38 candidates / 15 validation passes; equal-frequency
produced 25 / 21. D4's candidate manifest freezes all 21 passing equal-frequency
rules. It awaits A's G2 confirmation and main-branch integration. There has been
no rule evaluation on test. See D_VALIDATION_RESULTS.md and d_validation/.

D5 now has independent evidence for A's data preparation, but G3 reproduction,
classification/anomaly error linkage and D6/D7 remain incomplete. Do not tick
other members' tasks or group milestones on the basis of this run.

Codex assisted with dependency inspection, download, protocol/runner preparation,
test execution and this validation analysis on 2026-11-11; retain this actual
assistance in D's disclosures. No main-branch edits were made.

## Historical snapshot — 2026-11-09

The statuses below are the earlier snapshot, superseded by the update above.

## Scope and status

Repository: https://github.com/congrenhang-ship-it/AIAA3111-Group20

Baseline: `5bde9fc28d488455ea385ebd6dbfbe11c966e596`.
Working branch: `codex/d-pattern-mining`.

| Task | Actual status | Evidence / remaining condition |
|---|---|---|
| D1 | Implementation and local checks complete; task branch pushed, PR pending | pattern_mining.py, tests, report/OUTLINE.md, poster/OUTLINE.md, RULE_METHOD.md |
| D2 | Training binning/Apriori code prepared on synthetic data; real experiment pending | A3 split_data still raises NotImplementedError; no fixed raw data version/hash |
| D3 | Held-out application code checked on synthetic data; real stability analysis pending | requires D2 and the unique validation split |
| D4 | Not frozen | no real selected feature set, boundaries or surviving rules |
| D5 | Not complete | --help/--tasks work, but the shared experimental flow is not implemented |
| D6/D7 | Outlines only | await authors' evidence, G2/G3; no final report/poster/PDF |

TASKS.md remains unchecked pending the collaboration PR. The user supplied Git
author/committer identity has been configured for this repository only. D1's
implementation commit `74af196` was pushed successfully to the task branch
after repository access was granted. No PR, Issue or notification has been made.
The task branch remains separate from main.

## Files and checks

- `code/pattern_mining.py`: existing signatures retained, training-only bins,
  bounded Apriori, training metrics and frozen-bin held-out metrics/row IDs.
- `tests/test_pattern_mining.py`: synthetic standard-library unittest checks.
- `report/RULE_METHOD.md`: formulas, zero-count/boundary behavior, schemas,
  proposed limits and the remaining A3 → validation → G2 handoff.
- `report/OUTLINE.md`, `poster/OUTLINE.md`: shared narrative and evidence requests;
  no illustrative performance numbers or invented results.

Commands run from repository root:

```text
python -m unittest discover -s tests -p test_pattern_mining.py -v
Result: Ran 7 tests; OK.

python code/main.py --help
Result: help displayed; exit 0.

python -X utf8 code/main.py --tasks
Result: ownership displayed; exit 0. UTF-8 flag avoids this console's encoding mismatch.

git diff --check
Result: no whitespace errors.
```

Checks include hand-derived support=1/3, confidence=2/3, lift=2,
Kulczynski=5/6; exhaustive short-itemset comparison; distinct width/quantile
boundaries; constant/repeated values; held-out extremes and exact boundary
equality; no mutation/refitting; row identities; rare/empty/no-fraud/no-match cases;
and strict JSON serialization. These are program checks, not ULB results.

## Actual local environment and reproduction limits

- Windows PowerShell; Python 3.13.5; numpy 2.1.3; pandas 2.2.3.
- README recommends Python 3.12. These checks used the existing 3.13 environment;
  they do not establish that the full shared requirements work under 3.12.
- No new packages were installed or dependencies changed. Only the rule module's
  numpy/pandas path was verified; the other members' complete stack was not.
- No raw CSV downloaded/used; data version/hash and real split are not applicable
  to this synthetic check and must be supplied by A before D2 experiments.
- A/B/C-owned code and common interfaces were not modified. The new tests only
  exercise D's rule module; report/poster materials remain Markdown drafts.

## Next work, without bypassing dependencies

1. Open a PR from the pushed D1 task branch to main for collaboration review.
   Suggested title: `D1: prepare rule interfaces and document outlines`.
2. When A3 lands, use the shared split; record hash/counts/commit and select a
   small feature set using training data only. Run both proposed binning schemes.
3. Follow RULE_METHOD.md for actual validation, record negative results as well,
   freeze the selected set and hand A the reproducible G2 manifest.
4. Resume D5–D7 using actual public outputs and authors' contributions. Do not
   create a final PDF or tick off real experiments from synthetic program checks.

## Actual AIGC assistance record (D draft)

On 2026-11-09 Codex assisted with reading TASKS/README and existing interfaces,
implementing the rule module, writing/running synthetic checks, drafting the
methods/outline documents and recording local progress. No real experimental
results were generated. D should review the implementation and update this
record with their own actual validation and subsequent tool use before inclusion
in the report's Disclosures. Other members supply their own records.
