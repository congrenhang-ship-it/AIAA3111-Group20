# D local work record — 2026-11-09

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
