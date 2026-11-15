# D4 common-environment verification and A handoff — 2026-11-13

**D4 technical verification passed.** The fixed candidate can be included in A7.
Group G2 remains false. A6's main-branch workflow, D5 final reproduction and
A8 test evaluation are outstanding. No test model/rule scores were produced.

## Integration and environment

An isolated integration copy uses C commit `a936f29`, which includes A/B. D's
pattern_mining.py and seven checks were overlaid there. No A/B/C files were
edited. This is a local compatibility check, not an A6 main-branch merge.
Latest main's TASKS.md was merged into D's branch; main was not changed by D.

The isolated `.venv-d-common` uses Python 3.12.14 and all 22 exact packages from
C's requirements.txt, including numpy 2.5.4, pandas 3.1.1, sklearn 1.9.1 and
scipy 1.18.1. `uv pip check` reports compatible packages. The existing Python
environment was not upgraded. Data, private runs and environments are ignored.

## Reproduce the comparison

From D's repository root, with the integration worktree alongside it:

```powershell
$env:PYTHONUTF8 = '1'
.venv-d-common/Scripts/python.exe code/run_d_validation.py --preprocessing-code ../AIAA3111-C-check/code --output data/d_common_replay
.venv-d-common/Scripts/python.exe code/check_d_freeze.py --replay data/d_common_replay --lock ../AIAA3111-C-check/requirements.txt
```

The first command repeats the original declared training/validation protocol.
The second compares against the originally published candidate and fails on
changed identities/rules/counts or floats beyond 1e-12. The export preserves
the ORIGINAL bins/rules in `results/tables/d_freeze.json`. No thresholds or
selection criteria were changed.

- Raw SHA-256 and all three ordered split fingerprints match A.
- Features remain Amount, V17, V14, V12 and V3.
- All 63 candidate training/validation records match within 1e-12.
- Equal-width remains 38 candidates/15 passes; equal-frequency 25/21.
- All 21 retained rule identities/antecedents match exactly.
- Maximum absolute training-boundary difference: **0.0**.
- All 22 installed package versions match the common lock.

## Checks and Windows encoding

The first integration run passed 52/53 checks; one C B-reproduction check failed
because `Path.read_text()` used Windows GBK to read UTF-8 Chinese comments in
requirements.txt. With `PYTHONUTF8=1`, all 53 A/C/D checks pass with no skips.
B's separate eight checks pass as well: 61 total. No C source was patched.

From the integration worktree, use the isolated interpreter and UTF-8 setting:

```powershell
$env:PYTHONUTF8 = '1'
../AIAA3111-Group20/.venv-d-common/Scripts/python.exe -m unittest discover -s tests -v
../AIAA3111-Group20/.venv-d-common/Scripts/python.exe -m unittest discover -s code -p test_classification.py -v
```

A should retain the UTF-8 setting in Windows instructions or have the responsible
module owner specify text encoding explicitly.

## Apply the fixed configuration

```python
import json
from pattern_mining import validate_rules
with open("results/tables/d_freeze.json", encoding="utf-8") as source:
    frozen = json.load(source)
metrics = validate_rules(validation, frozen["frozen_binner"], frozen["frozen_rules"])
# A supplies the shared split. Only after G2: use the SAME objects on test.
```

Keep covered_row_ids local for joining to model predictions; strip them from
public exports. Do not sum overlapping fraud counts. Rules remain diagnostic
associations, not an exactly-K review strategy. G2 confirmation remains A's.

## D5/D6 progress

The real G1 original logistic-regression validation entry independently ran from
the integration copy. AP=0.6424292493857157 and all four budget records match
B's published G1 JSON exactly. The compact record is
`results/tables/d_g1_reproduction.json`; full PR arrays remain local. This is
not a claim that A6's forthcoming all-module entry has been reproduced.

D5 still needs A6/A7/A8, full frozen-flow reproduction and final error linkage.
D6 now has Report.md and poster/Poster.md content drafts using existing sources.
Final test results, cases, author review/disclosures and typeset page/layout
checks are outstanding. D7 awaits G3.

Codex assisted with environment setup, replay/comparison tooling, checks and
editorial compilation. Authors must review their content and disclosures.
