# D validation protocol — declared before the real rule run, 2026-11-11

Dependency: A's unmodified `Task_A` commit
`26c1c32e3b8dad50525298842c4448a78a9ac99c`, currently not in main. Run A's
download/checksum and split code from a detached worktree; do not copy or edit
A's implementation or create a new split. This is a dependency-branch integration
run, not evidence that G1/G2/G3 or the main-branch pipeline has passed.

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

Use the existing local Python/numpy/pandas first and record actual versions.
This does not certify A's Python 3.12 pinned environment; repeat there before
claiming that particular environment was reproduced.

Command from D's repository root (A worktree is alongside this repository):

```bash
python code/run_d_validation.py --preprocessing-code ../AIAA3111-Task-A-check/code
```

The source commit and SHA-256 values in the manifest identify this implementation.
The initial protocol commit precedes the experiment's aggregate result commit.
