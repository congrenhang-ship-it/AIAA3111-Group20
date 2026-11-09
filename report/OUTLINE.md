# Report working outline — D1

Status: outline only; no ULB experiment has been run by D. Do not turn placeholders
into performance claims. Main text target: 8 pages (allowed 6–10); references,
disclosures and appendices are outside the main-text count. Final pagination
must be checked after the real tables and figures are available.

| Planned pages | Shared story and evidence | Lead contribution |
|---|---|---|
| 1 | Problem: finite review capacity; research questions; contributions supported by actual experiments | All; D integrates |
| 2 | Data provenance/hash, audit decisions, chronological split counts; common leakage boundaries | A |
| 3 | Risk-ranking methods and compact settings table: four classifiers, representative imbalance contrasts, K-means and Isolation Forest | B/C |
| 4 | Common evaluation protocol: AP/PR, stable ranking, K rounding, random baseline, validation selection and frozen test protocol | C/A |
| 5 | Same-budget results: PR figure and main budget table; which observed differences matter at the chosen primary budget? | B/C |
| 6 | Complementary fraud coverage and exactly-K combined lists; group and relative-time differences with counts/uncertainty | C/A |
| 7 | Training-only rule discovery and held-out stability; link frozen-rule coverage to actual errors; representative cases | D with B/C |
| 8 | Limitations, justified conclusions, reproducibility summary | All |

Keep algorithm descriptions short so equal-budget evidence drives the narrative.
If 8 pages become crowded, allow up to 10 rather than shrinking text excessively.
If rules show no reliable generalization, report that outcome and its counts.

## Planned evidence register (nothing is populated yet)

| ID | Required artifact | Owner | Source needed |
|---|---|---|---|
| T1 | Train/validation/test transactions, fraud counts and time boundaries | A | dataset SHA-256, split commit |
| T2 | Frozen method configurations and validation selection criteria | A/B/C/D | G2 manifest, code commit |
| T3 | AP and top-K TP/FP/precision/recall, with K explicit | C | common final-test tables |
| T4 | Rule training/validation/test counts and metrics | D | frozen bins/rule IDs, split provenance |
| F1 | Overall PR and review-budget comparison | C | common final-test predictions (local only) |
| F2 | Equal-K overlap and combination results | C | selected quota and deduplication policy |
| F3 | Group and relative-time results with sample counts | C/A | group/global budget labels, frozen windows |

## Integration checklist for later D6

- Each author supplies methods, actual findings, limitations, source table/figure,
  command, environment, data hash and code commit for their module.
- Clearly distinguish training, validation and final test in filenames/captions.
- All K values and model names must agree across report, poster and common tables.
- V1–V28 retain their anonymous names; rules show association, not causality.
- Time is relative elapsed time; two days do not establish long-term drift.
- Collect each member's actual external collaboration and AIGC usage. D's initial
  record is in D_PROGRESS.md; do not assume other members used the same tools.
- No final export before G3 evidence/reproduction checks pass.
