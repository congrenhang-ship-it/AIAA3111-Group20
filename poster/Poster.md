# Group 20 poster

[Poster PDF](Poster.pdf) · [HTML source](Poster.html) · [Figure sources and checksums](FIGURE_MANIFEST.json)

The course requires one portrait page, 80 cm wide × 100 cm high. The PDF page size is 800 × 1000 mm.

Use `Poster.pdf` for viewing, printing and submission. The static HTML source is used to rebuild the PDF.

The opening six-transaction example illustrates how equal-length review lists work. The right-side headline gives the held-out ULB result. The poster uses a two-column layout, with the protocol and ULB comparisons on the left and the feature, dataset and diagnostic studies on the right.

| Poster section | Evidence in the report |
| --- | --- |
| Evaluation workflow and data partitions | Sections 2–3 and Table 1 |
| Eight ULB rankings at 568 reviews | Table 2 |
| Capacity, union of two lists and recall interval | Figure 1, Sections 4.2 and 4.4, and Table C1 |
| Handbook feature sets and missing history | Figures 4–5 and Sections 4.5–4.6 |
| PaySim leaderboard | Table 10 |
| Label availability, Apriori example, review feedback and group allocation | Sections 4.3 and 4.7 and Tables 8 and B2 |

The printed QR code points to the project repository.

## Rebuild the poster

From the repository root, use the [ARIS paper-poster-html scripts](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/tree/9f30e834d24a337dbdd06fb20a1150d4376dfa59/skills/paper-poster-html) with Python, Playwright Chromium and PyMuPDF installed. Set `POSTER_SKILL` to the local checkout's `skills/paper-poster-html` directory.

```sh
mkdir -p data/poster_build
python "$POSTER_SKILL/scripts/run_gates.py" poster/Poster.html \
  --tokens "$POSTER_SKILL/templates/tokens/generic.json" \
  --manifest poster/FIGURE_MANIFEST.json --report data/poster_build/GATE_REPORT.json
python "$POSTER_SKILL/scripts/render_preview.py" poster/Poster.html \
  --pdf poster/Poster.pdf --png data/poster_build/html_preview.png
# Set the exact page bounds after Chromium rounds them to device units.
python - <<'PYTHON'
import pymupdf

with pymupdf.open("poster/Poster.pdf") as pdf:
    pdf[0].set_mediabox(pymupdf.Rect(0, 0, 800 * 72 / 25.4, 1000 * 72 / 25.4))
    pdf.saveIncr()
PYTHON
python "$POSTER_SKILL/scripts/poster_check.py" verify-final poster/Poster.pdf \
  --from-html poster/Poster.html --max-size-mb 20
```

After editing the report figures, re-export the three poster images with `pdftoppm -r 750 -png -singlefile` and update their source and image hashes in the manifest.

The HTML adapts ARIS's portrait template and generic design tokens, originally derived from posterly (MIT, © 2026 Ruishuo Chen). [License](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/9f30e834d24a337dbdd06fb20a1150d4376dfa59/skills/paper-poster-html/LICENSES/posterly-MIT.txt)
