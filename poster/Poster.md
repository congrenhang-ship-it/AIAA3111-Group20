# Group 20 poster

[Poster PDF](Poster.pdf) · [HTML source](Poster.html) · [Figure sources and checksums](FIGURE_MANIFEST.json)

The course requires one portrait page, 80 cm wide × 100 cm high. The PDF page size is 800 × 1000 mm.

Use `Poster.pdf` for viewing, printing and submission. The static HTML source is used to rebuild the PDF.

The opening example shows how six hypothetical payments compete for two review slots. The main ULB test result and the wider review-budget panel connect that example to a staffing decision. Smaller panels cover uncertainty, history, PaySim and missed cases.

| Poster section | Evidence in the report |
| --- | --- |
| 1–2. Evaluation workflow, data partitions and ULB rankings | Sections 2–3 and 4.1, Tables 1–2 |
| 3. Review capacity, list overlap, score cutoffs and cost | Sections 4.2 and 4.4, Tables 4 and 6 |
| 4. Uncertainty in the gap between models | Table C1 |
| 5. History, missing events, label delay and cold-start slices | Sections 4.5–4.6, Figures 4–5 and Appendices B–C |
| 6. PaySim ranking | Table 10 |
| 7. Label availability, feature importance, rules, missed groups and feedback | Sections 4.3, 4.6 and 4.7, Figure 2, Tables 8, 9 and B2, and Appendix C |

The printed QR code points to the project repository.

## Rebuild the poster

From the repository root, use the [ARIS paper-poster-html scripts](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/tree/9f30e834d24a337dbdd06fb20a1150d4376dfa59/skills/paper-poster-html) with Python, Playwright Chromium and PyMuPDF installed. Set `POSTER_SKILL` to the local checkout's `skills/paper-poster-html` directory.

```sh
mkdir -p data/poster_build
python "$POSTER_SKILL/scripts/render_preview.py" poster/Poster.html \
  --pdf poster/Poster.pdf --png data/poster_build/html_preview.png
# Set the exact page bounds after Chromium rounds them to device units.
python - <<'PYTHON'
import pymupdf

with pymupdf.open("poster/Poster.pdf") as pdf:
    assert len(pdf) == 1
    pdf[0].set_mediabox(pymupdf.Rect(0, 0, 800 * 72 / 25.4, 1000 * 72 / 25.4))
    pdf.saveIncr()
PYTHON
python "$POSTER_SKILL/scripts/poster_check.py" verify-final poster/Poster.pdf \
  --from-html poster/Poster.html --max-size-mb 20
```

The manifest identifies the three plots used on the poster. After editing a source figure, export it at 750 dpi and update its source and image hashes. The budget comparison bars show the counts in Table 6.

The HTML adapts ARIS's portrait template and generic design tokens, originally derived from posterly (MIT, © 2026 Ruishuo Chen). [License](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/9f30e834d24a337dbdd06fb20a1150d4376dfa59/skills/paper-poster-html/LICENSES/posterly-MIT.txt)
