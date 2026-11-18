# Group 20 poster

[Poster PDF](Poster.pdf) · [HTML source](Poster.html) · [Figure sources and checksums](FIGURE_MANIFEST.json)

The print layout is one A0 portrait page (841 × 1189 mm). The course guide says the final formatting specifications will be released separately.

The poster uses Figures 1 and 4 from the report. Its ULB budget, overlap, labels, cost and rule results come from Sections 4.1–4.4 and 4.7. The Handbook comparisons come from Sections 4.5–4.7. The PaySim results come from Table 10.

## Rebuild the poster

From the repository root, use the [ARIS paper-poster-html scripts](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/tree/9f30e834d24a337dbdd06fb20a1150d4376dfa59/skills/paper-poster-html) with Python, Playwright Chromium and PyMuPDF installed. Set `POSTER_SKILL` to the local checkout's `skills/paper-poster-html` directory.

```sh
mkdir -p data/poster_build
python "$POSTER_SKILL/scripts/run_gates.py" poster/Poster.html \
  --tokens "$POSTER_SKILL/templates/tokens/generic.json" \
  --manifest poster/FIGURE_MANIFEST.json --report data/poster_build/GATE_REPORT.json
python "$POSTER_SKILL/scripts/render_preview.py" poster/Poster.html \
  --pdf poster/Poster.pdf --png data/poster_build/html_preview.png
python "$POSTER_SKILL/scripts/poster_check.py" verify-final poster/Poster.pdf \
  --from-html poster/Poster.html --max-size-mb 20
```

After editing the report figures, re-export the two poster images with `pdftoppm -r 750 -png -singlefile` and update their source and image hashes in the manifest.

The HTML adapts ARIS's portrait template and generic design tokens, originally derived from posterly (MIT, © 2026 Ruishuo Chen). [License](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/9f30e834d24a337dbdd06fb20a1150d4376dfa59/skills/paper-poster-html/LICENSES/posterly-MIT.txt)
