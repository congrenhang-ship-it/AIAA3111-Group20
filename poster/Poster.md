# Group 20 poster

[Poster PDF](Poster.pdf) · [HTML source](Poster.html) · [Figure sources and checksums](FIGURE_MANIFEST.json)

The print layout is one A0 portrait page (841 × 1189 mm). The course guide says the final formatting specifications will be released separately.

Open `Poster.html` in a browser with the `assets/` folder beside it. Use the link below the headline to compare four ULB review budgets or five Handbook feature sets. The ULB control switches the bars between recall and precision. Each report chart can be enlarged, and the metric explanations can be expanded. These controls run offline. Use `Poster.pdf` for printing.

The poster uses Figures 1, 4 and 5 from the report. Its ULB results come from Sections 4.1–4.3 and 4.7. The Handbook comparisons come from Sections 4.5–4.6. The PaySim results come from Table 10. The HTML controls use aggregate values embedded from `ulb_test.json` and `handbook_test.json` in `results/tables/`.

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

After editing the report figures, re-export the three poster images with `pdftoppm -r 750 -png -singlefile` and update their source and image hashes in the manifest.

The HTML adapts ARIS's portrait template and generic design tokens, originally derived from posterly (MIT, © 2026 Ruishuo Chen). [License](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/9f30e834d24a337dbdd06fb20a1150d4376dfa59/skills/paper-poster-html/LICENSES/posterly-MIT.txt)
