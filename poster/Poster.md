# Group 20 poster

[Open the poster PDF](Poster.pdf) or edit the [XeLaTeX source](Poster.tex). The layout is one A0 portrait page (841 × 1189 mm). The course guide says that final formatting specifications will be issued separately.

From the repository root, rebuild with XeLaTeX and TeX Gyre Heros installed:

```sh
mkdir -p data/poster_build
(cd poster && xelatex -halt-on-error -interaction=nonstopmode -output-directory=../data/poster_build Poster.tex)
cp data/poster_build/Poster.pdf poster/Poster.pdf
```

Both vector figures come directly from the [report](../report/Report.pdf). The poster's ULB budget, overlap, supervision and rule results correspond to report Sections 4.1–4.3 and 4.7. The Handbook context and missing-history comparisons come from Sections 4.5–4.6. The PaySim comparison comes from Table 10. Review the content and check the separate course formatting notice before submitting.
