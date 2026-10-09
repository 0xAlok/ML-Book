# Review log — Chapter 22: What is ML: data, models, tasks; supervised vs unsupervised

Branch: `draft/what-is-ml`. Reviewer: chapter author (self-review pass, 2026-10-09).
Sources re-read before writing: all six MLF Week 1 PPT decks (pdftotext; decks 3–6 are
LaTeX-image-heavy, numeric examples recovered from the transcripts) and all six
`Transcripts/Week 1/` PDFs (full reads — the primary authority). `Week 1 Tutorial-1.docx`
contains only an external slides.com link, no content — unusable as a source (thin-source
point, below).

## What was checked

1. **Every worked example recomputed independently** (Python cross-check, all match
   the chapter): eg 7 — $L(f)=0.064$, $L(g)=5.264$; eg 8 — $L(f)=0.22/6\approx0.0367$,
   $L(g)=37.62/6=6.27$; eg 9 — $L(f)=0$, $L(g)=1/6$ (the $(0,1)$ point); eg 10 —
   $L(f)=L(g)=0$, $L(h)=3/6$; eg 11 — bad pair $60.8/4=15.2$, good pair $0.08/4=0.02$;
   eg 12 — $4\ln10\approx9.21$, $4\ln5\approx6.44$, $P_3\to\infty$; eg 5 — $0.5$
   (5 lakhs); eg 6 — $-4<1$ Close, $5\not<1$ Far. Solutions: P5 $L(k)=8.67/5=1.734$,
   P6 $L(m)=5/6$, P8 third pair $0.02$ (identical round-trip to the good pair),
   P9 $2\ln5\approx3.219$ vs $2\ln2\approx1.386$.
2. **Every § cross-reference verified programmatically** (script extracted all
   `## X.Y` headers from `chapters/*.md` and all § refs from the chapter): 16 refs,
   0 missing — §20.1/§20.2/§20.8, §21.5/§21.6, §22.2–§22.13 all real. Forward refs
   to Chapters 23–25 are plain prose ("Chapter 23"), not §-refs, matching OUTLINE.md.
   Problem set (10) and solutions file (10) numbering match.
3. **Scope discipline**: no normal equations / least-squares solver (Ch 23), no PCA
   variance-maximization (Ch 24), no GMM/EM fitting (Ch 25) — GMM appears only as
   the lecture's own qualitative pointer. NLL uses the deck's sum convention,
   matching §20.2's risk $R(\theta)$.
4. **Figures inspected visually**: hierarchy diagram, regression scatter ($f$ through
   points, $g$ missing), classification boundaries ($f$: $x_1=2$ vertical, $g$:
   $x_1=2x_2$ slanted, one misclassified point), dim-red reconstructions, taxonomy
   tree. All original matplotlib, marked as original in HTML comments.
5. **Style**: no emojis; all math terms in LaTeX; `=` definitions, i)/ii)/iii)
   points, `Note:` callouts, worked `eg` blocks, 11 "Basically, ..." simplifications.

## What was fixed during review

- "wisdomofchopra.com" → "wisdomofchopra.com" (deck's spelling, both decks).
- §22.6(iii): garbled power-vs-superscript sentence rewritten to the lecture's
  actual convention ($(x^1)_2$ parenthesized = square of coordinate vs $x^2_1$ =
  first coordinate of second vector).
- §22.9: stray space inside `$...$` around $\mathbf{w}^T\mathbf{x}+b$ removed.

## Thin / contradictory source points

- **Transcript files mislabeled**: `Transcripts/Week 1/5. ... Dimensionality Reduction.pdf`
  is actually the density-estimation transcript (opens "we wrap up the dimensionality
  reduction problem and move on to density estimation"), and file 6 is the
  dimensionality-reduction one. Content is complete; only the filenames are swapped.
- **Tutorial docx is content-free**: `Week 1 Tutorial-1.docx` = one slides.com link.
  No tutorial exercises were available from the dump; the problem set is built from
  the decks' worked examples instead (new candidates $k$, $m$, third encoder pair,
  $P_4$, taxonomy-classification questions).
- **Density-estimation data typo**: transcript writes the second point as $2.6$ once,
  but the deck figure and the transcript's own earlier listing say $2.7$; chapter uses
  $2.7$ and notes the discrepancy.
- **Classification transcript listing jumbled**: the per-point $g(\mathbf{x})$
  evaluation in the transcript is out of order, but its conclusion (one mistake,
  loss $1/6$) matches the clean recomputation used in the chapter.
- **"hypothesis space"** in §22.5(iii) is a standard gloss for the lecture's
  "collection of models" — not the lecture's own term; flagged, not removed, since
  it is the conventional name Alok will meet everywhere.
- **iid assumption**: Week 1 never states it formally ("independently generated" for
  the tweet stream is the closest). The chapter states the standing bet honestly and
  points to §20.1 for the formalism rather than inventing a Week-1 source.
