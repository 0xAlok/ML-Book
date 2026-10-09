# Review log — Chapter 26: K-means clustering

## Sources read (before writing)
- `sources/course materials/MLT-20261009T001607Z-1-001/MLT/Notes/Week3.pdf` — the k-means lecture (12 pp). All formulas are images in the PDF; extracted by rendering pp. 2–4, 6, 7, 9–11 at 120–150 dpi and reading them: objective $F(z_1,\ldots,z_n)=\sum_i\|x_i-\mu_{z_i}\|^2$, centroid formula, Lloyd's init/mean/reassign steps, the four-line convergence argument, half-space/bisector geometry, k-means++ score $S(x)=\min_{j<l}\|x-\mu_j^0\|^2$, the $K=n\Rightarrow F=0$ + penalty($K$) discussion, and the $K^n$ count.
- `MLT/Practice Assignment/Week_3.pdf` — all 10 questions + solutions (step order, monotonicity $F^{t+1}\le F^t$, perpendicular bisector, init-dependence incl. the two-lines example, k-means++ probabilities, partition non-repetition, $K=1$ max / $K=n$ zero, local-vs-global minima, outlier sensitivity).
- `MLT/PPT/revision/Week3.pdf` — same content as the notes (no new material).
- `MLT/Live session slide - 2024 Sep/Week-3/Week 3.pdf` — handwritten images, no extractable k-means content; the TA notebook `W3_Sol.ipynb` confirms the standard Lloyd loop on Gaussian blobs (no new claims used).
- Transcripts folder is empty — no transcripts exist for MLT Week 3. Noted, not blocking.
- `chapters/25-gmm-em.md` (esp. §25.12(i) promise, §§25.8–25.9 for the hard/soft bridge), `chapters/24-pca.md` (notation), `chapters/22-what-is-ml.md` (§22.11), `chapters/20-estimation-mle-map.md` (§20.6), `chapters/12-convex-sets-functions.md` (§§12.3–12.4).

## Numbers recomputed (twice each: exact fractions + numpy floats)
- §26.5 worked example: $\mu_1^0=(8/3,3)$, $\mu_2^0=(5,14/3)$; all 12 squared distances ($\tfrac{61}{9},\tfrac{34}{9},\tfrac{40}{9},\tfrac{181}{9},\tfrac{244}{9},\tfrac{250}{9}$ vs $\tfrac{265}{9},\tfrac{208}{9},\tfrac{202}{9},\tfrac{25}{9},\tfrac{58}{9},\tfrac{52}{9}$); $F^0=\tfrac{196}{3}\approx65.33$; $z^1=(1,1,1,2,2,2)$; $\mu_1^1=(4/3,4/3)$, $\mu_2^1=(19/3,19/3)$; $F^1=\tfrac83\approx2.67$; convergence check ($z^2=z^1$; eg $x_3$: $\tfrac59$ vs $\tfrac{425}{9}$). Fractions and numpy agree to all digits.
- §26.6 eg 3 bisector: $10(x_1+x_2)=\tfrac{230}{3}\Rightarrow x_1+x_2=\tfrac{23}{3}$; midpoint $(\tfrac{23}{6},\tfrac{23}{6})$ satisfies it; all six membership claims ($2,3,3<7.67$; $12,13,13>7.67$) verified.
- §26.7 eg 4 k-means++ scores: $50,41,41,0,1,1$, total $134$, probabilities match.
- Problem set: P1(i) $F=1$, P1(ii) $F=\tfrac{146}{3}$; P2 $F^0=38$, $F^1=1$; P4(i) $x_1=2$; P5 total $61$, probs $0,25/61,36/61$ — all verified by independent script.

## Derivations double-checked
- Mean minimizes sum of squared deviations (componentwise §20.6 argument) — used in §26.3(i) and Problem 3.
- Monotonicity $F^{t+1}\le F^t$ from the two steps; strictness iff a point moves — matches lecture slide + practice Q2/Q6 solutions.
- Perpendicular bisector: general proof in Problem 4 solution ($2x^T(\mu_2-\mu_1)=\|\mu_2\|^2-\|\mu_1\|^2$, normal $\parallel\mu_2-\mu_1$, midpoint satisfies).
- EM M-step with 0/1 responsibilities $\to$ k-means centroid (Problem 8) — algebra checked.

## Cross-references verified
- Script-checked every `§X.Y` in the chapter against actual `## X.Y` headers in `chapters/*.md`: all resolve (§§12.3–12.4, §20.6, §§22.11–22.13, §§25.3/25.4/25.8/25.9/25.10/25.12, §§26.x). §25.12(i)'s promise ("The hard-assignment cousin, k-means, is Chapter 26's job") is fulfilled by §26.9.
- Figure path `assets/26-k-means-lloyd-iteration.png` resolves relative to `chapters/` (matches ch25 convention); file exists.

## Thin / contradictory / beyond-lecture points (flagged, not invented)
1. **Elbow method / concrete $K$-selection rules** — not in the sources; §26.8 covers only the lecture's $F+\text{penalty}(K)$ slogan and says so explicitly.
2. **k-means as the $\sigma^2\to0$ limit of a spherical GMM** — standard textbook claim, NOT in these lectures; omitted from the chapter, noted in §26.9's Note.
3. **Empty clusters** — the lecture's $K^n$ count allows them but the centroid formula divides by headcount; no fix prescribed. Chapter notes the gap honestly (§26.4 Note).
4. **Tie-breaking in $\arg\min$** — unspecified in the lecture; chapter states "keep the old label" as an explicit convention, not a source claim.
5. **"Voronoi cells"** — not used; chapter sticks to the lecture's "half-spaces" language.
6. Practice Q2's option texts were lost in PDF extraction (only the answer (b) + solution survived); the chapter states the monotonicity claim from the solution text, not the option lettering.
7. No contradictions found between the notes, revision slides, and practice assignment.

## Fixes made during self-review
- Intro scope line corrected: §§26.1–26.9 → §§26.1–26.10 (§26.10 is also sourced).
- §26.1(ii): clarified the tweets example is §22.11's, not the k-means lecture's.
- §26.2(ii): tightened "every step moving to a strictly better one" → "each step either moves to a strictly better one or stops".
- §26.3 eg 2: clarified the Problem 1 cross-mention uses different data (avoid implying the numbers belong to eg 2's dataset).
- Style sweep: no emojis, all math in LaTeX, "Basically, ..." under every complex topic (§§26.1–26.10), handwritten-notebook voice per STYLE.md.

## Figure
- `chapters/assets/26-k-means-lloyd-iteration.png` — original matplotlib (two panels: initial vs post-iteration assignment, centroid stars, perpendicular bisector dashed), marked original in the HTML comment. No external images used.
