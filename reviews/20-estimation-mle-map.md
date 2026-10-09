# Review log — Chapter 20: Estimation: MLE and Bayesian/MAP

Branch: `draft/estimation-mle-map`. Reviewer: chapter author (self-review pass, 2026-10-09).
Sources re-read before writing: MLF transcript "Estimation of parameters using ML.docx.pdf"
(full read), MLF deck `week12-part2.pdf` (pdftotext + spot render), `Week 12 tutorials_sol.pdf`,
MLT deck `MLE_Bayesian.pdf` (15 pp, image-only — all pages rendered and read visually),
MLT TA notes `Estimation.pdf` (full read), MLF transcript `week12-part2` outline pages.

## What was checked

1. **Every derivation re-derived by hand**: Bernoulli compact form and
   $dR/d\theta$ ($-a/\theta + (n-a)/(1-\theta) = 0 \Rightarrow \hat\theta = a/n$);
   Uniform $R(a,b) = n\log(b-a)$ feasible region; Normal $\mu$ ($\sigma^2=1$);
   joint $(\mu,\sigma^2)$ including the $\frac{n}{2}\log\sigma^2 \to \frac{n}{2\sigma^2}$
   and $-\frac{\sum}{2\sigma^4}$ partials; Beta mode
   $\frac{\alpha-1}{p} - \frac{\beta-1}{1-p} = 0 \Rightarrow p = \frac{\alpha-1}{\alpha+\beta-2}$;
   Beta–Bernoulli posterior exponents; through-origin regression
   $\hat w = \sum x_iy_i/\sum x_i^2$.
2. **All numerics recomputed independently** (Python cross-check, all match):
   $L(0.75) = 0.10546875 > L(0.5) = 0.0625$, $L(0.9) = 0.0729$ (eg 2);
   $R(6) = 3.8326 < R(5) = 4.6326$ (eg 4);
   2-D $\hat\Sigma_{ML} = \frac13\begin{pmatrix}2&1\\1&2\end{pmatrix}$, det $= 1/3$ (eg 5);
   MAP $= 2/3$, posterior mean $= 9/14$, MLE $= 0.7$ (eg 6);
   Problem 5 likelihoods $1/256$ vs $(1/2.2)^4 \approx 0.0427$;
   Problem 8 MAP $= 7/12$, post. mean $= 15/26$; Problem 10 $\hat w = 25/14$.
3. **Every § cross-reference verified against the actual files**: §14.11 (Bayes' theorem ✓),
   §10.8 (first-order necessary condition ✓ — was mislabeled §10.5, fixed),
   §7.9 (positive definite ✓), §19.1/§19.9 (mean/covariance, promise ✓),
   §5.8 (least squares ✓), §9 (vector gradients — text corrected to say matrix
   gradients go *beyond* it). Bare §23/§25/§28/§30/§31 are forward refs per
   OUTLINE.md, same convention as Ch 19.
4. **Figure regenerated and inspected**: likelihood peak at $3/4$ ✓; posterior
   peak at $2/3$ ✓; marked original-matplotlib in the HTML comment.

## What was fixed during review

- §10.5 → §10.8 (10.5 is the 1-D gradient-descent algorithm, not the optimality condition).
- Beta sketch labels: first draft said Beta(2,7)/Beta(5,2); re-rendered deck slide 13
  at 130 dpi shows Beta(2,5) [orange, leans left], Beta(2,2) [blue, symmetric],
  Beta(0.5,0.5) [green, U-shaped], flat red line = uniform/Beta(1,1). Corrected.
- Removed "§3.2"/"§3.3" citations to the MLT TA notes (could be mistaken for book
  sections); now cited as "(the MLT TA notes)".
- Rephrased "trace derivatives from §9's toolkit" — Ch 9 covers vector gradients,
  not matrix calculus; the multivariate-normal ML derivation is stated as the deck's
  extension without proof.

## Thin / contradictory source points

1. **"Flat prior ⇒ MAP = MLE" is derived, not quoted.** No source states it verbatim;
   it follows from the deck/notes definitions (MAP = posterior mode; deck draws the
   flat Beta(1,1) prior). It is presented as a consequence and *proved* for Bernoulli
   in Problem 9. Flag: fine as stated, but the ch-28 author should not cite Ch 20 as
   a source for the general claim.
2. **Multivariate-normal ML formulas.** The deck only says "can be extended to
   multivariate normal" (OCR garbled on the formula slide). Formulas
   $\hat{\boldsymbol{\mu}} = \bar{\mathbf{x}}$,
   $\hat{\boldsymbol{\Sigma}} = \frac1n\sum(\mathbf{x}_i-\hat{\boldsymbol{\mu}})
   (\mathbf{x}_i-\hat{\boldsymbol{\mu}})^T$ are the standard result
   (cross-checked against mml-book §6.5); stated as the deck's extension, derivation
   explicitly deferred as needing matrix calculus.
3. **No consistency/asymptotic-normality claims in sources** — omitted entirely,
   per the chapter brief.
4. **§20.12(ii) "ridge = MAP with Gaussian prior" is a forward pointer for Ch 28**,
   a standard fact but not in the Ch-20 sources. The Ch-28 author must verify it
   against the MLT W6 sources when writing that chapter.
5. **MLT `MLE_Bayesian.pdf` is image-only (no text layer).** Content was recovered by
   reading all 15 rendered pages; the TA `Estimation.pdf` notes mirror the deck
   closely, so coverage is believed complete, but a full-text OCR was not possible.
6. **Beta mode edge-case table** (mode 0/1/any/{0,1}) reproduced from the TA notes
   as stated; not independently re-derived for the boundary cases.
