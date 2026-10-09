# Review log — Chapter 30: Naive Bayes: generative vs discriminative models

## Sources read (before writing)

- `MLT-20261009T001607Z-1-001/MLT/Slides - Ashish Tendulkar/Week 6/MLT Week 6 Slides - Naive Bayes.pdf` (44 pp). Full text via `pdftotext` (extractable, not image-only). Sourced verbatim: "Generative counterpart of logistic regression"; "Makes strong (naive) conditional independence assumption"; "Simple yet very powerful classifier … document classification and spam filtering"; $p(x_1,\ldots,x_m\mid y)=\prod_j p(x_j\mid y)$; posterior with evidence expanded as $\sum_r p(x\mid y_r)p(y_r)$; parameters = $k$ priors + $k\times m$ class-conditional densities; the four feature-type distributions (Bernoulli compact form $\mu^{x_j}(1-\mu)^{1-x_j}$ + verification, categorical indicator form, multinomial, Gaussian, diagonal-covariance multivariate Gaussian); inference $\hat y=\arg\max_{y_c}(\sum_j\log p(x_j\mid y_c;w))+\log p(y_c;w)$ + underflow rationale + "does not return the probability" warning; likelihood $L(w)=\prod_i p(x^{(i)},y^{(i)};w)$, NLL $J(w)=-\ell(w)$ "to maintain uniformity"; 3-step MLE recipe; prior = class fraction; Bernoulli/categorical/multinomial/Gaussian MLE formulas; zero-count → $w=0$ → posterior $0$; Laplace smoothing $+c$/$+2c$, $+ce$, $+cm$, $c=1$ = Laplace, "too high $c$ leads to underfitting", $c$ a hyperparameter; evaluation = confusion matrix, precision/recall/F1, AUC ROC/PR. NB schematic credits https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html — NOT reused (chapter's own figure drawn instead).
- `MLT-20261009T001607Z-1-001/MLT/Live session slide - 2024 Sep/TA notes/Week 8/Week 8 Summary.pdf` — text-extractable. Sourced verbatim: generative models $P(x,y)$ vs discriminative models $P(y\mid x)$; examples {Naive Bayes, Gaussian NB} vs {decision trees, KNN}; "We just need $P(y\mid x)$ for prediction"; both factorizations $P(y\mid x)P(x)$ / $P(x\mid y)P(y)$; full generative story needs $2^{d+1}-1$ parameters ("computationally infeasible") vs NB $2d+1$; independence assumption "does not model the grammatical, semantic and other structures present in a language"; MLE = class fractions; prediction inequality $P(x_{\text{test}},1)>P(x_{\text{test}},0)$; Laplace smoothing as adding pseudo-emails; linear decision boundary $w^T x_{\text{test}}+b>0$ with $w_j,b$ derivation; the $d=4$ all-$1/2$ worked prediction.
- `MLT-20261009T001607Z-1-001/MLT/Notes/Using Naive Bayes to Estimate Parameters.png` and `Using Naive Bayes to Predict Label for New Datapoints.png` ("Get Your Concepts Right", IITM BS MLT SEP 2022 Week 8). Read as images. Complete 6-point, 3-binary-feature worked example — the chapter's §30.9 dataset. **Arithmetic slip found in the second PNG**: it prints $P(X_{\text{test}}\mid y{=}1)=1.00\times0.33\times0.67=0.22$, using $(1-\hat p_3^1)=0.67$ for the third feature whose value is $1$; correct is $\hat p_3^1=0.33$, giving $1/9\approx0.11$. Verdict (predict $1$) unchanged since $1/9>2/27$. Chapter uses the corrected value; flagged here, not propagated.
- `MLT-20261009T001607Z-1-001/MLT/Practice Assignment/Week_8.pdf` — text extraction loses math glyphs, but Q6 (spam: $P(\text{spam})=0.2$, four word-likelihoods, "Hurray! win exciting prizes") is fully legible and became Problem 4; Q1/Q2 (parameter counting with/without conditional independence) shaped Problem 2; Q7 (Gaussian NB boundary hint: solve $p(x\mid1)p(1)=p(x\mid0)p(0)$) shaped Problem 5.
- `MLT/.../Transcripts/` — empty; not blocking (slides + TA notes self-contained).
- `chapters/14-probability-basics.md` §14.11 (Bayes' theorem, four-term naming, "base rates dominate" → spam filters pointer); `chapters/20-estimation-mle-map.md` §20.4 (zero-count warning), §20.11 (Beta-prior smoothing cousin), §20.12(iv) (Bayesian story continues in Ch 30); `chapters/22-what-is-ml.md` §22.9 (classification setup/loss); `chapters/15-discrete-random-variables.md` §15.11 (Bernoulli).

## Numbers recomputed (twice each: exact fractions by hand + numpy)

- §30.9 worked example: priors $1/2,1/2$; conditionals class 1: $(1,2/3,1/3)$, class 0: $(1/3,2/3,2/3)$ — hand-counted from the table, numpy agrees. $X_{\text{test}}=[1,0,1]$: $P(X\mid1)=1\cdot\frac13\cdot\frac13=\frac19\approx0.111$; $P(X\mid0)=\frac13\cdot\frac13\cdot\frac23=\frac{2}{27}\approx0.074$; scores $\frac1{18}\approx0.0556$, $\frac1{27}\approx0.0370$; posteriors $3/5=0.6$, $2/5=0.4$; predict $1$. (Script `/tmp/verify30.py` — asserts exact `Fraction` equality plus numpy cross-check; numpy float gives $0.11111111111111112$ vs $1/9$ — pure float representation, fractions exact.)
- Smoothed ($c=1$): $(0.8,0.6,0.4)$ / $(0.4,0.6,0.6)$; likelihoods $0.128$, $0.096$ — numpy agrees; predict $1$.
- §30.10 Gaussian: $\hat\mu_0=2,\hat\mu_1=5,\hat\sigma^2=2/3$; boundary $(x-5)^2=(x-2)^2\Rightarrow x=3.5$; $x=4\to1$, $x=3\to0$ — density comparison at $x=3,4$ checked numerically.
- Problem 1 ($[0,0,1]$): likelihoods $0$, $4/27$; posteriors $0$, $1$; predict $0$ — fractions exact.
- Problem 2: $4\cdot2^5-1=127$; NB $4+20=24$ (notes-style), $23$ free.
- Problem 3 ($[0,1,1]$): unsmoothed $0$, $8/27$; smoothed $6/125=0.048$, $27/125=0.216$; posteriors $0.1818$, $0.8182$ — fractions exact.
- Problem 4 (spam): numerator $8.4\times10^{-5}$, denominator $8.416\times10^{-5}$, ratio $0.99809886\approx\boxed{1.00}$ — numpy agrees.
- Problem 5: $\hat\mu_0=2,\hat\mu_1=6,\hat\sigma^2=2/3$; boundary $x=4$ (algebra: $-8x=-32$); $x=3\to0$, $x=5\to1$ — numeric density check agrees.
- §30.5 log-odds weights: $w_j=\log\frac{p_{j1}(1-p_{j0})}{p_{j0}(1-p_{j1})}$ derived by hand from the TA notes' expansion; verified algebra (coefficient of $f_j$). With the chapter's unsmoothed estimates $w_1=\infty$ (since $\hat p_1^1=1$) — consistent with the zero-count pathology, not a formula error.

## Derivations double-checked

- Posterior with NB assumption (§30.3): numerator/denominator substitution from the slides, matches slides p.10 exactly.
- Bernoulli compact form verification (§30.4(i)): $x_j\in\{0,1\}$ cases checked.
- Evidence cancellation in argmax (§30.2, §30.5): $P(x)$ independent of $y_c$; log monotonic — Problem 7's proof.
- Bernoulli MLE (§30.7(ii)): $\frac{\partial}{\partial w}[\,x\log w+(1-x)\log(1-w)\,]=\frac{x}{w}-\frac{1-x}{1-w}=0\Rightarrow w=x$ per observation, averaging over class-$y_r$ examples gives the stated fraction — matches the slides' 3-step result.
- Multinomial coefficient written as $l!/x_1!\cdots x_m!$ (see flag below re: slide rendering).

## Cross-references verified

- Script-checked every `§X.Y` in `chapters/30-naive-bayes.md` and `solutions/30-naive-bayes.md` against actual `## X.Y` headers: 14.11, 15.11, 20.4, 20.10, 20.11, 20.12(iv), 22.9, 22.10, 25.12(iv), 29.2, 29.9, 30.x, 31, 32–35 — all resolve. (Ch 31 exists in OUTLINE as logistic regression; ch 29 §§29.2/29.9 headers confirmed.)
- Figures `assets/30-nb-schematic.png`, `assets/30-gaussian-nb-boundary.png` exist in `chapters/assets/`, linked relative to `chapters/` (ch22/26/28/29 convention), each marked original in an HTML comment; both viewed as rendered PNGs — legible, numbers match the chapter's fractions (1/18 vs 1/27, boundary 3.5). No external image URLs used, so no curl checks needed.

## Thin / contradictory source points (flagged, not invented)

1. **Source arithmetic slip (corrected, not copied):** the "Get Your Concepts Right" prediction PNG computes $P(X_{\text{test}}\mid y{=}1)$ as $1.00\times0.33\times\mathbf{0.67}=0.22$; the third factor must be $\hat p_3^1=0.33$ ($F_3=1$), giving $1/9\approx0.11$. Final verdict unaffected. Chapter §30.9 uses $1/9$ and carries an explicit Note.
2. **Multinomial coefficient rendering:** the slide's formula extracts ambiguously (`l!` vs `n!` in the glyphs) for $p(x\mid y_c;l,\mu)$ with $\sum_j x_j=l$; used $l!/(x_1!\cdots x_m!)$, which is the only form consistent with the slide's own constraint $\sum_j x_j=l$.
3. **Gaussian parametrization looseness:** slides write $\mathcal{N}(\mu_{jc},\sigma_{jc})$ then a density in $\sigma_{jc}^2$; chapter uses the standard $\mathcal{N}(\mu_{jc},\sigma_{jc}^2)$ throughout.
4. **"Why it works anyway" (argmax-forgiveness):** the slides only say "simple yet very powerful"; the one-line explanation in §30.3(iii) is the standard textbook rationale, marked as such, not attributed to the lectures.
5. **Categorical/multinomial parameter counts:** slides state $k\times\sum_j\lvert v_j\rvert$ and $k\times m$ (raw probabilities, no $-1$ for the sum-to-one constraint); presented as the slides give them, without the free-parameter refinement (that refinement *is* exercised in Problem 2 for priors, where the TA notes' convention makes it unambiguous).

## Fixes made during self-review

- Recomputed the §30.9 likelihood from fractions instead of the PNG's rounded $0.33/0.67$ (which is where the source's slip came from).
- Fixed matplotlib `\tfrac`/`\frac12` mathtext errors in the figure script; moved an overlapping bar label inside its bar.
- Problem 5's classes re-spaced ($\{5,6,7\}$ not $\{4,5,6\}$) so the boundary $x=4$ doesn't coincide with a training point.
- Verified no `§` reference points at a nonexistent section (script check above).
