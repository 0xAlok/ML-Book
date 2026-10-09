# Review log — Chapter 32: Hard-margin SVM

*Written 2026-10-09. Self-review pass done before commit: every derivation re-checked, every number recomputed by hand and in numpy, every § cross-reference verified against the actual files on disk.*

## Sources read (first, before writing)

1. **MLT Week 10 lecture slides** (`sources/course materials/MLT-20261009T001607Z-1-001/MLT/PPT/Week 10/week_10.pdf`, 10 pages). Image-only PDF — rendered each page with `pdftoppm -png -r 80` and read the formulas off the images. Contents: perceptron/margin motivation ("quality of final solution"; "given that we prefer classifiers with large margin, can we directly find them?"), the scale issue ($\lVert w\rVert^2 = 1$ fix), canonical form $\max_w \mathrm{width}(w)$ s.t. $(\tilde w^T\tilde x_i)y_i \ge 1$, width via a projection problem, primal $\min \frac12\lVert w\rVert^2$ s.t. $(\tilde w^T\tilde x_i)y_i \ge 1$, the full dual derivation (Lagrangian → min-max swap for convex $f,g$ → stationarity $w_\alpha^\star = \sum \alpha_i x_i y_i$ → dual $\max_{\alpha\ge0} [\alpha^T\mathbf 1 - \frac12\alpha^T Y X^T X Y\alpha]$), complementary slackness $\alpha_i^\star(1-(w^{\star T}x_i)y_i)=0$, support vectors ("only the points on the supporting hyperplane can contribute to $w^\star$"; "$w^\star$ is a sparse linear combination of the data points"; "dual depends on $x_i^T x_j$ so can be kernelized"), the kernelized decision function, and the closing outlier question → soft-margin primal (left for Ch 33).
2. **MLT Week 10 practice assignment** (`.../MLT/Practice Assignment/Week_10.pdf`, Q1–Q8 hard-margin part). Text-extractable except the equations/figures (dropped in extraction) — used only its textual statements: Q1 (every SV lies on a supporting hyperplane; the converse is *not* guaranteed by complementary slackness), Q3 (rescaling the boundary *equation* is fine; rescaling $w$ alone breaks the primal solution), Q4 (width $= 2/\lVert w\rVert$), Q5–Q6 (prediction $= \operatorname{sign}$ of the linear score; points inside the margin still get labelled by the boundary), Q7 (SVM is discriminative, like the perceptron's presentation), Q8 (similarities/differences perceptron vs LR vs SVM: all linear, all discriminative, boundary $w^Tx+b=0$; LR has explicit distance-calibrated probabilities; SVM is the principled max-margin choice that "will generalize better"; soft margin robust to outliers, perceptron converges only when separable, LR more forgiving).
3. Cross-references: `chapters/31-perceptron-logistic.md` §31.5 (margin $\gamma$, $(R/\gamma)^2$ bound), §31.15(i) (Ch 32–33 framing); `chapters/29-knn-trees.md` §29.8(iii) ("heavy machinery" line); `chapters/11-*.md` §§11.3, 11.7; `chapters/13-*.md` §§13.2–13.4, 13.7–13.11, 13.15 (which already states the SVM primal in augmented form and promises the dual/kernel story); `STYLE.md`; `OUTLINE.md` (Ch 33 = soft-margin SVM + kernel trick, the single forward pointer).

## Numbers recomputed (hand + numpy, all agree)

- Worked example §32.6 ($x^{(1)}=(2,0),+1$; $x^{(2)}=(1,1),+1$; $x^{(3)}=(-1,0),-1$): dual $2\alpha-\tfrac52\alpha^2$ maximized at $\alpha^\star=(0,0.4,0.4)$ (hand derivative); $w^\star=(0.8,0.4)$, $b^\star=-0.2$; margins $(1.4,1.0,1.0)$; CS products $(0,0,0)$; primal $\frac12\lVert w\rVert^2=0.4=$ dual $0.8-0.4=0.4$ (strong duality); width $2/\sqrt{0.8}=\sqrt5\approx2.236$; supporting lines $2x_1+x_2=3$ / $=-2$; boundary $2x_1+x_2=0.5$. Numpy: dual feasibility $\sum\alpha_i y_i=0$ ✓, and a $401\times401$ grid search over the dual feasible set ($\alpha_1+\alpha_2=\alpha_3$) confirmed max $0.4$ at $(0,0.4,0.4)$ — a QP-free check.
- Problem 3 (symmetric square): dual $4\alpha-8\alpha^2 \to \alpha^\star=1/4$; $w^\star=(1,0)$, $b^\star=0$; width $2$; supporting lines $x_1=\pm1$; all four points active → all four are SVs; primal $0.5=$ dual $0.5$ ✓ (hand; Gram cross-terms verified: $\sum_{i,j}y_iy_jx_i^Tx_j=16$).
- Problem 1(iii): distance from $(2,0)$ to $2x_1+x_2=0.5$ is $3.5/\sqrt5\approx1.565 > \sqrt5/2\approx1.118$; also equals geometric margin $1.4/\lVert w^\star\rVert=0.7\sqrt5$ ✓ (two routes, same number).
- Problem 5(i): SV sum at $(0,0)$ gives $-0.2 \to \hat y=-1$, matching §32.6's direct score.

## Derivations double-checked

- Functional/geometric margin, scale invariance, canonical form $\min_i y^{(i)}(w^Tx^{(i)}+b)=1$ — standard; rescaling argument verified.
- Width $=2/\lVert w\rVert$ via distance between parallel planes $|c_1-c_2|/\lVert w\rVert$.
- Primal from max-width: max $2/\lVert w\rVert \iff$ min $\lVert w\rVert \iff$ min $\frac12\lVert w\rVert^2$ (monotone; $\tfrac12$ for clean gradients).
- Dual: Lagrangian → min-max (§13.3) → swap (§13.4/13.7) → stationarity $w^\star=\sum\alpha_i y^{(i)}x^{(i)}$, $\sum\alpha_i y^{(i)}=0$ → substitution algebra re-done by hand; dual objective matches the lecture's $\alpha^T\mathbf1-\tfrac12\alpha^T YX^TXY\alpha$ form.
- KKT → support vectors; the one-way implication and its failing converse reproduced from the assignment's Q1 argument.
- $b^\star=y^{(j)}-w^{\star T}x^{(j)}$ from any SV; decision function; inner-products-only observation.

## Cross-references verified against the actual files

- §31.1, §31.2–§31.5, §31.15 (and §31.15(i) naming Ch 32–33) — all exist in `chapters/31-perceptron-logistic.md`.
- §29.8(iii) in `chapters/29-knn-trees.md` — confirmed text: "Outperformed on hard tasks — on significantly difficult problems it can lose to SVMs and neural networks".
- §§11.3, 11.7 in `chapters/11-constrained-optimization-lagrange.md`; §§13.2, 13.3, 13.4, 13.7, 13.8, 13.9, 13.10, 13.11, 13.15 in `chapters/13-duality-kkt.md` — all exist with matching titles/content.
- Ch 33 (soft-margin SVM + kernel trick) exists in `OUTLINE.md` — the single forward pointer in §32.8 is safe.
- No external links used; figure is original matplotlib (`chapters/assets/32-margin-example.png`), marked with the HTML originality comment as in Ch 31. No soft-margin/kernel content covered beyond the one pointer.

## Thin / contradictory source points (flagged, not invented)

1. **Lecture's width formula is off.** Slide 2's solution box writes $\mathrm{width}(w)=2/\lVert w\rVert^2$. That is the optimal *objective value* of the projection subproblem ($\min_z \tfrac12\lVert z-x\rVert^2 = 2/\lVert w\rVert^2$), not the geometric distance, which is $\sqrt{2\cdot(2/\lVert w\rVert^2)}=2/\lVert w\rVert$. The practice assignment's Q4 ("the width is given by" $2/\lVert w\rVert$) agrees with the corrected form. The chapter derives $2/\lVert w\rVert$ from scratch and uses it everywhere; the primal $\min\frac12\lVert w\rVert^2$ (which the lecture states correctly) is unaffected either way since both forms are monotone in $\lVert w\rVert$.
2. **Bias folded into $w$ in the lecture.** The slides use augmented $\tilde x$ (no separate $b$), so their dual has only $\alpha\ge0$. The chapter keeps $b$ explicit per the task's canonical form, which adds the stationarity condition $\sum_i\alpha_i y^{(i)}=0$ — a standard step, marked with a `Note:` in §32.4. (Ch 13's §13.15 preview also uses the augmented form; consistent, just notational.)
3. **Practice-assignment equations were dropped by text extraction.** Only its prose statements (Q1 subtlety, Q3 scaling warning, Q6/Q7/Q8 lists) were used; nothing was reconstructed from the missing formulas.
4. **Strengths/limits kept to sourced claims.** "Generalizes better than perceptrons", sparsity, kernelizable dual, discriminative, separability requirement, outlier sensitivity, no probabilities — all from the slides or the assignment. No invented complexity claims.

## Fixes made during self-review

- Re-derived the §32.6 dual quadratic coefficient by hand ($2\alpha_2^2+\alpha_3^2+2\alpha_2\alpha_3$ with the $y_2y_3$ sign) after catching that the symmetric-guess $w=(1,0)$ was *not* optimal for this dataset — the true optimum $(0.8,0.4)$ was found via the dual and verified by KKT + numpy grid search.
- Figure legend/text positions adjusted after first render (mathtext `\tfrac` unsupported in this matplotlib → used `\frac{1}{2}`).
- §32.5 `Note` on in-margin test points added to mirror practice Q6 explicitly.
- Problem 2(i) caveat added: doubling preserves $\ge1$ only from a $\ge 1/2$ start; the wall-invariance claim is what matters.
