# Review log — Chapter 12. Convex sets and convex functions

## Sources used (all committed under `sources/course materials/MLF-20261009T001559Z-1-001/MLF/`)

1. `Transcripts/Week 8/Introduction to Convexity.pdf` — convex-set definition ($\lambda\mathbf{x}_1 + (1-\lambda)\mathbf{x}_2$), the $\lambda$-as-lever intuition, intervals in $\mathbb{R}$, hyperplane convexity proof, half-space as exercise.
2. `Transcripts/Week 8/Properties of Convex Sets.pdf` — intersection theorem + proof, $\{A\mathbf{x}=\mathbf{b}\}$ as intersection of hyperplanes, convex combinations, convex hull (both definitions) + the two exercises (CH is convex; the two hull definitions agree).
3. `Transcripts/Week 8/Convex Functions.pdf` — all four definitions (epigraph; chord inequality $f(\lambda x_1+(1-\lambda)x_2)\le\lambda f(x_1)+(1-\lambda)f(x_2)$; first-order $f(y)\ge f(x)+(y-x)^T\nabla f(x)$; Hessian PSD $\iff$ convex), $x^2$ via Def 4.
4. `Transcripts/Week 8/Properties of Convex Functions.pdf` — local-min-is-global theorem + full contradiction proof, GD implication.
5. `Transcripts/Week 9/Properties of Convex Functions - 1.pdf` — non-uniqueness of global minima + "set of global minimizers is convex" exercise, $\nabla f(\mathbf{x}^\star)=\mathbf{0}$ iff global minimum (proof sketch), Property 1 (sums), Property 2 (convex non-decreasing $\circ$ convex), Property 3 (convex $\circ$ linear), the $e^{-x^2}$ composition counterexample, concave = $-f$ convex.
6. `Transcripts/Week 9/Applications of Optimization in Machine Learning.pdf` — least squares $f(\mathbf{w})=\sum_i(\mathbf{w}^T\mathbf{x}_i-y_i)^2$ proved convex via sum + composition.
7. `PPT/Week 8/Week 8 Tutorial.pdf` — intersection/convex-hull/epigraph summary slides; worked Examples 3 ($4x^2+2y^2$ convex, Hessian $\mathrm{diag}(8,4)$) and 4 ($x^2-y^2$ non-convex, Hessian $\mathrm{diag}(2,-2)$); $2\times2$ determinant test ($a>0$, $\det>0$).

## Cross-check anchors (correctness only, nothing copied)

- mml-book ch. 7 (via web search): chord inequality direction, "chord lies above the graph", local-min-is-global theorem — all consistent with the chapter.
- Book continuity: Ch 7 §7.9 (eigenvalue test for PD/PSD), Ch 10 §10.8 (Hessian interrogation), §10.9 (local vs global minima promise), Ch 11 §11.8 (projected GD needs convex sets), §11.2 (half-space picture).

## What I double-checked by hand (second pass)

- eg 1 (interval): $\lambda x_1+(1-\lambda)x_2 \ge \lambda a+(1-\lambda)a = a$ and $\le b$ — recomputed, correct.
- eg 2/3 (hyperplane/half-space): linearity of $\mathbf{w}^T(\cdot)$ distributes over the mixture; inequality step uses $\lambda,1-\lambda\ge0$ — correct.
- eg 4 (ball): triangle inequality then homogeneity then the bound $\lambda\theta+(1-\lambda)\theta=\theta$ — correct.
- eg 5 ($A\mathbf{x}=\mathbf{b}$): row-wise decomposition into $m$ hyperplanes, then intersection theorem — correct.
- eg 6 ($x^2$ chord proof): $\lambda x_1^2+(1-\lambda)x_2^2-(\lambda x_1+(1-\lambda)x_2)^2 = \lambda(1-\lambda)(x_1-x_2)^2 \ge 0$ — re-expanded term by term, correct.
- eg 7 ($x^3$ non-convex): $(-2,0,\lambda=1/2)$ gives $-1 \le -4$, false — correct. (Caught during review: an earlier draft of Problem 7 used $(0,2\pi,1/2)$ for $\sin x$, which gives $0\le0$ — no violation. Fixed to $(0,\pi,1/2)$: $1\le0$, false.)
- eg 8 (tangent check for $x^2$ at $x=1,y=-2$): $4 \ge 1+2(-3)=-5$ — correct.
- eg 9/10 (tutorial Hessians): $\mathrm{diag}(8,4)$ PD $\to$ convex; $\mathrm{diag}(2,-2)$ indefinite $\to$ non-convex — match the tutorial slides exactly.
- eg 11 (least squares): each $h_i$ is $z^2\circ(\text{affine})$, convex by Property 3; sum convex by Property 1 — correct.
- §12.9 proof: the chain $f(\mathbf{x}^\star)\le f(\mathbf{y}_\lambda)\le\lambda f(\mathbf{x}^\star)+(1-\lambda)f(\mathbf{z})<f(\mathbf{x}^\star)$ — strictness comes from $f(\mathbf{z})<f(\mathbf{x}^\star)$ with $1-\lambda>0$ — correct.
- Solutions 8 ($x^2+xy+y^2$): $\mathbf{H}=[[2,1],[1,2]]$, $a=2>0$, $\det=3>0$, eigenvalues $3,1$ — recomputed, convex, correct.
- Solutions 10: $\nabla^2\lVert X\mathbf{w}-\mathbf{y}\rVert^2 = 2X^TX$; $\mathbf{z}^T(2X^TX)\mathbf{z}=2\lVert X\mathbf{z}\rVert^2\ge0$ — correct.
- Solutions 12: $\nabla f=(2(x-3),2(y+1))$, zero at $(3,-1)$; tangent bound collapses to $(y_1-3)^2+(y_2+1)^2\ge0$ — correct.

## Discrepancies / judgment calls

1. **First-order definition scope.** The Week 8 *tutorial slide* says the tangent lower bound holds "for a point $y$ in the neighbourhood of $x$"; the *lecture transcript* (Convex Functions, 14:16–21:37) is explicit that the linear approximation lower-bounds $f$ "no matter how far you go away" — the entire space. The transcript is the correct/global statement (and matches standard theory); the chapter states "for all $\mathbf{x},\mathbf{y}$" per the transcript. Flagged here rather than silently following the slide.
2. **Least-squares proof route.** The Applications transcript invokes "property 2" (convex non-decreasing $\circ$ convex) for $z^2\circ(\mathbf{w}^T\mathbf{x}_i-y_i)$, but $z^2$ is not non-decreasing on all of $\mathbb{R}$ — the clean fit is Property 3 (convex $\circ$ affine/linear), whose proof goes through unchanged for affine $g$ ($g(\lambda\mathbf{x}+(1-\lambda)\mathbf{y})=\lambda g(\mathbf{x})+(1-\lambda)g(\mathbf{y})$ still holds with the constant term). The chapter uses Property 3; mathematically equivalent conclusion, more rigorous step.
3. **Jensen's inequality: OMITTED.** Searched all MLF and MLT committed sources for "jensen" — zero hits. Per instructions, not invented; the chapter ends §12.13 with a forward pointer noting Jensen's inequality appears in Ch 25 (EM)/MLT and that the chord inequality is its ancestor. No statement of Jensen is given since no source covers it.
4. **Logistic regression / SVM convexity: forward pointers only.** The task names them (Ch 31, Ch 32–33); those chapters don't exist yet and no source in scope proves them. §12.12 lists them as pointers without proofs — no invented content.
5. **"Neural nets are NOT convex":** one-liner as instructed by the task; standard, non-proved claim, kept to one line.
6. **Diagram:** no standard-resource figure was reused (task forbids fabricating source URLs). Drew one original 2×2 matplotlib figure (`chapters/assets/ch12-convex-sets-functions.png`): convex vs non-convex set, convex vs non-convex function with epigraph shading. Noted as original in the chapter via HTML comment.
7. **§12.5 convex hull:** kept brief (definition + both characterizations). The equivalence proof is a solutions-only follow-up (Solution 5), since the lecture leaves it as an exercise.
8. **Affine functions both convex and concave:** stated in §12.8; not explicit in the MLF transcripts but confirmed by the mml-book cross-check and immediate from Def 2 (equality holds). Low-risk, standard.

## Deliberate omissions

- Strict/strong convexity: not in the MLF sources at this level — omitted.
- Subgradients (for $|x|$ at the kink): outside source scope — the chapter only notes Def 3/4 don't apply there (Problem 6).
- Projected-GD convergence proof: belongs to Ch 11's promise; Ch 12 only supplies the "convex set" definition it needed — pointer, not proof.
- Duality/KKT details: Chapter 13's material — only forward pointers in §12.13.
