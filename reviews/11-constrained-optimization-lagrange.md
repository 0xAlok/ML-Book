# Review log — Chapter 11: Constrained optimization

## Sources used

Primary (all committed under `sources/course materials/MLF-20261009T001559Z-1-001/MLF/`):

1. `Transcripts/Week 8/Constrained Optimization (Part -1).pdf` — the optimality-check story: "no descent direction is feasible", descent direction $\mathbf{d}^T\nabla f < 0$, feasible directions as descent directions of $g$, anti-parallel configuration for the inequality case.
2. `Transcripts/Week 8/Constrained Optimization (Part -2).pdf` — anti-parallel vs parallel configurations; inequality: $\nabla f = -\lambda\nabla g$, $\lambda > 0$; equality: same with $\lambda$ unrestricted; "Lagrange multiplier" naming.
3. `Transcripts/Week 8/Method of Lagrange Multiplier  Projected Gradient Descent.pdf` — the worked example $f(x_1,x_2) = x_1^2 + 2x_2 + 4x_2^2$ on $x_1^2 + x_2^2 = 1$ (all four stationary points, values $6, 2, 2/3$, min/max/saddle classification); projected gradient descent definition $\Pi_S$ as closest-point map, the two-step update, the catches (projection must be efficiently computable; convergence needs a convex constraint set).
4. `PPT/Week 8/8.3.Method of Lagrange Multiplier, Projected Gradient Descent.pdf` — pages are scanned images; page 1 rendered via `pdftoppm` and read visually to confirm the sign convention $\nabla f(\mathbf{x}^\star) = -\lambda\nabla g(\mathbf{x}^\star)$ "[no sign constraint]" and the same worked example. Pages 2–4 not inspected (the transcript covers them).
5. `PPT/Week 8/8.1.Constrained Optimization (Part -1).pdf` — page 2 rendered and read visually; confirms the Part-1 slide content (matches transcript 1). Page 1 not inspected.
6. `PPT/Week 8/Week 8 Tutorial.pdf` — optimality-condition slides (confirming the $\lambda > 0$ anti-parallel statement for inequalities), the box-volume Example 1 (used as Problem 3), Example 2 (plane-cylinder — not used; see omissions). The tutorial's "Week 6" headers are a copy-paste artifact in the source, not a content issue.

Consistency reads: `chapters/10-unconstrained-optimization-gradient-descent.md` (§10.1 cow-and-grass reference, GD update, step-size discussion, tutorial caveats), `chapters/09-multivariable-calculus.md` (§9.4–9.5: gradient as steepest ascent, perpendicular to level sets — used for the §11.6 geometry bridge).

Cross-check anchor (correctness only, nothing copied): MML book (Deisenroth et al.), §7.2, via the free PDF at https://mml-book.github.io/book/mml-book.pdf — confirms the convention $L = f + \lambda g$ with $\nabla f = -\lambda\nabla g$ stationarity, i.e. the same stationarity condition as the lectures up to the sign of the unrestricted multiplier.

## Sign convention — what was matched and noted

The lectures state the condition as $\nabla f(\mathbf{x}^\star) = -\lambda\nabla g(\mathbf{x}^\star)$. The chapter defines $L(\mathbf{x},\lambda) = f(\mathbf{x}) - \lambda h(\mathbf{x})$, whose stationarity is $\nabla f = \lambda\nabla h$. Since $\lambda$ is unrestricted for equality constraints, the two forms are identical (rename $\lambda \to -\lambda$). This is stated explicitly in a `Note:` in §11.3 and exercised in Problem 1. For inequalities the chapter keeps the lecture's form ($\nabla f = -\lambda\nabla g$, $\lambda \ge 0$) with complementary slackness, and defers the full Lagrangian/KKT statement to Chapter 13.

## What was double-checked

- Recomputed the lecture example fully: gradients, the $2x_1(1+\lambda) = 0$ case split, $x_2 = -1/3$, $x_1 = \pm\sqrt{8}/3$, the four function values ($6$, $2$, $2/3$, $2/3$), min/max/saddle classification — all match the transcript.
- Recomputed every eg and problem-solution independently: the warm-up ($x = y = 1/2$, $f = 1/2$), the inactive/active inequality examples, the box problem ($(2,2,1)$, $V = 4$), the $x^2+xy+y^2$ problem (minimizers at $\mu = 1$, maximizers at $\mu = -1$), the two-constraint problem ($(1/3,1/3,1/3)$, $f = 1/3$), and all projected-GD iterates ($0 \to 1 \to 1.8 \to 2 \to 2$).
- Verified the illustration's geometry: at $(\sqrt{8}/3, -1/3)$, $\nabla f = \nabla g = (2\sqrt{8}/3, -2/3)$ — parallel, consistent with $\nabla f = \lambda\nabla g$, $\lambda = 1$ (chapter convention) / $\nabla f = -\lambda\nabla g$, $\lambda = -1$ (lecture convention). The level set of $f$ through the minimizer is tangent to the circle (visible in the rendered figure).
- Checked that no content was pulled from decks 8.4–8.7 (convexity) — convexity is only forward-pointed to Chapter 12, exactly as the lectures do ("we will define this formally next time").
- No emojis; all math in LaTeX; English only.

## Discrepancies / uncertainties

1. The exact cow-and-grass rope numbers from the lectures are not recoverable from the available sources (the 8.1/8.2 PPTs are scanned images; the transcripts formalize the problem as $\min f$ s.t. $g \le 0$ without the cow numbers). The chapter therefore uses Chapter 10's objective $(x_1-40)^2 + (x_2-40)^2$ with a generic rope ($x_1^2 + x_2^2 \le R^2$, peg at origin) — the *picture* is faithful; the rope length $R$ is a placeholder, clearly generic, not attributed to the lectures.
2. The tutorial PDF's Example 2 solution text is garbled in extraction (underlined fractions rendered as broken markup); it was not used anywhere in the chapter.
3. The lecture transcript for Part-2 states the inequality multiplier as "a positive scalar"; the chapter writes $\lambda \ge 0$ (allowing the boundary case $\lambda = 0$ for the inactive-but-touching case) — consistent with complementary slackness, and flagged as the light treatment pending Chapter 13.
4. `assets/ch11-lagrange-levelsets.png` is an original matplotlib figure drawn for this chapter (no external source URL exists for it); the HTML comment above its embed says so.

## Deliberate omissions

- Convexity theory (decks 8.4–8.7): Chapter 12's territory; only the lectures' one-line claims about projected GD are repeated.
- Full KKT conditions, duality, and the dual problem: Chapter 13; §11.7 says so explicitly.
- The tutorial's plane–cylinder Example 2: too heavy for this chapter's scope; a lighter two-constraint problem (Problem 10) covers "one multiplier per constraint" instead.
- Second-order (sufficiency) conditions for constrained optima: not in the sources at this level; the chapter instead classifies candidates by direct evaluation, as the lectures do.
