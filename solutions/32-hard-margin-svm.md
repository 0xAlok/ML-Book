# Solutions — Chapter 32: Hard-margin SVM

*Full worked solutions to the Chapter 32 problem set. Every number recomputed independently (hand algebra cross-checked in numpy; see the review log).*

## Problem 1 — Width and supporting lines

Given $w^\star = (0.8, 0.4)$, $b^\star = -0.2$.

**(i) Width.** $\lVert w^\star\rVert^2 = 0.64 + 0.16 = 0.8 = 4/5$, so $\lVert w^\star\rVert = 2/\sqrt5$. Width $= 2/\lVert w^\star\rVert = 2/(2/\sqrt5) = \boxed{\sqrt5 \approx 2.236}$.

**(ii) The three lines.** Supporting hyperplanes $w^{\star T}x + b^\star = \pm 1$:

- $0.8x_1 + 0.4x_2 - 0.2 = 1 \;\Rightarrow\; 0.8x_1 + 0.4x_2 = 1.2 \;\Rightarrow\; \boxed{2x_1 + x_2 = 3}$ (multiply by $2.5$),
- $0.8x_1 + 0.4x_2 - 0.2 = -1 \;\Rightarrow\; 0.8x_1 + 0.4x_2 = -0.8 \;\Rightarrow\; \boxed{2x_1 + x_2 = -2}$.

Boundary $w^{\star T}x + b^\star = 0$: $0.8x_1 + 0.4x_2 = 0.2 \Rightarrow \boxed{2x_1 + x_2 = 0.5}$.

**(iii) Distance of $x^{(1)} = (2,0)$ from the boundary.** Distance from $(x_1, x_2)$ to $2x_1 + x_2 - 0.5 = 0$ is $|2x_1 + x_2 - 0.5|/\sqrt{2^2 + 1^2}$. At $(2, 0)$: $|4 - 0.5|/\sqrt5 = 3.5/\sqrt5 \approx \boxed{1.565}$. Half-width $= \sqrt5/2 \approx 1.118$. Since $1.565 > 1.118$, the point lies strictly beyond the $+1$ supporting line — consistent with its functional margin $1.4 > 1$ (geometric margin $1.4/\lVert w^\star\rVert = 1.4\sqrt5/2 = 0.7\sqrt5 \approx 1.565$ ✓, same number both ways).

## Problem 2 — Scaling, canonical and otherwise

**(i)** Wall from $(2w, 2b)$: $\{x : 2w^T x + 2b = 0\} = \{x : w^T x + b = 0\}$ — same set, same wall. Functional margin: $y^{(i)}(2w^T x^{(i)} + 2b) = 2\,y^{(i)}(w^T x^{(i)} + b)$ — doubled. Geometric margin: $\dfrac{2\,y^{(i)}(w^T x^{(i)} + b)}{\lVert 2w\rVert} = \dfrac{y^{(i)}(w^T x^{(i)} + b)}{\lVert w\rVert}$ — unchanged. (If $(w,b)$ already satisfied the canonical constraints $y^{(i)}(w^Tx^{(i)}+b)\ge1$, doubling keeps them satisfied a fortiori; either way, the wall is unchanged.)

**(ii)** One line: the canonical condition fixes the ruler — it selects the *unique* scale $(w, b)$ per wall whose nearest point has functional margin exactly $1$, so the primal's minimizer is a single $(w, b)$, not a ray.

**(iii)** Rescaling $w$ alone changes the wall itself: $\{x : (tw)^T x + b = 0\} \ne \{x : w^T x + b = 0\}$ for $t \ne 1$ — the normal direction is the same but the offset-to-normal ratio changes, so the boundary moves. Worse, the canonical constraints $y^{(i)}(tw^T x^{(i)} + b) \ge 1$ now describe a *different* feasible set, and minimizing $\frac12\lVert tw\rVert^2$ over it no longer finds the max-margin wall (practice assignment, Q3: the scaled weight vector "would not be a solution to the primal problem").

## Problem 3 — The symmetric square

**(i) Dual by symmetry.** $x^{(1)} = (1,1), y=+1$; $x^{(2)} = (1,-1), y=+1$; $x^{(3)} = (-1,1), y=-1$; $x^{(4)} = (-1,-1), y=-1$. Set $\alpha_i = \alpha$ (dual constraint $\sum \alpha_i y^{(i)} = \alpha + \alpha - \alpha - \alpha = 0$ ✓). $\sum_i \alpha_i = 4\alpha$. Quadratic coefficient: $\sum_{i,j} y^{(i)}y^{(j)}x^{(i)T}x^{(j)} = \lVert\sum_i y^{(i)}x^{(i)}\rVert^2$ with $\sum_i y^{(i)}x^{(i)} = (1,1) + (1,-1) + (1,-1) + (1,1) = (4, 0)$, norm-squared $16$. So $D(\alpha) = 4\alpha - \tfrac12\cdot 16\alpha^2 = \boxed{4\alpha - 8\alpha^2}$, maximized at $\alpha^\star = 4/16 = \boxed{1/4}$, $D_{\max} = 1 - 1/2 = 0.5$.

**(ii) Recover.** $w^\star = \tfrac14(4, 0) = \boxed{(1, 0)}$. From $x^{(1)}$: $1\cdot(1 + b) = 1 \Rightarrow \boxed{b^\star = 0}$ (check $x^{(3)}$: $(-1)(-1 + 0) = 1$ ✓). Width $= 2/\lVert(1,0)\rVert = \boxed{2}$. Supporting hyperplanes $\boxed{x_1 = 1}$ and $\boxed{x_1 = -1}$; boundary $\boxed{x_1 = 0}$.

**(iii)** Functional margins: $x^{(1)}: 1$; $x^{(2)}: 1$; $x^{(3)}: (-1)(-1) = 1$; $x^{(4)}: (-1)(-1) = 1$ — all exactly $1$, all active, all $\alpha_i = 1/4 > 0$: **all four points are support vectors**. Primal value $\frac12\lVert w^\star\rVert^2 = 0.5 =$ dual $0.5$ ✓.

## Problem 4 — Complementary slackness, both directions

**(i)** $\alpha_i^\star\big(1 - y^{(i)}(w^{\star T}x^{(i)} + b^\star)\big) = 0$ with $\alpha_i^\star > 0$: divide by the positive $\alpha_i^\star$ to get $1 - y^{(i)}(w^{\star T}x^{(i)} + b^\star) = 0$, i.e. $\boxed{y^{(i)}(w^{\star T}x^{(i)} + b^\star) = 1}$ — the point lies exactly on its supporting hyperplane.

**(ii)** Complementary slackness constrains only the *product*. If $y^{(i)}(w^{\star T}x^{(i)} + b^\star) = 1$ (point on the supporting hyperplane), then the second factor is $0$ and the product is $0$ *whatever* $\alpha_i^\star$ is — including $\alpha_i^\star = 0$. So a point can sit on the margin line yet carry no vote. In the assignment's words: every support vector lies on a supporting hyperplane, but not every point on a supporting hyperplane is a support vector; the implication $\alpha_i > 0 \Rightarrow$ (on margin) holds, the converse does not.

## Problem 5 — Delete the bystander

**(i)** Support-vector sum at $x_{\mathrm{test}} = (0,0)$:

$$\sum_{i:\alpha_i>0} \alpha_i y^{(i)} {x^{(i)}}^T x_{\mathrm{test}} + b^\star = 0.4\cdot(+1)\cdot 0 + 0.4\cdot(-1)\cdot 0 + (-0.2) = -0.2,$$

$\operatorname{sign}(-0.2) = \boxed{-1}$ — matches §32.6's direct computation.

**(ii)** $x^{(1)}$ had $\alpha_1^\star = 0$, so it appears nowhere in $w^\star = \sum_i \alpha_i^\star y^{(i)} x^{(i)}$. Deleting it: the remaining $(w^\star, b^\star, \alpha_2^\star, \alpha_3^\star)$ still satisfies every KKT condition on the reduced dataset — stationarity is unchanged (the deleted term was zero), primal feasibility and complementary slackness only *drop* a constraint, dual feasibility is untouched. By KKT sufficiency for this convex problem (§13.10), the reduced problem's optimum is unchanged: same $w^\star, b^\star$, same prediction $-1$. The guaranteeing condition is the KKT system as a whole — concretely, complementary slackness ($\alpha_1 = 0$ for the inactive constraint) is what certifies the point was irrelevant.

## Problem 6 — Three linear classifiers

**Similarities** (practice assignment, Q8): (1) all three are *linear* models; (2) the decision boundary is $w^T x + b = 0$ for all three; (3) all three are *discriminative* — they model the boundary directly, with no model of how the data was generated.

**Differences** (practice assignment, Q8): (1) logistic regression attaches an explicit probability to each point, and confidence grows with distance from the boundary — farther means surer; the perceptron and SVM have no such notion. (2) The SVM is the *principled* choice of wall — the max-margin one — and "max-margin classifiers will generalize better than perceptrons", which just take the first wall that works. (3) The hard-margin SVM demands separable data and is not robust to outliers (the soft-margin formulation fixes this); the perceptron is likewise only guaranteed to converge in the separable case, while logistic regression is more forgiving with outliers.
