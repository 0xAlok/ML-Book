# Solutions — Chapter 12. Convex sets and convex functions

Full worked solutions to the Chapter 12 problem set. Same numbering as the chapter.

---

**1.** Prove from the definition that the half-space $\{\mathbf{x} : \mathbf{a}^T\mathbf{x} \le b\}$ is convex.

*Solution.* Let $S = \{\mathbf{x} : \mathbf{a}^T\mathbf{x} \le b\}$.

i) Take $\mathbf{x}_1, \mathbf{x}_2 \in S$: then $\mathbf{a}^T\mathbf{x}_1 \le b$ and $\mathbf{a}^T\mathbf{x}_2 \le b$. Take any $\lambda \in [0, 1]$.
ii) $\mathbf{a}^T(\lambda\mathbf{x}_1 + (1-\lambda)\mathbf{x}_2) = \lambda\,\mathbf{a}^T\mathbf{x}_1 + (1-\lambda)\,\mathbf{a}^T\mathbf{x}_2$. Since $\lambda \ge 0$ and $1 - \lambda \ge 0$, multiplying the two inequalities keeps their direction: $\lambda\,\mathbf{a}^T\mathbf{x}_1 \le \lambda b$ and $(1-\lambda)\,\mathbf{a}^T\mathbf{x}_2 \le (1-\lambda)b$.
iii) Adding: $\mathbf{a}^T(\lambda\mathbf{x}_1 + (1-\lambda)\mathbf{x}_2) \le \lambda b + (1-\lambda)b = b$. So the mixture lies in $S$. Since $\mathbf{x}_1, \mathbf{x}_2, \lambda$ were arbitrary, $S$ is convex. ∎

---

**2.** Prove that the Euclidean ball $B = \{\mathbf{x} : \lVert\mathbf{x}\rVert \le \theta\}$ is convex.

*Solution.*

i) Take $\mathbf{x}_1, \mathbf{x}_2 \in B$: $\lVert\mathbf{x}_1\rVert \le \theta$, $\lVert\mathbf{x}_2\rVert \le \theta$. Take any $\lambda \in [0, 1]$.
ii) Triangle inequality: $\lVert\lambda\mathbf{x}_1 + (1-\lambda)\mathbf{x}_2\rVert \le \lVert\lambda\mathbf{x}_1\rVert + \lVert(1-\lambda)\mathbf{x}_2\rVert$.
iii) Homogeneity with non-negative scalars: $= \lambda\lVert\mathbf{x}_1\rVert + (1-\lambda)\lVert\mathbf{x}_2\rVert \le \lambda\theta + (1-\lambda)\theta = \theta$.
iv) So $\lambda\mathbf{x}_1 + (1-\lambda)\mathbf{x}_2 \in B$. Arbitrary pair and $\lambda$ — $B$ is convex. ∎

---

**3.** Is the unit circle $C = \{\mathbf{x} \in \mathbb{R}^2 : \lVert\mathbf{x}\rVert = 1\}$ convex?

*Solution.* No. Take $\mathbf{x}_1 = (1, 0)$ and $\mathbf{x}_2 = (-1, 0)$, both in $C$ since $\lVert\mathbf{x}_1\rVert = \lVert\mathbf{x}_2\rVert = 1$. With $\lambda = \tfrac{1}{2}$:
$$\tfrac{1}{2}\mathbf{x}_1 + \tfrac{1}{2}\mathbf{x}_2 = (0, 0), \qquad \lVert(0,0)\rVert = 0 \ne 1,$$
so the midpoint is not in $C$. One violating triple is enough: **not convex**. (The *disk* $\lVert\mathbf{x}\rVert \le 1$ is convex — Problem 2 — but the boundary alone is not.)

---

**4.** Prove that the intersection of an *arbitrary collection* of convex sets is convex.

*Solution.* Let $\{S_i\}_{i \in I}$ be convex sets (any index set $I$) and $S = \bigcap_{i \in I} S_i$.

i) Take $\mathbf{x}_1, \mathbf{x}_2 \in S$ and $\lambda \in [0, 1]$.
ii) For *each* $i \in I$: $\mathbf{x}_1, \mathbf{x}_2 \in S_i$ (they lie in the intersection, hence in every member), so $\lambda\mathbf{x}_1 + (1-\lambda)\mathbf{x}_2 \in S_i$ by convexity of $S_i$.
iii) The mixture lies in every $S_i$, hence in $\bigcap_{i \in I} S_i = S$. So $S$ is convex. ∎

---

**5.** Prove that the convex hull $\mathrm{CH}(S)$ of a finite point set $S$ is a convex set. (Follow-up: show it equals the intersection of all convex sets containing $S$.)

*Solution.* Write $S = \{\mathbf{x}_1, \dots, \mathbf{x}_n\}$.

*Convexity.* Take $\mathbf{z}_1, \mathbf{z}_2 \in \mathrm{CH}(S)$: $\mathbf{z}_1 = \sum_{i=1}^n \lambda_i \mathbf{x}_i$, $\mathbf{z}_2 = \sum_{i=1}^n \mu_i \mathbf{x}_i$ with $\lambda_i, \mu_i \ge 0$, $\sum_i \lambda_i = \sum_i \mu_i = 1$. For $t \in [0, 1]$:
$$t\mathbf{z}_1 + (1-t)\mathbf{z}_2 = \sum_{i=1}^n \big(t\lambda_i + (1-t)\mu_i\big)\mathbf{x}_i.$$
The new coefficients satisfy $t\lambda_i + (1-t)\mu_i \ge 0$ and $\sum_i \big(t\lambda_i + (1-t)\mu_i\big) = t \cdot 1 + (1-t) \cdot 1 = 1$ — so the mixture is again a convex combination of the $\mathbf{x}_i$, i.e. in $\mathrm{CH}(S)$. ∎

*Follow-up (equivalence).* Let $\mathcal{C} = \bigcap\{C : C \text{ convex},\ S \subseteq C\}$.

i) $\mathrm{CH}(S) \subseteq \mathcal{C}$: take any convex $C \supseteq S$ and any $\mathbf{z} = \sum_{i=1}^n \lambda_i \mathbf{x}_i \in \mathrm{CH}(S)$. By induction on $n$, convex combinations of points of $C$ stay in $C$ (base $n = 2$ is the definition; for $n+1$, if $\lambda_{n+1} = 1$ then $\mathbf{z} = \mathbf{x}_{n+1} \in C$, else $\mathbf{z} = (1-\lambda_{n+1})\sum_{i=1}^n \frac{\lambda_i}{1-\lambda_{n+1}}\mathbf{x}_i + \lambda_{n+1}\mathbf{x}_{n+1}$, a mixture of two points of $C$ by the induction hypothesis). So $\mathbf{z} \in C$ for every such $C$, hence $\mathbf{z} \in \mathcal{C}$.
ii) $\mathcal{C} \subseteq \mathrm{CH}(S)$: $\mathrm{CH}(S)$ is convex (proved above) and contains $S$ (take $\lambda_i = 1$ for $\mathbf{x}_i$), so it is one of the sets being intersected — and an intersection is contained in each of its members.
iii) Both inclusions give $\mathrm{CH}(S) = \mathcal{C}$. ∎

---

**6.** Prove $f(x) = |x|$ is convex on $\mathbb{R}$ using the chord inequality. Where do Def 3 and Def 4 break down?

*Solution.* Take any $x_1, x_2$ and $\lambda \in [0, 1]$:
$$f(\lambda x_1 + (1-\lambda)x_2) = |\lambda x_1 + (1-\lambda)x_2| \le |\lambda x_1| + |(1-\lambda)x_2| = \lambda|x_1| + (1-\lambda)|x_2| = \lambda f(x_1) + (1-\lambda)f(x_2),$$
using the triangle inequality then $|\lambda x| = \lambda|x|$ for $\lambda \ge 0$. So Def 2 holds: **convex**. ✓

Def 3 needs $f$ differentiable everywhere — $|x|$ has no derivative at $x = 0$ (left slope $-1 \ne$ right slope $+1$), so $\nabla f(0)$ does not exist and the tangent lower bound cannot even be stated there. Def 4 needs $f$ twice differentiable — it fails at $0$ for the same reason (it is not even once differentiable there). Both definitions are inapplicable at the kink; Def 2 works globally.

---

**7.** Show $f(x) = \sin x$ is *not* convex on $[0, 2\pi]$.

*Solution.* Take $x_1 = 0$, $x_2 = \pi$, $\lambda = \tfrac{1}{2}$. Left side:
$$f(\tfrac{1}{2} \cdot 0 + \tfrac{1}{2} \cdot \pi) = \sin(\pi/2) = 1.$$
Right side:
$$\tfrac{1}{2}f(0) + \tfrac{1}{2}f(\pi) = \tfrac{1}{2} \cdot 0 + \tfrac{1}{2} \cdot 0 = 0.$$
Convexity demands $1 \le 0$ — false. Verdict: **not convex**. (Geometrically: the chord from $(0, 0)$ to $(\pi, 0)$ is the $x$-axis, and the graph bulges above it.)

---

**8.** Classify $f(x, y) = x^2 + xy + y^2$ via the Hessian.

*Solution.*

i) First partials: $f_x = 2x + y$, $f_y = x + 2y$.
ii) Second partials: $f_{xx} = 2$, $f_{xy} = 1$, $f_{yx} = 1$, $f_{yy} = 2$. So $\mathbf{H} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ (symmetric ✓).
iii) $2 \times 2$ test: $a = f_{xx} = 2 > 0$ and $\det(\mathbf{H}) = 2 \cdot 2 - 1 \cdot 1 = 3 > 0$ — positive definite, hence PSD. (Eigenvalue check: $\det(\mathbf{H} - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = 0$ gives $\lambda = 3, 1$, both $> 0$. ✓)
iv) Verdict: **convex**. (Indeed $f(x,y) = \tfrac{1}{2}(x+y)^2 + \tfrac{1}{2}(x^2 + y^2)$, a sum of convex pieces.)

---

**9.** Show $f(x) = -\ln x$ is convex on $(0, \infty)$.

*Solution.* $f'(x) = -1/x$, $f''(x) = 1/x^2$. For every $x > 0$, $f''(x) = 1/x^2 > 0 \ge 0$. By the 1-D second-derivative test (Def 4 with $d = 1$), $f$ is **convex** on $(0, \infty)$. ✓

---

**10.** Least squares via the Hessian: $f(\mathbf{w}) = \lVert X\mathbf{w} - \mathbf{y}\rVert^2$.

*Solution.*

(a) Expand: $f(\mathbf{w}) = (X\mathbf{w} - \mathbf{y})^T(X\mathbf{w} - \mathbf{y}) = \mathbf{w}^TX^TX\mathbf{w} - 2\mathbf{y}^TX\mathbf{w} + \mathbf{y}^T\mathbf{y}$. The matrix $X^TX$ is symmetric. Gradient: $\nabla f(\mathbf{w}) = 2X^TX\mathbf{w} - 2X^T\mathbf{y}$ (using $\nabla_{\mathbf{w}}\,\mathbf{w}^TA\mathbf{w} = 2A\mathbf{w}$ for symmetric $A$, and $\nabla_{\mathbf{w}}\,\mathbf{c}^T\mathbf{w} = \mathbf{c}$). Differentiating once more: $\nabla^2 f(\mathbf{w}) = 2X^TX$, constant in $\mathbf{w}$.

(b) For any $\mathbf{z} \in \mathbb{R}^d$:
$$\mathbf{z}^T(2X^TX)\mathbf{z} = 2\,(X\mathbf{z})^T(X\mathbf{z}) = 2\lVert X\mathbf{z}\rVert^2 \ge 0.$$
So the Hessian is positive semi-definite (this is exactly the §7.9 eigenvalue picture: all eigenvalues of $X^TX$ are $\ge 0$).

(c) A twice-differentiable function with PSD Hessian everywhere is convex (Def 4). Hence $f$ is **convex** — the Hessian route to the same conclusion as eg 11. ∎

---

**11.** Prove that the set of all global minimizers of a convex function $f$ is a convex set.

*Solution.* Let $f^\star = \min_{\mathbf{x}} f(\mathbf{x})$ (assume the minimum is attained) and $M = \{\mathbf{x} : f(\mathbf{x}) = f^\star\}$.

i) Take $\mathbf{x}_1, \mathbf{x}_2 \in M$: $f(\mathbf{x}_1) = f(\mathbf{x}_2) = f^\star$. Take $\lambda \in [0, 1]$.
ii) By convexity: $f(\lambda\mathbf{x}_1 + (1-\lambda)\mathbf{x}_2) \le \lambda f(\mathbf{x}_1) + (1-\lambda)f(\mathbf{x}_2) = \lambda f^\star + (1-\lambda)f^\star = f^\star$.
iii) But $f^\star$ is the *global minimum value*, so $f(\cdot) \ge f^\star$ everywhere — combined with (ii), $f(\lambda\mathbf{x}_1 + (1-\lambda)\mathbf{x}_2) = f^\star$, i.e. the mixture lies in $M$.
iv) So $M$ is convex. ∎ (This is why a flat-bottomed convex function has an *interval* of minimizers in 1-D.)

---

**12.** For $f(x, y) = (x - 3)^2 + (y + 1)^2$: (a) find where $\nabla f = \mathbf{0}$; (b) certify it is the global minimum; (c) write the tangent-plane lower bound there.

*Solution.*

(a) $\nabla f(x, y) = \big(2(x - 3),\, 2(y + 1)\big)$. Setting to zero: $x = 3$, $y = -1$. The unique stationary point is $\mathbf{x}^\star = (3, -1)$ with $f(\mathbf{x}^\star) = 0$.

(b) $f$ is differentiable, and convex: it is a sum of the convex quadratics $(x-3)^2$ and $(y+1)^2$ (each has second derivative $2 > 0$; sums of convex are convex, Property 1). By the §12.10 theorem, for differentiable convex $f$, $\nabla f(\mathbf{x}^\star) = \mathbf{0}$ **iff** $\mathbf{x}^\star$ is a global minimum. Hence $(3, -1)$ is certified the global minimizer — no Hessian interrogation needed.

(c) Def 3 at $\mathbf{x}^\star$: $f(\mathbf{y}) \ge f(\mathbf{x}^\star) + (\mathbf{y} - \mathbf{x}^\star)^T\nabla f(\mathbf{x}^\star) = 0 + 0 = 0$ for all $\mathbf{y} \in \mathbb{R}^2$. Explicitly: $(y_1 - 3)^2 + (y_2 + 1)^2 \ge 0$ — the tangent "plane" at the bottom of the bowl is the flat plane $z = 0$, and the bowl sits above it everywhere. ∎
