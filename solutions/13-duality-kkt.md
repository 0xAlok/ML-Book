# Solutions — 13. Duality and KKT conditions

**1.** $\min_x x^2$ s.t. $x \ge 2$.

$h(x) = 2 - x \le 0$. Lagrangian: $L(x, \lambda) = x^2 + \lambda(2 - x)$, $\lambda \ge 0$.
Primal: $\min_x \max_{\lambda \ge 0} L(x, \lambda)$; geometrically the answer is $x^\star = 2$, $p^\star = 4$.
Dual function: $g(\lambda) = \min_x [x^2 + \lambda(2 - x)]$. Unconstrained min: $\frac{d}{dx} = 2x - \lambda = 0 \Rightarrow x = \lambda/2$. So
$$g(\lambda) = \frac{\lambda^2}{4} + \lambda\left(2 - \frac{\lambda}{2}\right) = 2\lambda - \frac{\lambda^2}{4}.$$
Dual: $\max_{\lambda \ge 0} g(\lambda)$: $g'(\lambda) = 2 - \lambda/2 = 0 \Rightarrow \lambda^\star = 4 \ge 0$ ✓; $g'' = -1/2 < 0$ so this is the max. $d^\star = g(4) = 8 - 4 = 4$.
Gap: $p^\star - d^\star = 4 - 4 = 0$ — strong duality (convex: $f'' = 2 > 0$, $h$ linear).
KKT at $(2, 4)$: stationarity $2(2) - 4 = 0$ ✓; CS $4(2-2) = 0$ ✓; PF $2 - 2 = 0 \le 0$ ✓; DF $4 \ge 0$ ✓.

**2.** Tutorial LP at $\mathbf{x}^\star = (18, 22)$, $\mathbf{u}^\star = (9, 4, 0, 0)$, with $h_1 = x_1 - x_2 + 4$, $h_2 = -3x_1 + 2x_2 + 10$, $h_3 = -x_1$, $h_4 = -x_2$.

Residuals: $h_1 = 18 - 22 + 4 = 0$; $h_2 = -54 + 44 + 10 = 0$; $h_3 = -18$; $h_4 = -22$. All $\le 0$ — primal feasibility ✓.
CS products: $u_1 h_1 = 9 \cdot 0 = 0$; $u_2 h_2 = 4 \cdot 0 = 0$; $u_3 h_3 = 0 \cdot (-18) = 0$; $u_4 h_4 = 0 \cdot (-22) = 0$ ✓.
Stationarity: $\frac{\partial L}{\partial x_1} = 3 + u_1 - 3u_2 - u_3 = 3 + 9 - 12 - 0 = 0$ ✓; $\frac{\partial L}{\partial x_2} = 1 - u_1 + 2u_2 - u_4 = 1 - 9 + 8 - 0 = 0$ ✓.
Dual feasibility: $9, 4, 0, 0 \ge 0$ ✓. All four KKT conditions hold — and since an LP is convex, $(18, 22)$ is certified globally optimal with $f^\star = 76$.

**3.** Diet problem.

(a) Primal feasibility of $(2, 1.5)$: $3(2) = 6 \ge 6$ ✓; $2(2) + 4(1.5) = 10 \ge 10$ ✓; $2(2) + 5(1.5) = 11.5 \ge 8$ ✓; $2, 1.5 \ge 0$ ✓. Value: $50(2) + 80(1.5) = 220$.
(b) Dual feasibility of $(10/3, 20, 0)$: $3(10/3) + 2(20) + 2(0) = 10 + 40 = 50 \le 50$ ✓; $4(20) + 5(0) = 80 \le 80$ ✓; all $\ge 0$ ✓. Value: $6(10/3) + 10(20) + 8(0) = 20 + 200 = 220$.
(c) For a minimization primal, any feasible point upper-bounds the optimum: $p^\star \le 220$. For the maximization dual, any feasible point lower-bounds it: $d^\star \ge 220$. Weak duality: $d^\star \le p^\star$. Chain: $220 \le d^\star \le p^\star \le 220$ — equality throughout, so $p^\star = d^\star = 220$ and both points are optimal. This is the certificate argument in action: matching feasible values prove optimality on both sides.

**4.** Concavity of $g$ (full steps). Fix $\theta \in [0,1]$, $\lambda_1, \lambda_2 \ge 0$.
$$g(\theta\lambda_1 + (1-\theta)\lambda_2) = \min_{\mathbf{x}} L(\mathbf{x}, \theta\lambda_1 + (1-\theta)\lambda_2) = \min_{\mathbf{x}} \big[\theta L(\mathbf{x},\lambda_1) + (1-\theta)L(\mathbf{x},\lambda_2)\big],$$
using that $L$ is affine in $\lambda$ for fixed $\mathbf{x}$.
Now for *every* $\mathbf{x}$: $\theta L(\mathbf{x},\lambda_1) + (1-\theta)L(\mathbf{x},\lambda_2) \ge \theta \min_{\mathbf{z}} L(\mathbf{z},\lambda_1) + (1-\theta)\min_{\mathbf{z}} L(\mathbf{z},\lambda_2) = \theta g(\lambda_1) + (1-\theta)g(\lambda_2)$, since each $L(\mathbf{x}, \cdot)$ is at least its own minimum and $\theta, 1-\theta \ge 0$. The left side, minimized over $\mathbf{x}$, is still $\ge$ the right side (a constant in $\mathbf{x}$):
$$g(\theta\lambda_1 + (1-\theta)\lambda_2) \ge \theta g(\lambda_1) + (1-\theta)g(\lambda_2),$$
the concavity chord inequality. ∎

**5.** $\min_x (x-3)^2$ s.t. $x \le 1$: $h(x) = x - 1 \le 0$, $L(x,\lambda) = (x-3)^2 + \lambda(x-1)$, $\lambda \ge 0$.

(i) $x = 0$: $L(0,\lambda) = 9 + \lambda(-1) = 9 - \lambda$. Over $\lambda \ge 0$ the max is at $\lambda = 0$: value $9$. $J(0) = (0-3)^2 = 9$ since $0 \le 1$ (feasible) ✓.
(ii) $x = 2$: $L(2,\lambda) = 1 + \lambda(1) = 1 + \lambda \to \infty$ as $\lambda \to \infty$. $J(2) = \infty$ since $2 > 1$ (infeasible) ✓.

**6.** $\min_{x,y} x^2 + y^2$ s.t. $x + y \ge 1$.

(a) $h(x,y) = 1 - x - y \le 0$; $L = x^2 + y^2 + \lambda(1 - x - y)$, $\lambda \ge 0$.
(b) $\frac{\partial L}{\partial x} = 2x - \lambda = 0$, $\frac{\partial L}{\partial y} = 2y - \lambda = 0 \Rightarrow x = y = \lambda/2$. Then $g(\lambda) = 2(\lambda^2/4) + \lambda(1 - \lambda) = \lambda - \lambda^2/2$. Dual $\max_{\lambda \ge 0} g$: $g' = 1 - \lambda = 0 \Rightarrow \lambda^\star = 1$; $d^\star = g(1) = 1 - 1/2 = 1/2$.
(c) Weak duality: $p^\star \ge d^\star = 1/2$.
(d) Primal: by Cauchy–Schwarz, $x^2 + y^2 \ge (x+y)^2/2 \ge 1/2$ on the feasible set (since $x+y \ge 1$). Equality at $x = y = 1/2$, which is feasible ($1/2 + 1/2 = 1 \ge 1$ ✓). So $p^\star = 1/2 = d^\star$ — gap $0$. (Convex: Hessian $\mathrm{diag}(2,2) \succ 0$; $h$ linear — strong duality applies.)

**7.** KKT at $(1, 2)$ for $\min x^2$ s.t. $1 - x \le 0$: stationarity $2(1) - 2 = 0$ ✓; complementary slackness $2(1-1) = 0$ ✓; primal feasibility $1 - 1 = 0 \le 0$ ✓; dual feasibility $2 \ge 0$ ✓.

**8.** Testing $(x, u) = (0, 0)$: stationarity $2(0) - 0 = 0$ ✓ (holds!); CS $0 \cdot (1-0) = 0$ ✓ (holds!); dual feasibility $0 \ge 0$ ✓ (holds!); primal feasibility $1 - 0 = 1 \le 0$ ✗ **fails**. The proposal is not even feasible. Moral: stationarity alone certifies nothing — exactly §10.8's warning that first-order conditions need the full interrogation. Here the missing interrogation is feasibility, and KKT supplies it.

**9.** $\min 2x_1 + 3x_2$ s.t. $x_1 + x_2 \ge 4$, $x_1, x_2 \ge 0$.

(a) Standard form: $h_1 = 4 - x_1 - x_2 \le 0$, $h_2 = -x_1 \le 0$, $h_3 = -x_2 \le 0$.
(b) $L = 2x_1 + 3x_2 + u_1(4 - x_1 - x_2) - u_2x_1 - u_3x_2$. KKT: stationarity $2 - u_1 - u_2 = 0$, $3 - u_1 - u_3 = 0$; CS $u_1(4-x_1-x_2) = 0$, $u_2 x_1 = 0$, $u_3 x_2 = 0$; PF the three $h_i \le 0$; DF $u_1, u_2, u_3 \ge 0$.
(c) Guess: $h_1$ active, $x_2 = 0$ (cheaper variable $x_1$ does all the work), so $u_3 = 0$ free, $u_1 > 0$, $u_2 = 0$ (since $x_1 > 0$ expected). Then stationarity: $2 - u_1 = 0 \Rightarrow u_1 = 2$; $3 - 2 - u_3 = 0 \Rightarrow u_3 = 1$. CS: $h_1 = 0 \Rightarrow x_1 + x_2 = 4$; with $x_2 = 0$: $\mathbf{x}^\star = (4, 0)$. Verify: PF: $4 - 4 - 0 = 0 \le 0$ ✓, $-4 \le 0$ ✓, $0 \le 0$ ✓. CS: $2(0) = 0$ ✓, $0 \cdot 4 = 0$ ✓, $1 \cdot 0 = 0$ ✓. DF: $2, 0, 1 \ge 0$ ✓. $f^\star = 2(4) + 3(0) = 8$. (Sanity: on $x_1 + x_2 = 4$, $f = 2x_1 + 3(4-x_1) = 12 - x_1$, minimized at the largest feasible $x_1 = 4$. ✓) LP ⇒ convex ⇒ globally optimal.

**10.** Certificate theorem (full steps). Let the problem be convex ($f$ convex, each $h_i$ convex, equalities affine) and $(\mathbf{x}^\star, \mathbf{u}^\star)$ satisfy KKT.

i) Primal feasibility gives $\mathbf{x}^\star$ feasible, so $p^\star \le f(\mathbf{x}^\star)$ (the min over the feasible set is at most the value at one feasible point).
ii) For fixed $\mathbf{u}^\star \ge 0$, $L(\cdot, \mathbf{u}^\star) = f + \sum u_i^\star h_i$ is convex (nonneg-weighted sum of convex functions). Stationarity $\nabla_{\mathbf{x}} L(\mathbf{x}^\star, \mathbf{u}^\star) = \mathbf{0}$ plus convexity ⇒ $\mathbf{x}^\star$ globally minimizes $L(\cdot, \mathbf{u}^\star)$ (§12.10). Hence $g(\mathbf{u}^\star) = \min_{\mathbf{x}} L(\mathbf{x}, \mathbf{u}^\star) = L(\mathbf{x}^\star, \mathbf{u}^\star) = f(\mathbf{x}^\star) + \sum_i u_i^\star h_i(\mathbf{x}^\star) = f(\mathbf{x}^\star)$, using complementary slackness $u_i^\star h_i(\mathbf{x}^\star) = 0$.
iii) $d^\star = \max_{\mathbf{u} \ge 0} g(\mathbf{u}) \ge g(\mathbf{u}^\star) = f(\mathbf{x}^\star)$.
iv) Chain: $f(\mathbf{x}^\star) = g(\mathbf{u}^\star) \le d^\star \le p^\star \le f(\mathbf{x}^\star)$, the middle inequality being weak duality. Equality throughout: $\boxed{p^\star = d^\star = f(\mathbf{x}^\star)}$. So KKT alone — no optimizer run — certifies $\mathbf{x}^\star$ primal-optimal, $\mathbf{u}^\star$ dual-optimal, and zero duality gap. ∎
