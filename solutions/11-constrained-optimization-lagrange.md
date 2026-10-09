# Solutions — Chapter 11: Constrained optimization

## Problem 1

$L(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda h(\mathbf{x})$.

i) $\nabla_{\mathbf{x}} L = \nabla f(\mathbf{x}) - \lambda \nabla h(\mathbf{x}) = \mathbf{0}$, i.e. $\nabla f(\mathbf{x}^\star) = \lambda\, \nabla h(\mathbf{x}^\star)$.
ii) $\frac{\partial L}{\partial \lambda} = -h(\mathbf{x}) = 0$, i.e. $h(\mathbf{x}^\star) = 0$.

One line: the lectures write $\nabla f = -\lambda \nabla h$ with $\lambda$ unrestricted, and since $\lambda$ ranges over all of $\mathbb{R}$, renaming $\lambda \to -\lambda$ turns one form into the other — they are the same condition.

## Problem 2

$f(x, y) = (x-3)^2 + (y-4)^2$, constraint $x + y = 3$. $L = (x-3)^2 + (y-4)^2 - \lambda(x + y - 3)$.

i) $2(x - 3) - \lambda = 0$, $2(y - 4) - \lambda = 0$ — so $2(x-3) = 2(y-4)$, i.e. $y = x + 1$.
ii) Constraint: $x + (x + 1) = 3$ — $x = 1$, $y = 2$, $\lambda = 2(1-3) = -4$.
iii) Value: $f(1, 2) = 4 + 4 = 8$. Single candidate; $f \to \infty$ at infinity along the line — the global minimum.

Check: distance from $(3,4)$ to the line $x + y - 3 = 0$ is $\frac{|3 + 4 - 3|}{\sqrt{2}} = \frac{4}{\sqrt{2}} = 2\sqrt{2}$; squared distance $= 8$. ✓ Also $\nabla f(1,2) = (-4, -4) = -4(1,1) = \lambda \nabla h$. ✓

## Problem 3

Maximize $f = xyz$ subject to $g(x,y,z) = 2xz + 2yz + xy - 12 = 0$, $x,y,z > 0$ (lectures' sign form $\nabla f = -\lambda \nabla g$).

i) $\nabla f = (yz, xz, xy)$, $\nabla g = (2z + y, 2z + x, 2x + 2y)$. Equations:
$$yz = -\lambda(2z + y), \qquad xz = -\lambda(2z + x), \qquad xy = -\lambda(2x + 2y).$$
(The sign of $\lambda$ is free for an equality; $-\lambda$ versus $\lambda$ makes no difference.)
ii) Multiply by $x$, $y$, $z$ respectively:
$$xyz = -\lambda(2xz + xy) = -\lambda(2yz + xy) = -\lambda(2xz + 2yz).$$
iii) Since $xyz > 0$ (nonzero volume, positive sides): from the first two, $2xz + xy = 2yz + xy$ — $2xz = 2yz$ — $x = y$. From the second and third, $2yz + xy = 2xz + 2yz$ — $xy = 2xz$ — $y = 2z$. So $x = y = 2z$.
iv) Constraint: $2xz + 2yz + xy = 2(2z)(z) + 2(2z)(z) + (2z)(2z) = 12z^2 = 12$ — $z = 1$ (positive root), $(x, y, z) = (2, 2, 1)$.
v) Volume: $f = 2 \cdot 2 \cdot 1 = 4\,\text{m}^3$, the maximum (boundary cases with a zero side give volume $0$).

## Problem 4

$f(x,y) = x^2 + xy + y^2$, $x^2 + y^2 = 1$. $L = x^2 + xy + y^2 - \lambda(x^2 + y^2 - 1)$.

i) $(2 - 2\lambda)x + y = 0$, $x + (2 - 2\lambda)y = 0$. Put $\mu = 2 - 2\lambda$: $\mu x + y = 0$, $x + \mu y = 0$.
ii) $y = -\mu x$ into the second: $x - \mu^2 x = x(1 - \mu^2) = 0$. $x = 0$ forces $y = 0$ (infeasible), so $\mu = \pm 1$.
iii) $\mu = 1$: $y = -x$; circle gives $2x^2 = 1$ — $(\tfrac{1}{\sqrt{2}}, -\tfrac{1}{\sqrt{2}})$, $(-\tfrac{1}{\sqrt{2}}, \tfrac{1}{\sqrt{2}})$. $f = \tfrac{1}{2} - \tfrac{1}{2} + \tfrac{1}{2} = \tfrac{1}{2}$ at both — the **minimizers**.
iv) $\mu = -1$: $y = x$; $x = \pm \tfrac{1}{\sqrt{2}}$ — $(\tfrac{1}{\sqrt{2}}, \tfrac{1}{\sqrt{2}})$, $(-\tfrac{1}{\sqrt{2}}, -\tfrac{1}{\sqrt{2}})$. $f = \tfrac{1}{2} + \tfrac{1}{2} + \tfrac{1}{2} = \tfrac{3}{2}$ at both — the **maximizers**.

## Problem 5

$f(x,y) = x^2 + y^2$, $g(x,y) = x + y - 1 \le 0$. The unconstrained minimum $(0,0)$ has $g(0,0) = -1 < 0$ — feasible with slack. So the constraint is **inactive** at the optimum: $\mathbf{x}^\star = (0,0)$, $f = 0$ (global, since $f \ge 0$ everywhere). Multiplier $\lambda = 0$; complementary slackness: $\lambda \cdot g(\mathbf{x}^\star) = 0 \cdot (-1) = 0$. ✓

## Problem 6

$f(x,y) = x^2 + y^2$, $g(x,y) = x + y + 1 \le 0$. Unconstrained minimum $(0,0)$ has $g(0,0) = 1 > 0$ — infeasible, so the constraint is **active**: solve $x + y + 1 = 0$.

i) $\nabla f = (2x, 2y) = -\lambda(1,1) = -\lambda \nabla g$ — so $2x = -\lambda$, $2y = -\lambda$, $x = y = -\lambda/2$.
ii) $x + y = -1$ gives $-\lambda = -1$ — $\lambda = 1 \ge 0$. ✓
iii) $\mathbf{x}^\star = (-\tfrac{1}{2}, -\tfrac{1}{2})$, $f = \tfrac{1}{2}$. Anti-parallel check: $\nabla f(-\tfrac{1}{2},-\tfrac{1}{2}) = (-1,-1)$, $\nabla g = (1,1)$ — opposite directions. ✓ Complementary slackness: $g(\mathbf{x}^\star) = 0$. ✓

## Problem 7

$f(x,y) = (x-2)^2 + (y-2)^2$, $g(x,y) = x^2 + y^2 - 1 \le 0$.

(a) Unconstrained minimum: $\nabla f = (2(x-2), 2(y-2)) = \mathbf{0}$ — $(2,2)$. $g(2,2) = 8 - 1 = 7 > 0$ — infeasible. So the constraint is active.

(b) Equality problem $x^2 + y^2 = 1$, lecture form $\nabla f = -\lambda \nabla g$:
$$2(x - 2) = -\lambda \cdot 2x, \qquad 2(y - 2) = -\lambda \cdot 2y.$$
So $x(1 + \lambda) = 2$, $y(1 + \lambda) = 2$ — $x = y = \tfrac{2}{1+\lambda}$. Circle: $2x^2 = 1$ — $x = \pm \tfrac{1}{\sqrt{2}}$.

- $x = y = \tfrac{1}{\sqrt{2}}$: $\tfrac{1}{\sqrt{2}}(1 + \lambda) = 2$ — $\lambda = 2\sqrt{2} - 1 \approx 1.83 \ge 0$. ✓ Minimizer. Value: $f = 2(\tfrac{1}{\sqrt{2}} - 2)^2 = 2(\tfrac{9}{2} - 2\sqrt{2}) = 9 - 4\sqrt{2} \approx 3.34$.
- $x = y = -\tfrac{1}{\sqrt{2}}$: $\lambda = 1 - 2\sqrt{2} \approx -1.83 < 0$ — fails the sign condition: the maximizer. Value: $9 + 4\sqrt{2} \approx 14.66$.

The $\lambda \ge 0$ rule picks the minimizer and rejects the maximizer.

## Problem 8

$f(x) = (x-4)^2$, $x \le 1$, $\eta = 0.25$, $x_0 = 0$. Update: $x_{t+1} = \min(x_t - 0.25 \cdot 2(x_t - 4),\, 1) = \min(0.5x_t + 2,\, 1)$.

i) $x_1 = \min(0.5 \cdot 0 + 2,\, 1) = \min(2, 1) = 1$.
ii) $x_2 = \min(0.5 \cdot 1 + 2,\, 1) = \min(2.5, 1) = 1$.
iii) $x_3 = 1$.

Constrained optimum: the closest point of $(-\infty, 1]$ to $4$ is $x^\star = 1$, $f = 9$ — the iterates reach it in one step and stay.

## Problem 9

$f(x) = (x-5)^2$, $x \le 2$, $\eta = 0.1$, $x_0 = 0$.

(a) Plain GD: $x_{t+1} = x_t - 0.2(x_t - 5) = 0.8x_t + 1$. $x_1 = 1$, $x_2 = 0.8 + 1 = 1.8$, $x_3 = 1.44 + 1 = 2.44$. $2.44 > 2$ — infeasible; plain GD leaves the feasible set on the third step (and would keep marching to $5$).

(b) Projecting $x_3$: $\min(2.44, 2) = 2$ — exactly eg 6's $x_3$. With projection every step, the iterates are $0 \to 1 \to 1.8 \to 2 \to 2$, frozen at the constrained optimum $x^\star = 2$.

## Problem 10

$L = x^2 + y^2 + z^2 - \lambda_1(x + y + z - 1) - \lambda_2(x - y)$.

i) $2x = \lambda_1 + \lambda_2$, $2y = \lambda_1 - \lambda_2$, $2z = \lambda_1$.
ii) Constraint $x - y = 0$ gives $x = y$, so $\lambda_2 = 0$ and $x = z = \lambda_1/2$.
iii) $x + y + z = 1$ gives $\tfrac{3\lambda_1}{2} = 1$ — $\lambda_1 = \tfrac{2}{3}$, $x = y = z = \tfrac{1}{3}$.
iv) Value: $f = 3 \cdot \tfrac{1}{9} = \tfrac{1}{3}$ — the global minimum (single candidate, $f \to \infty$ at infinity).

Geometrically: the two planes intersect in a line; the optimum is the point of that line closest to the origin, with $f$ the squared distance.

## Problem 11

Equality $h(\mathbf{x}) = 0$: to first order, feasible directions satisfy $\mathbf{d}^T \nabla h(\mathbf{x}^\star) = 0$ — a *line*, symmetric: if $\mathbf{d}$ is feasible, so is $-\mathbf{d}$. The descent directions are the half-space $\mathbf{d}^T \nabla f < 0$. These avoid each other exactly when the line is perpendicular to $\nabla f$ — i.e. $\nabla f \parallel \nabla h$ — and parallelism covers *both* directions ($\lambda > 0$ and $\lambda < 0$).

Inequality $g(\mathbf{x}) \le 0$: feasible directions include the *half-space* $\mathbf{d}^T \nabla g < 0$ — one-sided, not symmetric. Only the anti-parallel configuration ($\nabla f = -\lambda \nabla g$, $\lambda > 0$) points the descent half-space entirely away from the feasible half-space. With $\lambda < 0$ (parallel), every descent direction would be feasible — the point could be improved, so it cannot be a minimum.
