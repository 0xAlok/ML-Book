# Solutions — Chapter 10: Unconstrained optimization and gradient descent

## Solution 1
$f(x) = x^2 - 6x + 5$.

i) $f'(x) = 2x - 6$. Set $= 0$: $2x = 6$, so $x = 3$ — the only critical point ($f$ is a polynomial, differentiable everywhere).
ii) $f''(x) = 2 > 0$: local minimum at $x = 3$ (§8.7.2).
iii) Complete the square: $f(x) = x^2 - 6x + 9 - 4 = (x - 3)^2 - 4 \ge -4 = f(3)$ for all $x$. So $x = 3$ is the **global minimum**, value $-4$.

## Solution 2
$f(x) = x^2$, $f'(x) = 2x$, $x_0 = 4$, $\eta = 0.1$.

i) $x_1 = x_0 - \eta f'(x_0) = 4 - 0.1 \cdot 8 = 4 - 0.8 = 3.2$.
ii) $f'(3.2) = 6.4$. $x_2 = 3.2 - 0.1 \cdot 6.4 = 3.2 - 0.64 = 2.56$.
iii) Values: $f(4) = 16$, $f(3.2) = 10.24$, $f(2.56) = 6.5536$. $16 > 10.24 > 6.5536$: decreasing at each step, homing in on $0$.

## Solution 3
$f(x) = (x - 5)^2$, $f'(x) = 2(x - 5)$, $x_0 = 10$.

(a) Constant $\eta = 1$:
- $x_1 = 10 - 1 \cdot 2(10 - 5) = 10 - 10 = 0$.
- $x_2 = 0 - 1 \cdot 2(0 - 5) = 0 + 10 = 10$.
- $x_3 = 10 - 10 = 0$.
Behaviour: pure oscillation $10 \to 0 \to 10 \to 0 \to \cdots$ — the step always overshoots the minimum at $5$ by the same amount, landing on the mirror image. It never converges.

(b) Schedule $\eta_t = \frac{1}{t+1}$:
- $t = 0$: $\eta_0 = 1$. $x_1 = 10 - 1 \cdot 10 = 0$.
- $t = 1$: $\eta_1 = \frac{1}{2}$. $f'(x_1) = 2(0 - 5) = -10$. $x_2 = 0 - \frac{1}{2}(-10) = 5$.
Behaviour: $x_2 = 5$ — the exact minimum, reached in two steps. Shrinking the step size killed the oscillation.

## Solution 4
$S_n = 1 + \frac{1}{2} + \cdots + \frac{1}{2^{n-1}}$ is geometric with ratio $\frac{1}{2}$:
$$S_n = \frac{1 - (1/2)^n}{1 - 1/2} = 2\bigl(1 - 2^{-n}\bigr) < 2 \quad \text{for every } n,$$
and $S_n \to 2$. So a step-size schedule $\eta_t = \frac{1}{2^t}$ has a *finite* total budget: starting anywhere, you can travel at most $2$ units — possibly stranding you short of the minimum (§10.4, eg 4).

$H_n = 1 + \frac{1}{2} + \cdots + \frac{1}{n}$: group the terms after the first two:
$$\underbrace{\tfrac{1}{3} + \tfrac{1}{4}}_{> 2\cdot \tfrac14 = \tfrac12} + \underbrace{\tfrac{1}{5} + \cdots + \tfrac{1}{8}}_{> 4\cdot \tfrac18 = \tfrac12} + \underbrace{\tfrac{1}{9} + \cdots + \tfrac{1}{16}}_{> 8\cdot \tfrac1{16} = \tfrac12} + \cdots$$
Each block exceeds $\frac{1}{2}$, and there are infinitely many blocks, so $H_n \to \infty$. The schedule $\eta_t = \frac{1}{t+1}$ still shrinks (no oscillation) but its *total* budget is infinite — you can reach a minimum no matter how far away it is. This is the two-enemies compromise of §10.4.

## Solution 5
$f(x_1, x_2) = x_1^2 + 4x_2^2$.

(a) $\nabla f = \begin{pmatrix} 2x_1 \\ 8x_2 \end{pmatrix}$.

(b) From $(2, 3)$, $\eta = 0.2$:
- $\nabla f(2, 3) = (4, 24)^T$. $\mathbf{x}_1 = (2, 3) - 0.2(4, 24) = (2 - 0.8,\ 3 - 4.8) = (1.2,\ -1.8)$.
- $\nabla f(1.2, -1.8) = (2.4, -14.4)^T$. $\mathbf{x}_2 = (1.2, -1.8) - 0.2(2.4, -14.4) = (1.2 - 0.48,\ -1.8 + 2.88) = (0.72,\ 1.08)$.

(c) $f(2, 3) = 4 + 4\cdot 9 = 40$. $f(1.2, -1.8) = 1.44 + 4\cdot 3.24 = 1.44 + 12.96 = 14.4$. $f(0.72, 1.08) = 0.5184 + 4\cdot 1.1664 = 0.5184 + 4.6656 = 5.184$. $40 > 14.4 > 5.184$: decreased both times. (The true minimum is $(0, 0)$ with value $0$.)

## Solution 6
$f(x, y) = x^2 + y^2$, $\nabla f = (2x, 2y)^T$. At $(2, -1)$: $\nabla f(2, -1) = (4, -2)^T$.

- $\mathbf{d}_1^T \nabla f = (1, 1)\cdot(4, -2) = 4 - 2 = 2 > 0$: **not** a descent direction — it points partly uphill (angle $< 90^\circ$ with the gradient).
- $\mathbf{d}_2^T \nabla f = (-2, 1)\cdot(4, -2) = -8 - 2 = -10 < 0$: **descent** direction — it makes an obtuse angle with the gradient, pointing into the downhill half-plane.
- $\mathbf{d}_3^T \nabla f = (1, 2)\cdot(4, -2) = 4 - 4 = 0$: **neither** — perpendicular to the gradient, so a small step is locally flat (it runs along the contour, §9.5).

## Solution 7

**(a)** $f(x, y) = x^2 + y^2 - 2x - 4y + 5$.

i) $\nabla f = (2x - 2,\ 2y - 4)^T = \mathbf{0}$ gives $(x, y) = (1, 2)$ — the only critical point.
ii) Hessian: $\mathbf{H} = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$, constant.
iii) $a = 2 > 0$, $\det = 4 > 0$ → positive definite → **local minimum**. $f(1, 2) = 1 + 4 - 2 - 8 + 5 = 0$. (Completing the square: $f = (x-1)^2 + (y-2)^2 \ge 0$ — it is in fact the global minimum.)

**(b)** $f(x, y) = -x^2 - y^2 + 4x$.

i) $\nabla f = (-2x + 4,\ -2y)^T = \mathbf{0}$ gives $(x, y) = (2, 0)$.
ii) $\mathbf{H} = \begin{pmatrix} -2 & 0 \\ 0 & -2 \end{pmatrix}$.
iii) $a = -2 < 0$, $\det = 4 > 0$ → negative definite → **local maximum**. $f(2, 0) = -4 - 0 + 8 = 4$. (Indeed $f = 4 - (x-2)^2 - y^2 \le 4$: the global maximum.)

**(c)** $f(x, y) = x^2 - y^2$.

i) $\nabla f = (2x, -2y)^T = \mathbf{0}$ gives $(0, 0)$.
ii) $\mathbf{H} = \begin{pmatrix} 2 & 0 \\ 0 & -2 \end{pmatrix}$.
iii) $\det = -4 < 0$ → indefinite → **saddle point**. Check: along the $x$-axis $f = x^2 \ge 0$ (uphill), along the $y$-axis $f = -y^2 \le 0$ (downhill) — up some ways, down others.

## Solution 8
We want the step to a smaller value: $f(x + \eta d) - f(x) < 0$.

From the Taylor approximation,
$$f(x + \eta d) - f(x) \approx \eta\, d\, f'(x),$$
so the requirement becomes $\eta\, d\, f'(x) < 0$. Since $\eta > 0$ is positive, it cannot flip the sign, so equivalently:
$$d \cdot f'(x) < 0$$
(the direction must oppose the derivative — §10.6).

Check $d = -f'(x)$: $d \cdot f'(x) = \bigl(-f'(x)\bigr)\cdot f'(x) = -\,f'(x)^2$. If $f'(x) \ne 0$, then $f'(x)^2 > 0$ and $-f'(x)^2 < 0$ strictly. So the condition holds strictly: for small enough $\eta$, the step $x \to x + \eta d$ is guaranteed to decrease $f$. If $f'(x) = 0$, then $d = \mathbf{0}$ and the algorithm does not move (§10.9).

## Solution 9
$f(x) = (x - a)^2$, $f'(x) = 2(x - a)$.

(a) $x_{t+1} = x_t - \eta \cdot 2(x_t - a)$. Subtract $a$ from both sides:
$$x_{t+1} - a = x_t - a - 2\eta(x_t - a) = (1 - 2\eta)(x_t - a).$$
The error $e_t = x_t - a$ evolves as $e_{t+1} = (1 - 2\eta)\, e_t$, so $e_t = (1 - 2\eta)^t e_0$.

(b) $|e_t| \to 0$ iff $|1 - 2\eta| < 1$, i.e. $-1 < 1 - 2\eta < 1$, i.e. $0 < \eta < 1$.

- If $\eta = 1$: $1 - 2\eta = -1$, so $e_t = (-1)^t e_0$ — pure oscillation, the §10.4 example ($10 \to 0 \to 10$).
- If $\eta > 1$: $|1 - 2\eta| > 1$ — the error grows geometrically: divergence (eg 5's blow-up).
- If $\eta \le 0$: the "step" moves uphill or nowhere.

Inside $(0, 1)$ the error shrinks by a constant factor each step — linear (geometric) convergence to $a$.

## Solution 10
$f(x) = (3x - 9)^2$: $f'(x) = 18x - 54$, $f''(x) = 18$.

$$x_1 = x_0 - \frac{f'(x_0)}{f''(x_0)} = 2 - \frac{18\cdot 2 - 54}{18} = 2 - \frac{-18}{18} = 2 + 1 = 3.$$

One step lands exactly on $x = 3$, the minimizer ($f(3) = 0$). Why: Newton's step minimizes the *second-order* Taylor model $f(x_t) + f'(x_t)(x - x_t) + \frac{1}{2}f''(x_t)(x - x_t)^2$, and since $f$ is itself quadratic, that model is *exact* (§9.8's note) — minimizing the model is minimizing $f$.

## Solution 11
(a) The update is $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta_t \nabla f(\mathbf{x}_t)$. If $\mathbf{x}_{t+1} = \mathbf{x}_t$, then $\eta_t \nabla f(\mathbf{x}_t) = \mathbf{0}$, and since $\eta_t > 0$:
$$\nabla f(\mathbf{x}_t) = \mathbf{0}.$$
The algorithm has reached a critical (stationary) point.

(b) "Gradient zero" is the *first-order* condition shared by minima, maxima, and saddles (§10.8): the update cannot distinguish them, because at all of them the step is $\mathbf{0}$. In particular, at a saddle point (e.g. $(0,0)$ for $f = x^2 - y^2$, where $\nabla f = \mathbf{0}$ but the Hessian is indefinite) gradient descent freezes just as it does at a minimum — the tutorial's caveat. So GD's guarantee is "converges to a stationary point (typically a local minimum)", never "converges to the global minimum".
