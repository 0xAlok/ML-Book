# Solutions — Chapter 33: Soft-margin SVM and the kernel trick

*Full worked solutions to the Chapter 33 problem set. Every number recomputed independently (hand algebra cross-checked in numpy; see the review log).*

## Problem 1 — Reading the slack

Given $w^\star = (1,0)$, $b^\star = 0$; points $x^{(1)} = (1,1), x^{(2)} = (1,-1)$ ($y = +1$); $x^{(3)} = (-1,1), x^{(4)} = (-1,-1)$ ($y = -1$); $x^{(5)} = (0.5,0)$ ($y = -1$).

**(i) The slacks.** $\xi_i = \max\big(1 - y^{(i)}(w^{\star T}x^{(i)} + b^\star),\, 0\big)$:

- Point 1: $1 - (+1)(1) = 0 \Rightarrow \boxed{\xi_1 = 0}$.
- Point 2: $1 - (+1)(1) = 0 \Rightarrow \boxed{\xi_2 = 0}$.
- Point 3: $1 - (-1)(-1) = 1 - 1 = 0 \Rightarrow \boxed{\xi_3 = 0}$.
- Point 4: $1 - (-1)(-1) = 0 \Rightarrow \boxed{\xi_4 = 0}$.
- Point 5: $1 - (-1)(0.5) = 1 + 0.5 = 1.5 \Rightarrow \boxed{\xi_5 = 1.5}$.

**(ii) Regimes.** Points 1–4: $\xi_i = 0$ (on the supporting hyperplanes, no bribe). Point 5: $\xi_5 = 1.5 > 1$ (crossed the wall — misclassified).

**(iii)** The outlier pays the largest bribe: $\boxed{\xi_5 = 1.5}$.

## Problem 2 — The box, derived

Primal: $\min_{w,b,\xi}\ \frac12\lVert w\rVert^2 + C\sum_i\xi_i$ s.t. $1 - \xi_i - y^{(i)}(w^Tx^{(i)} + b) \le 0$ and $-\xi_i \le 0$.

Lagrangian ($\alpha_i \ge 0$, $\mu_i \ge 0$):

$$L = \frac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^{n}\xi_i + \sum_{i=1}^{n}\alpha_i\big(1 - \xi_i - y^{(i)}(w^T x^{(i)} + b)\big) - \sum_{i=1}^{n}\mu_i\xi_i.$$

**(i) Stationarity.**

$$\frac{\partial L}{\partial w} = w - \sum_{i=1}^{n}\alpha_i y^{(i)}x^{(i)} = 0 \quad\Longrightarrow\quad \boxed{w^\star = \sum_{i=1}^{n}\alpha_i y^{(i)}x^{(i)}},$$

$$\frac{\partial L}{\partial b} = -\sum_{i=1}^{n}\alpha_i y^{(i)} = 0 \quad\Longrightarrow\quad \boxed{\sum_{i=1}^{n}\alpha_i y^{(i)} = 0},$$

$$\frac{\partial L}{\partial\xi_i} = C - \alpha_i - \mu_i = 0 \quad\Longrightarrow\quad \boxed{\alpha_i + \mu_i = C}.$$

**(ii)** Dual feasibility gives $\mu_i \ge 0$, so $\alpha_i = C - \mu_i \le C$; with $\alpha_i \ge 0$: $\boxed{0 \le \alpha_i \le C}$.

**(iii)** Substituting $w^\star$ (the $\xi_i$ terms cancel via $\alpha_i + \mu_i = C$), the dual is

$$\boxed{\max_{\substack{0 \le \alpha_i \le C \\ \sum_i \alpha_i y^{(i)} = 0}}\ \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i,j=1}^{n}\alpha_i\alpha_j\,y^{(i)}y^{(j)}\,{x^{(i)}}^T x^{(j)}}.$$

## Problem 3 — Three $\alpha$ regimes

KKT at the optimum: (a) $\alpha_i^\star\big(1 - \xi_i^\star - m_i\big) = 0$ where $m_i = y^{(i)}(w^{\star T}x^{(i)} + b^\star)$; (b) $\mu_i^\star\xi_i^\star = (C - \alpha_i^\star)\xi_i^\star = 0$; (c) primal feasibility $m_i \ge 1 - \xi_i^\star$, $\xi_i^\star \ge 0$.

**(i)** $m_i > 1 \Rightarrow \alpha_i^\star = 0$. Suppose $\alpha_i^\star > 0$. Then (a) gives $\xi_i^\star = 1 - m_i < 0$, contradicting $\xi_i^\star \ge 0$. Hence $\boxed{\alpha_i^\star = 0}$.

**(ii)** $0 < \alpha_i^\star < C \Rightarrow m_i = 1,\ \xi_i^\star = 0$. Since $\alpha_i^\star < C$, $\mu_i^\star = C - \alpha_i^\star > 0$, so (b) gives $\boxed{\xi_i^\star = 0}$. Since $\alpha_i^\star > 0$, (a) gives $1 - 0 - m_i = 0$, i.e. $\boxed{m_i = 1}$.

**(iii)** $m_i < 1 \Rightarrow \alpha_i^\star = C$. Suppose $\alpha_i^\star < C$. Then $\mu_i^\star > 0$, so (b) gives $\xi_i^\star = 0$; primal feasibility (c) then needs $m_i \ge 1 - 0 = 1$, contradicting $m_i < 1$. Hence $\boxed{\alpha_i^\star = C}$.

## Problem 4 — The $C$ endpoints

**(i)** $C \to \infty$. The objective is $\frac12\lVert w\rVert^2 + C\sum_i\xi_i$. Any candidate with some $\xi_i > 0$ has objective $\to +\infty$ as $C \to \infty$, while (when the data is separable) feasible hard-margin walls with all $\xi_i = 0$ keep a finite objective. The minimizer is therefore forced to $\xi_i = 0$ for all $i$, and the problem becomes $\boxed{\min_{w,b}\ \frac12\lVert w\rVert^2\ \text{s.t.}\ y^{(i)}(w^Tx^{(i)} + b) \ge 1}$ — the hard-margin primal (§32.3).

**(ii)** $C = 0$. The box collapses to $0 \le \alpha_i \le 0$, so every $\alpha_i^\star = 0$. Then $w^\star = \sum_i \alpha_i^\star y^{(i)}x^{(i)} = \boxed{0}$ — with free bribes, the cheapest answer ignores the data entirely (the TA notes' "$C = 0 \Rightarrow w^\star = 0$").

## Problem 5 — Gram matrix and Mercer

**(i)** Dots: $p_1^Tp_1 = 0$, $p_2^Tp_2 = p_3^Tp_3 = 1$, all cross dots $0$. $K_{ij} = (\text{dot} + 1)^2$:

$$\boxed{K = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 4 & 1 \\ 1 & 1 & 4 \end{pmatrix}}.$$

**(ii)** Try $v = (0,1,-1)^T$: $Kv = (0,\ 3,\ -3)^T = 3v$, so $\boxed{\lambda = 3}$. Try $v = (1,t,t)^T$: $Kv = (1+2t,\ 1+5t,\ 1+5t)^T \stackrel{!}{=} \lambda(1,t,t)^T$ gives $1 + 2t = \lambda$ and $1 + 5t = \lambda t$. Substituting: $1 + 5t = t + 2t^2$, i.e. $2t^2 - 4t - 1 = 0$, so $t = 1 \pm \frac{\sqrt{6}}{2}$ and $\lambda = 1 + 2t = 3 \pm \sqrt{6}$. Eigenvalues: $\boxed{3,\ 3 + \sqrt{6} \approx 5.449,\ 3 - \sqrt{6} \approx 0.551}$ (numpy confirms: $0.55051,\ 3,\ 5.44949$).

**(iii)** $K$ is symmetric with all eigenvalues $\ge 0$ for this dataset — Mercer condition (b) of §24.10 — consistent with $\kappa(x,z) = (x^Tz+1)^2$ being a valid kernel (indeed $\kappa(x,z) = \phi(x)^T\phi(z)$ with an explicit $\phi$, §24.10's eg).

## Problem 6 — A two-point kernel dual

**(i)** $K_{11} = ((-1)(-1)+1)^2 = 4$, $K_{22} = 4$, $K_{12} = ((-1)(1)+1)^2 = 0$: $K = \left(\begin{smallmatrix}4 & 0 \\ 0 & 4\end{smallmatrix}\right)$. $\alpha_1 - \alpha_2 = 0$ gives $\alpha_1 = \alpha_2 =: \alpha \ge 0$; $D(\alpha) = 2\alpha - \frac12(4\alpha^2 + 4\alpha^2) = \boxed{2\alpha - 4\alpha^2}$. $D'(\alpha) = 2 - 8\alpha = 0 \Rightarrow \boxed{\alpha^\star = 1/4}$.

**(ii)** $\sum_j\alpha_j^\star y^{(j)}K_{j1} = \frac14\cdot 4 + \frac14(-1)\cdot 0 = 1$; $y^{(1)}(1 + b^\star) = 1 \Rightarrow \boxed{b^\star = 0}$.

**(iii)** $\sum_i\alpha_i^\star y^{(i)}\kappa(x^{(i)},x) = \frac14(1-x)^2 - \frac14(1+x)^2 = \frac14(-4x) = -x$, so $\boxed{\hat y(x) = \operatorname{sign}(-x)}$. Then $\hat y(-0.5) = \boxed{+1}$ (matches $y^{(1)}$) and $\hat y(0.5) = \boxed{-1}$ (matches $y^{(2)}$).

## Problem 7 — True or false

(i) **True.** For any $(w, b)$, $\xi_i = \max\big(1 - y^{(i)}(w^Tx^{(i)} + b),\, 0\big)$ satisfies both constraint sets — e.g. $(w, b) = (0,0)$ is always feasible, unlike the hard-margin primal.

(ii) **False.** $\alpha_i^\star = C$ means margin $\le 1$ (§33.4(iii)) — that includes $0 < \text{margin} \le 1$, i.e. inside the street but correctly classified. Misclassification needs margin $\le 0$.

(iii) **False.** The kernelized dual (§33.6) and the decision function use only kernel evaluations $\kappa(x^{(i)}, x^{(j)})$; $\phi$ never appears (that is the whole point of the trick).

(iv) **False.** Larger $C$ = stricter (§33.3(i), §33.5: at $C = 10$ the outlier pins the wall). Tolerance grows as $C$ *shrinks*.

(v) **True.** $\kappa(x,z) = x^Tz$ makes $K_{ij} = {x^{(i)}}^Tx^{(j)}$ the ordinary Gram matrix, and the kernelized dual becomes §32.4's dual (with the $0 \le \alpha_i \le C$ box for the soft-margin version, without it for hard margin).
