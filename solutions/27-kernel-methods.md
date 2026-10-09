# Solutions — Chapter 27: Kernel methods and kernel regression

## Problem 1 — Cubic-kernel $\phi$, by hand

Expand $(xz + 1)^3$ for scalars $x, z$:
$$(xz + 1)^3 = (xz)^3 + 3(xz)^2 + 3(xz) + 1 = x^3z^3 + 3x^2z^2 + 3xz + 1.$$
Take
$$\boxed{\phi(x) = \begin{pmatrix} x^3 \\ \sqrt{3}\,x^2 \\ \sqrt{3}\,x \\ 1 \end{pmatrix}}.$$
Then
$$\phi(x)^T\phi(z) = x^3z^3 + (\sqrt{3}x^2)(\sqrt{3}z^2) + (\sqrt{3}x)(\sqrt{3}z) + 1 = x^3z^3 + 3x^2z^2 + 3xz + 1 = (xz+1)^3 \;\checkmark$$
This is the explicit 4-dimensional feature map behind the Week 5 assignment's `ker = ((xi.T@xj)+1)**3`. Note the $\sqrt{3}$ weights — the middle features must be *scaled* for the dot product to reproduce the kernel exactly (same point as §24.10's $\sqrt{2}$ factors).

## Problem 2 — Guided derivation

**(i)** $J(w) = \frac12(\Phi w - y)^T(\Phi w - y) + \frac{\lambda}{2}w^Tw$. Differentiating (the §23.3 calculation with $\Phi$ in place of $A$):
$$\nabla_w J = \Phi^T(\Phi w - y) + \lambda w.$$
(Entrywise: $\frac{\partial}{\partial w_k}\frac12\sum_i(\sum_j\Phi_{ij}w_j - y_i)^2 = \sum_i\Phi_{ik}(\Phi w - y)_i = [\Phi^T(\Phi w - y)]_k$; the penalty contributes $\lambda w_k$.)

**(ii)** Setting $\nabla_w J = 0$: $\Phi^T(\Phi w - y) + \lambda w = 0$, i.e.
$$\lambda w = \Phi^T(y - \Phi w) \quad\Rightarrow\quad \boxed{w = \Phi^T\alpha}, \qquad \alpha := \frac{y - \Phi w}{\lambda} \in \mathbb{R}^n.$$
So the stationary point is a linear combination of the rows of $\Phi$ — the transformed training points.

**(iii)** Substitute $w = \Phi^T\alpha$ into the definition of $\alpha$:
$$\lambda\alpha = y - \Phi(\Phi^T\alpha) = y - (\Phi\Phi^T)\alpha = y - K\alpha,$$
with $K_{ij} = \phi(x_i)^T\phi(x_j) = \kappa(x_i, x_j)$. Hence $\boxed{(K + \lambda I)\alpha = y}$.

**(iv)** $K$ is positive semi-definite (Mercer, §24.10): $z^TKz \ge 0$ for all $z$. For $\lambda > 0$ and $z \ne 0$:
$$z^T(K + \lambda I)z = z^TKz + \lambda z^Tz \ge 0 + \lambda\|z\|^2 > 0,$$
so $K + \lambda I$ is positive definite, hence invertible. $\boxed{\hat\alpha = (K + \lambda I)^{-1}y}$ — the same one-line proof as §23.10's homework for $A^TA + \lambda I$.

## Problem 3 — Kernel ridge numeric

**(i)** $\kappa(x, z) = xz + 1$:
$$K = \begin{pmatrix} 0\cdot 0 + 1 & 0\cdot 2 + 1 \\ 2\cdot 0 + 1 & 2\cdot 2 + 1 \end{pmatrix} = \boxed{\begin{pmatrix} 1 & 1 \\ 1 & 5 \end{pmatrix}}.$$

**(ii)** With $\lambda = 1$:
$$K + I = \begin{pmatrix} 2 & 1 \\ 1 & 6 \end{pmatrix}, \qquad \det = 12 - 1 = 11, \qquad (K+I)^{-1} = \frac{1}{11}\begin{pmatrix} 6 & -1 \\ -1 & 2 \end{pmatrix}.$$
$$\hat\alpha = \frac{1}{11}\begin{pmatrix} 6 & -1 \\ -1 & 2 \end{pmatrix}\begin{pmatrix} 2 \\ 4 \end{pmatrix} = \frac{1}{11}\begin{pmatrix} 12 - 4 \\ -2 + 8 \end{pmatrix} = \boxed{\begin{pmatrix} 8/11 \\ 6/11 \end{pmatrix}}.$$

**(iii)** $f(x) = \alpha_1\kappa(x, 0) + \alpha_2\kappa(x, 2) = \frac{8}{11}(x\cdot 0 + 1) + \frac{6}{11}(2x + 1)$:
$$\boxed{f(x) = \frac{12}{11}x + \frac{14}{11}}.$$
Check: $f(0) = 14/11 \approx 1.27$ (vs $y_1 = 2$), $f(2) = (24 + 14)/11 = 38/11 \approx 3.45$ (vs $y_2 = 4$). Both miss — regularization shrinks the fit, exactly as in §27.8. **Note.** The $+1$ in the kernel is what permits a nonzero intercept: without it, $\kappa(x, z) = xz$ gives $f(x) = x(\sum_i\alpha_i x_i)$, always $0$ at $x = 0$.

## Problem 4 — Interpolation at $\lambda = 0$

**(i)** Training predictions are $\hat y = K\hat\alpha$ (row $i$: $\hat y_i = \sum_j K_{ij}\hat\alpha_j = \sum_j\hat\alpha_j\kappa(x_i, x_j) = f(x_i)$). With $\hat\alpha = K^{-1}y$:
$$\hat y = K(K^{-1}y) = y,$$
so $\boxed{\hat y_i = y_i}$ for all $i$ — training error exactly $0$.

**(ii)** The Week 5 lecture warned: "by *memorizing*, we can get zero error on training data." $\lambda = 0$ is that memorizer in kernel form — the model spends all its freedom hitting the training labels and says nothing about test points. The assignment's `pinv(K)@y` is precisely this endpoint (with `pinv` standing in for $K^{-1}$ when $K$ is singular).

## Problem 5 — Polynomial kernel = polynomial regression

**(i)** $\kappa(x, x_i) = (xx_i + 1)^p$ expands by the binomial theorem to $\sum_{k=0}^{p}\binom{p}{k}(xx_i)^k$ — a polynomial in $x$ of degree $\le p$ (coefficients depend on $x_i$).

**(ii)** $f(x) = \sum_i\alpha_i\kappa(x, x_i)$ is a sum of degree-$\le p$ polynomials, hence a polynomial in $x$ of degree $\le p$.

**(iii)** Kernel regression with the degree-$p$ polynomial kernel searches (a subset of) the same degree-$p$ polynomials as §23.8's polynomial regression — but without ever forming the $(1, x, \ldots, x^p)$ features. For large $p$ or multivariate $x$ (where explicit features number $\approx O(d^p)$), the kernel route is the only feasible one. The price: §23.8's explicit $\theta$ is interpretable coefficient-by-coefficient, while kernel regression's $\alpha$ is not.

## Problem 6 — RBF $\sigma$, qualitatively

**(i)** $\sigma \to 0$: $\kappa(x, x_i) = \exp(-\|x - x_i\|^2/(2\sigma^2)) \to 1$ if $x = x_i$ and $\to 0$ otherwise — each bump collapses to a spike at its training point. The prediction becomes a sum of isolated spikes.

**(ii)** $\sigma \to \infty$: $\kappa(x, x_i) \to 1$ for every $x$ — each bump flattens to a constant, and $f(x) \to \sum_i\alpha_i$, a constant function.

**(iii)** The $\sigma \to 0$ regime risks the §27.4 memorizer: each training point gets its own private spike, so the model can hit every training label exactly ($f(x_i) \approx \alpha_i$) while predicting near-zero everywhere else — zero training error, useless generalization. Narrow bumps = high capacity, same disease as §23.9's degree-4 polynomial.

## Problem 7 — True or false

(i) **False.** The algorithm (§27.6) evaluates $\kappa$ only — Step 1 builds $K_{ij} = \kappa(x_i, x_j)$, Step 3 sums $\alpha_i\kappa(x, x_i)$. No $\phi(x)$ is ever formed; that is the kernel trick's whole point.
(ii) **True.** $K$ is PSD and $\lambda > 0$; $z^T(K + \lambda I)z = z^TKz + \lambda\|z\|^2 > 0$ for $z \ne 0$, so $K + \lambda I$ is positive definite ⇒ invertible (the §27.6 Note).
(iii) **False.** Negative weights are ordinary coefficients, not outlier flags: §27.8's $\alpha_2 = -1/4$ simply pulls the fit *down* near $x_2 = 0$.
(iv) **False.** §24.10: the RBF kernel's $\phi$ is infinite-dimensional — no finite vector exists. The kernel computes the inner product anyway.
(v) **False.** $f(x) = \sum_i\alpha_i x^Tx_i = x^T(\sum_i\alpha_i x_i)$ satisfies $f(0) = 0$ for every $\alpha$ — homogeneous linear, no intercept. The $+1$ in $(x^Tx' + 1)^p$ is what smuggles the intercept in (see Problem 3's Note).

## Problem 8 — Cost accounting

**(i)** Step 1 fills an $n \times n$ matrix: $n^2$ kernel evaluations ($n(n+1)/2$ distinct, by symmetry). Each RBF evaluation computes $\|x - x'\|^2$ — $O(d)$ arithmetic.

**(ii)** Solving the dense $n \times n$ system $(K + \lambda I)\alpha = y$: $\boxed{O(n^3)}$ — the same bill as §24.9's $n \times n$ eigendecomposition. The kernel trick killed the $D$-dependence (even infinite $D$ is fine) but left the $n$-dependence.

**(iii)** One prediction $f(x) = \sum_{i=1}^{n}\alpha_i\kappa(x, x_i)$ needs $n$ kernel evaluations: $\boxed{O(nd)}$ for the RBF kernel — linear in the training-set size, the price of the "sum of similarities" form.
