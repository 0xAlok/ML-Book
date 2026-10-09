# Solutions — Chapter 23: Linear and polynomial regression

**1.** (i) Rows $(1, x^i)$:
$$A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 4 \end{pmatrix}, \qquad Y = \begin{pmatrix} 3 \\ 5 \\ 9 \end{pmatrix}.$$
(ii) $A\theta = \begin{pmatrix} 1+2(1) \\ 1+2(2) \\ 1+2(4) \end{pmatrix} = \begin{pmatrix} 3 \\ 5 \\ 9 \end{pmatrix} = Y$, so $A\theta - Y = (0,0,0)^T$.
(iii) $L(\theta) = \tfrac12\|0\|^2 = \boxed{0}$. The line $f(x) = 2x + 1$ passes exactly through all three points — the consistent case (§5.8): no least squares needed, the loss is already at its floor.

**2.** Differentiate term by term:
$$\frac{\partial L}{\partial w} = \sum_{i=1}^{n}(wx^i + b - y^i)\cdot x^i, \qquad \frac{\partial L}{\partial b} = \sum_{i=1}^{n}(wx^i + b - y^i)\cdot 1.$$
Set both to zero:
$$\sum_i x^i(wx^i + b) = \sum_i x^iy^i, \qquad \sum_i (wx^i + b) = \sum_i y^i.$$
With $A = \begin{pmatrix} 1 & x^1 \\ \vdots & \vdots \\ 1 & x^n \end{pmatrix}$ and $\theta = (b, w)^T$, these are exactly
$$A^TA\theta = A^TY: \quad \begin{pmatrix} n & \sum x^i \\ \sum x^i & \sum (x^i)^2 \end{pmatrix}\!\begin{pmatrix} b \\ w \end{pmatrix} = \begin{pmatrix} \sum y^i \\ \sum x^iy^i \end{pmatrix},$$
the $2\times 2$ normal equations. (Check: $(A^TA)_{11} = n$, $(A^TA)_{12} = \sum x^i$, $(A^TA)_{22} = \sum (x^i)^2$; $(A^TY)_1 = \sum y^i$, $(A^TY)_2 = \sum x^iy^i$ ✓.)

**3.** Assume $A^TAx = 0$. Multiply on the left by $x^T$:
$$0 = x^T(A^TAx) = (x^TA^T)(Ax) = (Ax)^T(Ax) = \|Ax\|^2.$$
A vector of length $0$ is the zero vector, so $Ax = 0$. Hence every $x$ killed by $A^TA$ is killed by $A$: $N(A^TA) \subseteq N(A)$. (The reverse inclusion is immediate: $Ax = 0 \Rightarrow A^TAx = A^T0 = 0$.) The two null spaces are equal, so by rank–nullity (§4.8) $A$ and $A^TA$ have the same rank.

**4.** $$A = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 \\ 2 \\ 3 \end{pmatrix}, \quad A^TA = \begin{pmatrix} 3 & 3 \\ 3 & 5 \end{pmatrix}, \quad A^TY = \begin{pmatrix} 5 \\ 8 \end{pmatrix}.$$
$\det(A^TA) = 15 - 9 = 6 \ne 0$ (independent columns — §23.5):
$$\hat\theta = \frac{1}{6}\begin{pmatrix} 5 & -3 \\ -3 & 3 \end{pmatrix}\!\begin{pmatrix} 5 \\ 8 \end{pmatrix} = \frac{1}{6}\begin{pmatrix} 25-24 \\ -15+24 \end{pmatrix} = \begin{pmatrix} 1/6 \\ 3/2 \end{pmatrix}.$$
$$\boxed{f(x) = \tfrac32x + \tfrac16}.$$
Predictions: $\tfrac16, \tfrac{10}{6}, \tfrac{19}{6}$; residuals (pred − $y$): $\tfrac16, -\tfrac13, \tfrac16$. Minimum squared error:
$$\|A\hat\theta - Y\|^2 = \frac{1}{36} + \frac{4}{36} + \frac{1}{36} = \frac{6}{36} = \boxed{\frac16}.$$
(And $Y \notin C(A)$ — no line hits all three points — so this is genuinely the least-squares compromise, §5.8.)

**5.** (i) From §23.3, $\nabla L = A^T(A\theta - Y)$, which is linear in $\theta$; differentiating once more:
$$\nabla^2 L = A^TA.$$
(ii) For any $z$: $z^T(A^TA)z = (Az)^T(Az) = \|Az\|^2 \ge 0$ — positive semidefinite, so $L$ is convex (§12.7's second-order test).
(iii) A stationary point of a convex function is a global minimum (§12.9): the $\hat\theta$ solving the normal equations minimizes $L$ over all of $\mathbb{R}^d$.

**6.** (i) $$\tilde A = \begin{pmatrix} 1 & 0 & 0 \\ 1 & 1 & 1 \\ 1 & 2 & 4 \end{pmatrix}, \qquad Y = \begin{pmatrix} 2 \\ 1 \\ 2 \end{pmatrix}.$$
(ii) $\tilde A^T\tilde A = \begin{pmatrix} 3&3&5 \\ 3&5&9 \\ 5&9&17 \end{pmatrix}$, $\tilde A^TY = \begin{pmatrix} 5 \\ 5 \\ 9 \end{pmatrix}$ (sums: $2+1+2 = 5$; $0+1+4 = 5$; $0+1+8 = 9$):
$$\begin{cases} 3\theta_0+3\theta_1+5\theta_2 = 5 \\ 3\theta_0+5\theta_1+9\theta_2 = 5 \\ 5\theta_0+9\theta_1+17\theta_2 = 9 \end{cases}.$$
(iii) Try $\theta = (2,-2,1)^T$: $6-6+5 = 5$ ✓; $6-10+9 = 5$ ✓; $10-18+17 = 9$ ✓. $\tilde A$ has distinct $x$'s (Vandermonde — full rank), so the solution is unique: $\boxed{\hat y = 2 - 2x + x^2}$. Check on the data: $2-0+0 = 2$ ✓; $2-2+1 = 1$ ✓; $2-4+4 = 2$ ✓.
(iv) Three parameters, three points, full-rank design matrix — the quadratic interpolates exactly (eg 3's lesson): $m = n-1$ can always thread every point, so the residual is $0$. (Whether that's wise is §23.9's question.)

**7.** (i) **True**: $(A^TA)^T = A^T(A^T)^T = A^TA$. (ii) **True**: full column rank ⇒ $A^TA$ invertible (§23.5) ⇒ exactly one $\hat\theta$. (iii) **True**: the degree-$m$ family contains the degree-$(m-1)$ family (pad extra $\theta_j = 0$), and minimizing over a bigger set can't give a bigger minimum. (iv) **False**: eg 4 — test MSE rose $0.0030 \to 0.0191$ from $m = 2$ to $m = 4$. That's overfitting.

**8.** (i) $A^TA = \begin{pmatrix} 3&6 \\ 6&14 \end{pmatrix}$, $\det = 42-36 = 6 \ne 0$ — **exists**. ML reading: intercept and $x$ are independent features; both coefficients are pinned down.
(ii) $A^TA = \begin{pmatrix} 3&9&9 \\ 9&29&29 \\ 9&29&29 \end{pmatrix}$ — columns 2 and 3 identical, rank $\le 2 < 3$, singular — **no inverse**. ML reading: the "feature" was duplicated; the model can't decide how to split credit between $\theta_2$ and $\theta_3$ (any shift $\theta_2 \to \theta_2 + t$, $\theta_3 \to \theta_3 - t$ gives identical predictions).
(iii) $A^TA = \begin{pmatrix} 14&0 \\ 0&0 \end{pmatrix}$, $\det = 0$ — **no inverse**. ML reading: the always-$0$ feature contributes nothing to any prediction, so its coefficient is completely free.

**9.** (i) **$m = 2$** — decided on the **validation** MSE ($0.01$, the smallest column-3 entry). The training column was only for fitting; the degree is model selection (§23.9). (ii) **Overfitting**: train MSE $0.00$ (memorized) but validation MSE $0.09$ (worst of the four). (iii) The **test** split's number — on data untouched by both fitting and selection. The validation number is already optimistic (it chose the winner); §22.10's three-way split exists exactly so the reported number is honest.

**10.** (i) Each $y^i \mid x^i \sim N(\theta^Tx^i, \sigma^2)$, independent:
$$L(\theta) = \prod_{i=1}^{n}\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(y^i - \theta^Tx^i)^2}{2\sigma^2}\right).$$
(ii) $$\ell(\theta) = \log L(\theta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y^i - \theta^Tx^i)^2.$$
(iii) The first term doesn't involve $\theta$, and $1/(2\sigma^2) > 0$, so
$$\arg\max_\theta \ell(\theta) = \arg\min_\theta \sum_{i=1}^{n}(y^i - \theta^Tx^i)^2,$$
the least-squares objective (§23.1). Maximum likelihood under Gaussian noise **is** least squares — §20.8's result, §23.6's meaning.
