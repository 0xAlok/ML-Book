# Solutions — Chapter 24. PCA

## Problem 1

$D = \{(-1,-1)^T, (0,0)^T, (1,1)^T\}$, $m = 1$.

(i) $\bar x = \frac13[(-1,-1) + (0,0) + (1,1)] = (0,0)^T$.
$$C = \frac{1}{3}\left[\begin{pmatrix} -1 \\ -1 \end{pmatrix}\begin{pmatrix} -1 & -1 \end{pmatrix} + 0 + \begin{pmatrix} 1 \\ 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \end{pmatrix}\right] = \frac{2}{3}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}.$$

(ii) $\operatorname{tr}(C) = 4/3$, $\det(C) = (2/3)^2(1\cdot1 - 1\cdot1) = 0$. Characteristic: $\lambda^2 - \frac{4}{3}\lambda = 0$, so $\lambda_1 = 4/3$, $\lambda_2 = 0$. For $\lambda_1$: $\frac{2}{3}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}u = \frac{4}{3}u \Rightarrow u_1 + u_2$ direction; unit: $u_1 = \frac{1}{\sqrt{2}}(1,1)^T$. For $\lambda_2 = 0$: orthogonal direction, $u_2 = \frac{1}{\sqrt{2}}(1,-1)^T$.

(iii) $\tilde x_i = (x_i^T u_1)u_1$ (centered, so no mean term): $x_1^Tu_1 = -2/\sqrt{2} = -\sqrt{2}$, $\tilde x_1 = -\sqrt{2}\,u_1 = (-1,-1)^T = x_1$; $\tilde x_2 = (0,0)^T = x_2$; $x_3^Tu_1 = \sqrt{2}$, $\tilde x_3 = (1,1)^T = x_3$.

(iv) Direct: every $\|x_i - \tilde x_i\| = 0$, so $J^* = 0$. Eigenvalue formula: $J^* = \lambda_2 = 0$ ✓. Projected variance $= \lambda_1 = 4/3$; check: $\frac13[(\sqrt{2})^2 + 0 + (\sqrt{2})^2] = 4/3$ ✓.

## Problem 2

$u^TCu = 3\cos^2\theta + \sin^2\theta = 3\cos^2\theta + (1 - \cos^2\theta) = 1 + 2\cos^2\theta$. Max at $\cos^2\theta = 1$, i.e. $\theta = 0, \pi$: $u = \pm(1,0)^T = \pm e_1$, value $3$ — the eigenvector for the largest eigenvalue $\lambda = 3$. Min at $\cos^2\theta = 0$, i.e. $\theta = \pi/2, 3\pi/2$: $u = \pm(0,1)^T = \pm e_2$, value $1$ — the eigenvector for the smallest eigenvalue $\lambda = 1$. This is §24.4's claim in miniature: extrema of $u^TCu$ on the unit circle sit on eigenvectors, with values equal to the eigenvalues.

## Problem 3

Let $Cu = \lambda u$ with $u \ne 0$. Then $\lambda\|u\|^2 = u^T(\lambda u) = u^TCu = \frac1n\sum_{i=1}^n\big(u^T(x_i - \bar x)\big)^2 \ge 0$ (a sum of squares). Since $\|u\|^2 > 0$, $\lambda \ge 0$. This is §19.1's covariance-is-PSD argument applied to the sample covariance — the same reason a covariance matrix "can never have a negative eigenvalue."

## Problem 4

(i) $\bar x = \frac14[(1,2)+(2,1)+(3,4)+(4,3)] = (2.5, 2.5)^T$. Centered: $(-1.5,-0.5), (-0.5,-1.5), (0.5,1.5), (1.5,0.5)$. $\sum (x_1\text{-coords})^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5$; same for $x_2$; cross terms: $(-1.5)(-0.5) + (-0.5)(-1.5) + (0.5)(1.5) + (1.5)(0.5) = 0.75+0.75+0.75+0.75 = 3$. So $C = \frac14\begin{pmatrix} 5 & 3 \\ 3 & 5 \end{pmatrix} = \begin{pmatrix} 1.25 & 0.75 \\ 0.75 & 1.25 \end{pmatrix}$.

(ii) $\operatorname{tr} = 2.5$, $\det = 1.25^2 - 0.75^2 = 1.5625 - 0.5625 = 1$. $\lambda = \frac{2.5 \pm \sqrt{6.25-4}}{2} = \frac{2.5\pm1.5}{2}$: $\lambda_1 = 2$, $\lambda_2 = 0.5$. For $\lambda_1$: $\begin{pmatrix} -0.75 & 0.75 \\ 0.75 & -0.75 \end{pmatrix}u = 0 \Rightarrow u_1 = \frac{1}{\sqrt{2}}(1,1)^T$; $u_2 = \frac{1}{\sqrt{2}}(1,-1)^T$.

(iii) Recenter, project, add $\bar x$: $x_1 = (1,2) \to (-1.5,-0.5)$, dot with $u_1$ is $-2/\sqrt{2} = -\sqrt{2}$, $\tilde x_1 = -\sqrt{2}\,u_1 + \bar x = (-1,-1) + (2.5,2.5) = (1.5,1.5)$. Similarly $\tilde x_2 = (1.5,1.5)$, $\tilde x_3 = \tilde x_4 = (3.5,3.5)$.

(iv) $\|x_1 - \tilde x_1\|^2 = (-0.5)^2 + (0.5)^2 = 0.5$; same $0.5$ for all four. $J^* = 0.5 = \lambda_2$ ✓ (the dropped direction's eigenvalue).

(v) Total variance $\operatorname{tr}(C) = 2.5$; first component explains $2/2.5 = 0.8 = 80\%$.

## Problem 5

(i) Each dropped direction is an eigenvector: $Cu_j = \lambda_j u_j$ with $\|u_j\| = 1$, so $u_j^TCu_j = u_j^T(\lambda_j u_j) = \lambda_j$. Hence $J^* = \sum_{j=m+1}^{d}\lambda_j$.

(ii) $J^*$ is a sum of $d - m$ terms $u^TCu$ over orthonormal dropped directions. The lecture's single-direction result says the minimum of $u^TCu$ over unit $u$ is the smallest eigenvalue $\lambda_d$, achieved at $u_d$; removing that direction, the minimum over the remaining orthogonal complement is $\lambda_{d-1}$, and so on. Any other orthonormal choice replaces some $\lambda_j$ by a larger value — e.g. keeping $u_d$ instead of $u_1$ puts $\lambda_1$ into the dropped sum. Formally this is the same greedy argument the lecture uses for the variance-maximization side (§24.4 extended to $m \ge 1$: the top-$m$ eigenvectors maximize the projected variance).

## Problem 6

(i) $\frac1nAA^T(Au) = A\big(\frac1nA^TAu\big) = A(Cu) = A(\lambda u) = \lambda(Au)$. So $Au$ is an eigenvector of $\frac1nAA^T$ with the same eigenvalue $\lambda$.

(ii) $C(A^Tv) = \frac1nA^TA(A^Tv) = A^T\big(\frac1nAA^Tv\big) = A^T(\lambda v) = \lambda(A^Tv)$. Since $\lambda > 0$, $A^Tv \ne 0$ (else $\frac1nAA^Tv = 0 \ne \lambda v$), so it is a genuine eigenvector of $C$.

(iii) When $d \gg n$, $\operatorname{rank}(C) \le n$: all but at most $n$ eigenvalues of the $d \times d$ matrix $C$ are $0$. By (i)–(ii) the nonzero spectrum is recovered exactly from the $n \times n$ matrix $\frac1nAA^T$ — eigendecompose that ($O(n^3)$, not $O(d^3)$) and map eigenvectors back via $u = A^Tv/\|A^Tv\|$.

## Problem 7

$(x^Tx' + 1)^2 = (f_1g_1 + f_2g_2 + 1)^2$. Expanding $(a+b+c)^2 = a^2+b^2+c^2+2ab+2ac+2bc$:
$$= f_1^2g_1^2 + f_2^2g_2^2 + 1 + 2f_1g_1f_2g_2 + 2f_1g_1 + 2f_2g_2.$$
Take $\phi(x) = \big(f_1^2,\ f_2^2,\ 1,\ \sqrt{2}f_1f_2,\ \sqrt{2}f_1,\ \sqrt{2}f_2\big)^T \in \mathbb{R}^6$. Then $\phi(x)^T\phi(x') = f_1^2g_1^2 + f_2^2g_2^2 + 1 + (\sqrt{2}f_1f_2)(\sqrt{2}g_1g_2) + (\sqrt{2}f_1)(\sqrt{2}g_1) + (\sqrt{2}f_2)(\sqrt{2}g_2)$, which is exactly the expansion above. So $(x^Tx'+1)^2$ is a valid kernel — it computes a 6-D dot product in $O(d)$ time without forming $\phi$.

## Problem 8

(i) $(1,1,1)^T$: rows sum to $40-8-32 = 0$, so eigenvalue $0$. $(1,0,-1)^T$: first row $40+32 = 72$, second $-8+8 = 0$, third $-32-40 = -72 = 72\cdot(-1)$ — eigenvalue $72$. $(1,-2,1)^T$: first $40+16-32 = 24$, second $-8-32-8 = -48 = 24\cdot(-2)$, third $-32+16+40 = 24$ — eigenvalue $24$. So the inner matrix has eigenvalues $72, 24, 0$; $K^c$ (divided by $9$) has $8,\ 8/3,\ 0$. Check: $8 + 8/3 + 0 = 32/3 = \operatorname{tr}(K^c) = (40+16+40)/9$ ✓.

(ii) $K^c$'s eigenvalues are $n\lambda_k$ with $n = 3$: $n\lambda_1 = 8 \Rightarrow \lambda_1 = 8/3$; $n\lambda_2 = 8/3 \Rightarrow \lambda_2 = 8/9$.

(iii) Unit eigenvector for $8$: $\beta_1 = \frac{1}{\sqrt{2}}(-1,0,1)^T$. Normalized: $\alpha_1 = \beta_1/\sqrt{8} = (-1/4, 0, 1/4)^T$ (eigenvector signs are arbitrary — $(1/4,0,-1/4)^T$ is equally valid). Check: $\alpha_1^TK^c\alpha_1 = \frac{1}{16}\cdot\frac{1}{9}\cdot(-1,0,1)\cdot(-72,0,72)^T = \frac{144}{144} = 1$ ✓.

(iv) Compressed coordinates $K^c\alpha_1 = (-2, 0, 2)^T$ (the sign-flipped $(2,0,-2)^T$ is equally valid). Variance: $\frac{4+0+4}{3} = 8/3 = \lambda_1$ ✓ — the feature-space projected-variance rule.

## Problem 9

(i) $\tilde\phi(x_i) = \phi(x_i) - \mu$ with $\mu = \frac1n\sum_k\phi(x_k)$. Then
$$K^c_{ij} = (\phi(x_i) - \mu)^T(\phi(x_j) - \mu) = \underbrace{\phi(x_i)^T\phi(x_j)}_{K_{ij}} - \phi(x_i)^T\mu - \mu^T\phi(x_j) + \mu^T\mu.$$

(ii) $\phi(x_i)^T\mu = \frac1n\sum_k\phi(x_i)^T\phi(x_k) = \frac1n\sum_kK_{ik} = \theta_i$; similarly $\mu^T\phi(x_j) = \theta_j$; $\mu^T\mu = \frac{1}{n^2}\sum_{k\ell}\phi(x_k)^T\phi(x_\ell) = \frac{1}{n^2}\sum_{k\ell}K_{k\ell} = P$. So $\boxed{K^c_{ij} = K_{ij} - \theta_i - \theta_j + P}$. This centers the data *in feature space* using only kernel evaluations — $\phi$ itself never appears.

## Problem 10

(i) **True.** $C^T = \frac1n\sum_i\big((x_i-\bar x)(x_i-\bar x)^T\big)^T = C$ term by term.

(ii) **False.** The first principal component is the eigenvector of the *largest* eigenvalue — the direction of maximum projected variance (§24.4); the smallest eigenvalue's eigenvector is the *first direction dropped* (§24.5).

(iii) **True.** $J^*(m) = \sum_{j=m+1}^{d}\lambda_j$; raising $m$ removes a nonnegative term ($\lambda_j \ge 0$ by Problem 3) from the sum.

(iv) **False.** On ring data the eigenvalues are nearly equal — PCA keeps both directions and finds no 1-D structure (§24.10's figure); the circular relation is nonlinear.

(v) **False.** Reconstructing $w_k = \sum_j\phi(x_j)\alpha_{kj}$ needs $\phi$ explicitly, which defeats the kernel trick's purpose; only the compressed coordinates $\sum_j\alpha_{kj}K^c_{ij}$ are computable (§24.10, Step 3).
