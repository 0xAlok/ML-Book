# Solutions — Chapter 19. Random vectors and the multivariate normal

## Problem 1

Marginals of $X$: $P(X=0) = 0.1+0.2 = 0.3$, $P(X=1) = 0.3+0.4 = 0.7$; of $Y$: $P(Y=0) = 0.1+0.3 = 0.4$, $P(Y=1) = 0.2+0.4 = 0.6$.

- $\boldsymbol{\mu} = (E[X], E[Y])^T = (0.7,\ 0.6)^T$.
- $E[X^2] = 0^2(0.3) + 1^2(0.7) = 0.7$, so $\operatorname{Var}(X) = 0.7 - 0.49 = 0.21$. $E[Y^2] = 0.6$, $\operatorname{Var}(Y) = 0.6 - 0.36 = 0.24$.
- $E[XY] = \sum x y\,P(x,y) = 1\cdot1\cdot0.4 = 0.4$ (all other terms have $x = 0$ or $y = 0$). $\operatorname{Cov}(X,Y) = 0.4 - 0.7\cdot0.6 = -0.02$.

$$\boxed{\boldsymbol{\Sigma} = \begin{pmatrix} 0.21 & -0.02 \\ -0.02 & 0.24 \end{pmatrix}}.$$

Symmetry: off-diagonals equal ✓. PSD: diagonal entries $> 0$ and $\det = 0.21\cdot0.24 - (-0.02)^2 = 0.0504 - 0.0004 = 0.05 > 0$, so by §7.9's $2\times2$ test the matrix is positive definite (hence PSD) ✓. Correlation: $\rho = -0.02/\sqrt{0.21\cdot0.24} = -0.02/\sqrt{0.0504} \approx -0.02/0.2245 \approx \boxed{-0.089}$.

## Problem 2

$$\begin{aligned}
\boldsymbol{\Sigma} &= E\big[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^T\big] \\
&= E\big[\mathbf{X}\mathbf{X}^T - \mathbf{X}\boldsymbol{\mu}^T - \boldsymbol{\mu}\mathbf{X}^T + \boldsymbol{\mu}\boldsymbol{\mu}^T\big] \\
&= E[\mathbf{X}\mathbf{X}^T] - E[\mathbf{X}]\boldsymbol{\mu}^T - \boldsymbol{\mu}E[\mathbf{X}]^T + \boldsymbol{\mu}\boldsymbol{\mu}^T \qquad\text{(linearity, $\boldsymbol{\mu}$ constant)} \\
&= E[\mathbf{X}\mathbf{X}^T] - \boldsymbol{\mu}\boldsymbol{\mu}^T - \boldsymbol{\mu}\boldsymbol{\mu}^T + \boldsymbol{\mu}\boldsymbol{\mu}^T \\
&= \boxed{E[\mathbf{X}\mathbf{X}^T] - \boldsymbol{\mu}\boldsymbol{\mu}^T}.
\end{aligned}$$

This is the matrix version of $\operatorname{Var}(X) = E[X^2] - (E[X])^2$ (§16.6).

## Problem 3

$E[\mathbf{Y}] = \mathbf{A}\boldsymbol{\mu} + \mathbf{b} = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 \\ 2 \end{pmatrix} + \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 3 \\ -1 \end{pmatrix} + \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \boxed{\begin{pmatrix} 3 \\ 0 \end{pmatrix}}$.

$\operatorname{Cov}(\mathbf{Y}) = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T$. First $\boldsymbol{\Sigma}\mathbf{A}^T = \begin{pmatrix} 4 & 1 \\ 1 & 2 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \begin{pmatrix} 5 & 3 \\ 3 & -1 \end{pmatrix}$; then
$$\mathbf{A}(\boldsymbol{\Sigma}\mathbf{A}^T) = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 5 & 3 \\ 3 & -1 \end{pmatrix} = \boxed{\begin{pmatrix} 8 & 2 \\ 2 & 4 \end{pmatrix}}.$$

Checks: $\operatorname{Var}(Y_1) = \operatorname{Var}(X_1+X_2) = 4 + 2 + 2(1) = 8$ ✓; symmetric ✓; $\det = 32-4 = 28 > 0$, PSD ✓. Note $Y_1 = X_1 + X_2 + 0$ and $Y_2 = X_1 - X_2 + 1$ — the shift $1$ appears only in the mean.

## Problem 4

$\boldsymbol{\Sigma} = \begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix} = \begin{pmatrix} 4 & 1.2 \\ 1.2 & 1 \end{pmatrix}$.

- $\det\boldsymbol{\Sigma} = 4\cdot1 - 1.44 = 2.56$; $\sqrt{\det\boldsymbol{\Sigma}} = 1.6$. Normalizer: $\boxed{(2\pi)\cdot1.6 = 3.2\pi \approx 10.053}$.
- $\boldsymbol{\Sigma}^{-1} = \frac{1}{2.56}\begin{pmatrix} 1 & -1.2 \\ -1.2 & 4 \end{pmatrix} = \begin{pmatrix} 0.390625 & -0.46875 \\ -0.46875 & 1.5625 \end{pmatrix}$.

$$\boxed{f(x_1,x_2) = \frac{1}{3.2\pi}\exp\!\left(-\frac{1}{2}\Big[0.390625\,(x_1-1)^2 - 0.9375\,(x_1-1)(x_2+1) + 1.5625\,(x_2+1)^2\Big]\right)}$$
(the cross term: $2\cdot(-0.46875)(x_1-1)(x_2+1) = -0.9375\,(x_1-1)(x_2+1)$).

## Problem 5

(a) By §19.5: $X_2 \mid X_1 = x_1 \sim N(\rho x_1,\ 1-\rho^2)$. With $\rho = 0.5$, $x_1 = 2$:
$$\boxed{X_2 \mid X_1 = 2 \sim N(1,\ 0.75)}.$$

(b) Standardize: $P(X_2 > 1.5 \mid X_1 = 2) = P\!\left(Z > \frac{1.5-1}{\sqrt{0.75}}\right) = P(Z > 0.5774) = 1 - \Phi(0.5774)$. $\Phi(0.5774) \approx 0.71815$, so the answer is $\boxed{\approx 0.282}$.

Compare with the marginal: $P(X_2 > 1.5) = 1 - \Phi(1.5) \approx 0.0668$ — conditioning on the large $X_1$ more than quadruples the probability.

## Problem 6

$Z = \mathbf{c}^T\mathbf{Y}$ with $\mathbf{c} = (1,\ 2,\ -1)^T$. By §19.7(i), $Z \sim N(\mathbf{c}^T\boldsymbol{\mu},\ \mathbf{c}^T\boldsymbol{\Sigma}\mathbf{c})$.

- Mean: $1(2) + 2(-1) + (-1)(5) = 2 - 2 - 5 = \boxed{-5}$.
- $\boldsymbol{\Sigma}\mathbf{c} = \begin{pmatrix} 9 & 2 & -1 \\ 2 & 4 & 1 \\ -1 & 1 & 6 \end{pmatrix}\begin{pmatrix} 1 \\ 2 \\ -1 \end{pmatrix} = \begin{pmatrix} 9+4+1 \\ 2+8-1 \\ -1+2-6 \end{pmatrix} = \begin{pmatrix} 14 \\ 9 \\ -5 \end{pmatrix}$; variance $= \mathbf{c}^T(\boldsymbol{\Sigma}\mathbf{c}) = 1(14) + 2(9) + (-1)(-5) = 14 + 18 + 5 = \boxed{37}$.

$$\boxed{Z \sim N(-5,\ 37)}.$$
(Sanity: $\boldsymbol{\Sigma}$ is PD — leading minors $9 > 0$, $36-4 = 32 > 0$, $\det = 175 > 0$ — so it is a valid covariance.)

## Problem 7

Eigendecomposition of $\boldsymbol{\Sigma} = \begin{pmatrix} 5 & 3 \\ 3 & 5 \end{pmatrix}$: characteristic equation $(5-\lambda)^2 - 9 = 0$ gives $\boxed{\lambda_1 = 8,\ \lambda_2 = 2}$. Unit eigenvectors: $\mathbf{q}_1 = (1,1)^T/\sqrt{2}$ (for $\lambda = 8$: $(5-8)x + 3y = 0 \Rightarrow y = x$), $\mathbf{q}_2 = (1,-1)^T/\sqrt{2}$.

$$\boldsymbol{\Sigma}^{-1/2} = \frac{1}{\sqrt{8}}\mathbf{q}_1\mathbf{q}_1^T + \frac{1}{\sqrt{2}}\mathbf{q}_2\mathbf{q}_2^T = \frac{1}{2\sqrt{2}}\cdot\frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} + \frac{1}{\sqrt{2}}\cdot\frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}.$$

Combining over the common denominator $4\sqrt{2}$:
$$\boxed{\boldsymbol{\Sigma}^{-1/2} = \frac{1}{4\sqrt{2}}\begin{pmatrix} 3 & -1 \\ -1 & 3 \end{pmatrix} \approx \begin{pmatrix} 0.5303 & -0.1768 \\ -0.1768 & 0.5303 \end{pmatrix}}.$$

Verification: $\operatorname{Cov}(\mathbf{W}) = \boldsymbol{\Sigma}^{-1/2}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{-1/2}$. Since $\boldsymbol{\Sigma}^{-1/2}$ is symmetric and $\boldsymbol{\Sigma}^{-1/2}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{-1/2} = \boldsymbol{\Sigma}^{-1/2}(\boldsymbol{\Sigma}^{1/2}\boldsymbol{\Sigma}^{1/2})\boldsymbol{\Sigma}^{-1/2} = \mathbf{I}$ (both built from the same $\mathbf{Q}$), $\boxed{\operatorname{Cov}(\mathbf{W}) = \mathbf{I}}$ — and by §19.7, $\mathbf{W} \sim \mathcal{N}_2(\mathbf{0}, \mathbf{I})$. Numerically, $\boldsymbol{\Sigma}^{-1/2}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{-1/2} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ to machine precision ✓.

## Problem 8

(a) For a normal vector, $X_i \perp X_j \iff \Sigma_{ij} = 0$ (§19.7(iii)). The only zero off-diagonal is $\Sigma_{12} = \Sigma_{21} = 0$, so $\boxed{X_1 \perp X_2}$; the other two pairs are dependent (nonzero covariance).

(b) $\rho_{13} = \Sigma_{13}/\sqrt{\Sigma_{11}\Sigma_{33}} = 1.5/\sqrt{4\cdot16} = 1.5/8 = \boxed{0.1875}$. $\rho_{23} = \Sigma_{23}/\sqrt{\Sigma_{22}\Sigma_{33}} = -2/\sqrt{9\cdot16} = -2/12 = \boxed{-1/6 \approx -0.1667}$.

(Validity: $\det\boldsymbol{\Sigma} = 539.75 > 0$ with positive leading minors, so $\boldsymbol{\Sigma}$ is PD ✓.)

## Problem 9

Partition with $\mathbf{X}_{(1)} = (X_1, X_2)$, $\mathbf{X}_{(2)} = X_3$: $\boldsymbol{\mu}_1 = (1, 0)^T$, $\mu_2 = 2$, $\boldsymbol{\Sigma}_{11} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$, $\boldsymbol{\Sigma}_{21} = (0.5,\ 0.5)$, $\Sigma_{22} = 1$.

- $\boldsymbol{\Sigma}_{11}^{-1} = \frac{1}{3}\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$.
- Regression coefficients: $\boldsymbol{\Sigma}_{21}\boldsymbol{\Sigma}_{11}^{-1} = (0.5,\ 0.5)\cdot\frac{1}{3}\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} = \frac{1}{3}(0.5,\ 0.5) = (1/6,\ 1/6)$.
- Conditional mean: $2 + \frac{1}{6}(3 - 1) + \frac{1}{6}(-1 - 0) = 2 + \frac{1}{3} - \frac{1}{6} = 2 + \frac{1}{6} = \boxed{13/6 \approx 2.1667}$.
- Conditional variance (Schur complement): $1 - (0.5,\ 0.5)\cdot(1/6,\ 1/6)^T = 1 - \frac{1}{6} = \boxed{5/6 \approx 0.8333}$.

$$\boxed{X_3 \mid X_1 = 3,\ X_2 = -1 \sim N(13/6,\ 5/6)}.$$

## Problem 10

$\mathbf{d} = \mathbf{x} - \boldsymbol{\mu} = (3-1,\ 3-2)^T = (2,\ 1)^T$. $\det\boldsymbol{\Sigma} = 4\cdot1 - 1 = 3$, so $\boldsymbol{\Sigma}^{-1} = \frac{1}{3}\begin{pmatrix} 1 & -1 \\ -1 & 4 \end{pmatrix}$.

Squared Mahalanobis distance:
$$d_{\boldsymbol{\Sigma}}^2 = \mathbf{d}^T\boldsymbol{\Sigma}^{-1}\mathbf{d} = \frac{1}{3}(2,\ 1)\begin{pmatrix} 1 & -1 \\ -1 & 4 \end{pmatrix}\begin{pmatrix} 2 \\ 1 \end{pmatrix} = \frac{1}{3}(2,\ 1)\begin{pmatrix} 1 \\ 2 \end{pmatrix} = \frac{1}{3}(2 + 2) = \boxed{\frac{4}{3}}.$$

Density: $f_{\mathbf{X}}(3,3) = \frac{1}{2\pi\sqrt{3}}\exp(-d_{\boldsymbol{\Sigma}}^2/2) = \frac{1}{2\pi\sqrt{3}}e^{-2/3} \approx \frac{0.51342}{10.88280} \approx \boxed{0.0472}$.

(Reading: Euclidean distance from $\boldsymbol{\mu}$ is $\sqrt{5} \approx 2.236$, but the Mahalanobis distance is $\sqrt{4/3} \approx 1.155$ — the point lies mostly along the high-variance direction, so the density judges it "closer" than raw distance suggests.)
