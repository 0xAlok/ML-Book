# Solutions — Chapter 6. Eigenvalues, eigenvectors, diagonalization

**1.** $A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$.

(a) $A - \lambda I = \begin{bmatrix} 4 - \lambda & 1 \\ 2 & 3 - \lambda \end{bmatrix}$, $\det = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2)$. Eigenvalues: $\lambda_1 = 5$, $\lambda_2 = 2$.

(b) $\lambda_1 = 5$: $A - 5I = \begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix}$; $-x_1 + x_2 = 0$, so $\mathbf{x}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$. $\lambda_2 = 2$: $A - 2I = \begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix}$; $2x_1 + x_2 = 0$, so $\mathbf{x}_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$.

(c) $A\mathbf{x}_1 = \begin{bmatrix} 4+1 \\ 2+3 \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \end{bmatrix} = 5\mathbf{x}_1$ ✓. $A\mathbf{x}_2 = \begin{bmatrix} 4-2 \\ 2-6 \end{bmatrix} = \begin{bmatrix} 2 \\ -4 \end{bmatrix} = 2\mathbf{x}_2$ ✓.

(d) $5 + 2 = 7 = 4 + 3 = \operatorname{tr}(A)$ ✓; $5 \cdot 2 = 10 = 4\cdot 3 - 1\cdot 2 = \det(A)$ ✓.

**2.** $A = \begin{bmatrix} 2 & 2 \\ 1 & 3 \end{bmatrix}$.

(a) True: $A\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 4 \\ 4 \end{bmatrix} = 4\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ ✓.
(b) False: $A\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 0 \\ -2 \end{bmatrix}$, but $2\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 2 \\ -2 \end{bmatrix}$ ✗.
(c) True: $A\begin{bmatrix} 2 \\ -1 \end{bmatrix} = \begin{bmatrix} 4-2 \\ 2-3 \end{bmatrix} = \begin{bmatrix} 2 \\ -1 \end{bmatrix} = 1\cdot\begin{bmatrix} 2 \\ -1 \end{bmatrix}$ ✓.
(Indeed the characteristic polynomial is $\lambda^2 - 5\lambda + 4 = (\lambda-4)(\lambda-1)$, so $2$ is not even an eigenvalue.)

**3.** $P\mathbf{x} = \lambda \mathbf{x}$, $\mathbf{x} \ne \mathbf{0}$, and $P^2 = P$. Apply $P$ to both sides of $P\mathbf{x} = \lambda \mathbf{x}$: $P^2\mathbf{x} = \lambda P\mathbf{x} = \lambda^2 \mathbf{x}$. But $P^2 = P$, so $P^2\mathbf{x} = P\mathbf{x} = \lambda \mathbf{x}$. Hence $\lambda^2 \mathbf{x} = \lambda \mathbf{x}$, i.e. $(\lambda^2 - \lambda)\mathbf{x} = \mathbf{0}$. Since $\mathbf{x} \ne \mathbf{0}$, $\lambda^2 - \lambda = 0$, so $\lambda = 0$ or $\lambda = 1$. (This is exactly the plane-projection example of §6.3: $\lambda = 1$ in the plane, $\lambda = 0$ perpendicular to it.)

**4.** (a) By the shift rule (§6.5): $A = B + 3I$ keeps $B$'s eigenvectors and adds $3$ to each eigenvalue. Eigenpairs of $A$: $(1 + 3,\ [1,1]^T) = (4,\ [1,1]^T)$ and $(-1 + 3,\ [1,-1]^T) = (2,\ [1,-1]^T)$ — matching the §6.4 computation.
(b) The rule used $(B + cI)\mathbf{x} = B\mathbf{x} + c\mathbf{x}$, which needs $\mathbf{x}$ to be an eigenvector of **both** $B$ and $cI$ — true since every vector is an eigenvector of $I$. A general $C$ does not share $B$'s eigenvectors, so $(B + C)\mathbf{x} \ne (\mu + \nu)\mathbf{x}$ in general (§6.5 warning).

**5.** $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$: $\det(A - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 3)(\lambda - 1)$; $\lambda_1 = 3$, $\lambda_2 = 1$.

$\lambda_1 = 3$: $A - 3I = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix}$, $\mathbf{x}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$. $\lambda_2 = 1$: $A - I = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$, $\mathbf{x}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$.

$S = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$, $\Lambda = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$. $\det S = -2$, $S^{-1} = -\frac{1}{2}\begin{bmatrix} -1 & -1 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{bmatrix}$.
$$AS = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} = \begin{bmatrix} 3 & 1 \\ 3 & -1 \end{bmatrix},$$
$$S^{-1}AS = \begin{bmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{bmatrix}\begin{bmatrix} 3 & 1 \\ 3 & -1 \end{bmatrix} = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix} = \Lambda \;\checkmark.$$

**6.** $\Lambda^3 = \operatorname{diag}(27, 1)$. $S\Lambda^3 = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} 27 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 27 & 1 \\ 27 & -1 \end{bmatrix}$.
$$A^3 = S\Lambda^3 S^{-1} = \begin{bmatrix} 27 & 1 \\ 27 & -1 \end{bmatrix}\begin{bmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{bmatrix} = \begin{bmatrix} 14 & 13 \\ 13 & 14 \end{bmatrix}.$$
Direct: $A^2 = \begin{bmatrix} 5 & 4 \\ 4 & 5 \end{bmatrix}$, $A^3 = A^2A = \begin{bmatrix} 5 & 4 \\ 4 & 5 \end{bmatrix}\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 14 & 13 \\ 13 & 14 \end{bmatrix}$ ✓.

**7.** $A = \begin{bmatrix} 3 & 1 \\ 0 & 3 \end{bmatrix}$.

(a) $\det(A - \lambda I) = (3-\lambda)^2$: $\lambda = 3$ (double root).
(b) $A - 3I = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$; $(A - 3I)\mathbf{x} = \mathbf{0}$ gives $x_2 = 0$, $x_1$ free. $N(A - 3I) = \operatorname{span}\{[1,0]^T\}$, dimension $1$.
(c) A $2 \times 2$ matrix needs $2$ independent eigenvectors to diagonalize; there is only one independent direction here. $A$ is defective — not diagonalizable.

**8.** $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ is symmetric; from Problem 5, eigenvalues $3, 1$ with (orthogonal) eigenvectors $[1,1]^T$, $[1,-1]^T$.

Unit eigenvectors: $\mathbf{q}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$, $\mathbf{q}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}$; $Q = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$.
$$Q^TQ = \frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} = \frac{1}{2}\begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} = I \;\checkmark.$$
With $\Lambda = \operatorname{diag}(3, 1)$: $\Lambda Q^T = \frac{1}{\sqrt{2}}\begin{bmatrix} 3 & 3 \\ 1 & -1 \end{bmatrix}$,
$$Q\Lambda Q^T = \frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} 3 & 3 \\ 1 & -1 \end{bmatrix} = \frac{1}{2}\begin{bmatrix} 4 & 2 \\ 2 & 4 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} = A \;\checkmark.$$

**9.** (a) $\det(R - \lambda I) = \begin{vmatrix} -\lambda & -1 \\ 1 & -\lambda \end{vmatrix} = \lambda^2 + 1 = 0$: $\lambda_1 = i$, $\lambda_2 = -i$.
(b) $i + (-i) = 0 = \operatorname{tr}(R)$ ✓; $i \cdot (-i) = -i^2 = 1 = \det(R)$ ✓. (The trace/determinant relations hold over $\mathbb{C}$ too.)
(c) No contradiction: the spectral theorem applies to **symmetric** matrices, and $R^T = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} = -R \ne R$. A non-symmetric real matrix may have complex eigenvalues — exactly what §6.3 warned about.

**10.** (a) $F_k = \dfrac{\lambda_1^k - \lambda_2^k}{\sqrt{5}}$ with $\lambda_1 = \frac{1+\sqrt{5}}{2}$, $\lambda_2 = \frac{1-\sqrt{5}}{2}$.
(b) Dropping the $\lambda_2$ term: $F_{20} \approx \lambda_1^{20}/\sqrt{5} \approx 6765.0000296$. The neglected term is $|\lambda_2|^{20}/\sqrt{5} \approx 2.96 \times 10^{-5}$ — the estimate is correct to $4$ decimal places, and rounding gives the exact integer $F_{20} = 6765$.
(c) Because $|\lambda_2| \approx 0.618 < 1$: a number smaller than $1$ raised to the $20$th power is tiny.
