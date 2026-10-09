# Solutions — Chapter 4. Vector spaces and the four fundamental subspaces

**1.** Apply the subspace test (§4.2): closed under addition and scaling, and $\mathbf{0}$ must be in.

(a) $W = \{(x, y) : y = 3x\}$. Addition: $(x_1, 3x_1) + (x_2, 3x_2) = (x_1 + x_2,\ 3(x_1 + x_2)) \in W$. Scaling: $c(x, 3x) = (cx,\ 3(cx)) \in W$. Verdict: subspace.

(b) First quadrant: $(1, 1)$ is in it, but $(-1)(1, 1) = (-1, -1)$ is not. Verdict: not a subspace.

(c) $\{(0, 0, 0)\}$: $\mathbf{0} + \mathbf{0} = \mathbf{0}$, $c\mathbf{0} = \mathbf{0}$. Verdict: subspace (the trivial one).

(d) Line $y = x + 1$: $(0, 0)$ is not on it ($0 \ne 0 + 1$). Verdict: not a subspace.

**2.** $\operatorname{span}\{(1, 1)\} = \{t(1, 1) : t \in \mathbb{R}\} = \{(t, t) : t \in \mathbb{R}\}$ — the line $y = x$ through the origin in $\mathbb{R}^2$.

$\operatorname{span}\{(1, 1, 1)\} = \{t(1, 1, 1) : t \in \mathbb{R}\} = \{(t, t, t) : t \in \mathbb{R}\}$ — the line through the origin in the direction $(1, 1, 1)$ in $\mathbb{R}^3$.

**3.** Set $a(1, 2) + b(-1, 2) = (0, 0)$. That reads
$$\begin{aligned}
a - b &= 0 \\
2a + 2b &= 0.
\end{aligned}$$
From the first, $b = a$. Substitute into the second: $2a + 2a = 4a = 0 \Rightarrow a = 0$, hence $b = 0$. Only the trivial coefficients work — the set is linearly independent.

**4.** $W = \operatorname{span}\{(1, 1, 0), (1, 0, 0), (2, 1, 0)\}$. Spot the redundancy:
$$(1, 1, 0) + (1, 0, 0) = (1+1,\ 1+0,\ 0) = (2, 1, 0) \text{ ✓}$$
so the third vector adds nothing. Drop it: $\{(1, 1, 0), (1, 0, 0)\}$ still spans $W$. Check independence: $a(1, 1, 0) + b(1, 0, 0) = (0, 0, 0)$ reads $a + b = 0$ and $a = 0$, so $a = b = 0$ — independent. Hence $\{(1, 1, 0), (1, 0, 0)\}$ is a basis and $\dim(W) = 2$ (a plane in $\mathbb{R}^3$).

**5.** $m = 5$, $n = 7$, $r = 4$.

- $\operatorname{nullity}(A) = n - r = 7 - 4 = 3$ (rank-nullity).
- $\dim C(A^T)$ = row rank = $r = 4$.
- $\dim N(A^T) = m - r = 5 - 4 = 1$.

**6.** $A\mathbf{x} = x_1(\text{col 1}) + \cdots + x_n(\text{col } n)$ — multiplying $A$ by $\mathbf{x}$ is exactly forming the linear combination of the columns with coefficients $x_i$ (§2.10 columns view). So $A\mathbf{x} = \mathbf{0}$ with $\mathbf{x} \ne \mathbf{0}$ is the same thing as "a nontrivial combination of the columns makes $\mathbf{0}$," which is the definition of dependence (§4.4). Therefore: columns independent $\iff$ no nonzero $\mathbf{x}$ gives $A\mathbf{x} = \mathbf{0}$ $\iff$ $N(A) = \{\mathbf{0}\}$.

**7.** Solvable $\iff$ $\mathbf{b} \in C(A)$. The columns are $(1, 3)^T$ and $(2, 6)^T = 2(1, 3)^T$, so
$$C(A) = \operatorname{span}\left\{\begin{pmatrix}1\\3\end{pmatrix}\right\} = \left\{t\begin{pmatrix}1\\3\end{pmatrix} : t \in \mathbb{R}\right\} = \{(t, 3t) : t \in \mathbb{R}\}.$$
So $Ax = \mathbf{b}$ is solvable iff $\mathbf{b} = (b_1, b_2)^T$ satisfies $b_2 = 3b_1$ — $\mathbf{b}$ must lie on the line through $(1, 3)$. Check: with $b_2 = 3b_1$, take $x_1 = b_1,\ x_2 = 0$; then $A\mathbf{x} = b_1(1, 3)^T = (b_1, 3b_1)^T = \mathbf{b}$ ✓. If $b_2 \ne 3b_1$, no recipe of columns can produce $\mathbf{b}$.

**8.** $A = \begin{pmatrix} 1 & 3 & 3 & 2 \\ 2 & 6 & 9 & 7 \\ -1 & -3 & 3 & 4 \end{pmatrix}$ ($3 \times 4$). Eliminate:
$$\begin{aligned}
R_2 \leftarrow R_2 - 2R_1 &: [2-2,\ 6-6,\ 9-6,\ 7-4] = [0,\ 0,\ 3,\ 3], \\
R_3 \leftarrow R_3 + R_1 &: [-1+1,\ -3+3,\ 3+3,\ 4+2] = [0,\ 0,\ 6,\ 6], \\
R_3 \leftarrow R_3 - 2R_2 &: [0,\ 0,\ 6-6,\ 6-6] = [0,\ 0,\ 0,\ 0].
\end{aligned}$$
$$U = \begin{pmatrix} 1 & 3 & 3 & 2 \\ 0 & 0 & 3 & 3 \\ 0 & 0 & 0 & 0 \end{pmatrix}.$$
Pivots in columns 1 and 3: $\operatorname{rank}(A) = 2$, $\operatorname{nullity}(A) = 4 - 2 = 2$.

- $C(A)$: pivot columns of $A$:
$$C(A) = \operatorname{span}\left\{\begin{pmatrix}1\\2\\-1\end{pmatrix}, \begin{pmatrix}3\\9\\3\end{pmatrix}\right\}, \qquad \dim = 2.$$
- $N(A)$: from $U$, $x_1 + 3x_2 + 3x_3 + 2x_4 = 0$ and $3x_3 + 3x_4 = 0$, so $x_3 = -x_4$. Free variables $x_2, x_4$:
  - $x_2 = 1,\ x_4 = 0$: $x_3 = 0$, $x_1 = -3(1) - 3(0) - 2(0) = -3$ → $\mathbf{u} = (-3, 1, 0, 0)^T$.
  - $x_2 = 0,\ x_4 = 1$: $x_3 = -1$, $x_1 = -3(0) - 3(-1) - 2(1) = 1$ → $\mathbf{v} = (1, 0, -1, 1)^T$.

  Verify: $A\mathbf{u} = (-3+3,\ -6+6,\ 3-3)^T = \mathbf{0}$ ✓; $A\mathbf{v} = (1-3+2,\ 2-9+7,\ -1-3+4)^T = \mathbf{0}$ ✓.
  $$N(A) = \operatorname{span}(\mathbf{u}, \mathbf{v}), \qquad \dim = 2.$$
- Row space $C(A^T)$: the nonzero rows of $U$:
$$C(A^T) = \operatorname{span}\{(1, 3, 3, 2),\ (0, 0, 3, 3)\}, \qquad \dim = 2.$$
- $N(A^T)$: $\mathbf{y}^T A = \mathbf{0}$ with $\mathbf{y} = (y_1, y_2, y_3)^T$ reads $y_1 + 2y_2 - y_3 = 0$, $3y_1 + 9y_2 + 3y_3 = 0$, $2y_1 + 7y_2 + 4y_3 = 0$ (the second column equation is a multiple of the first). From the first: $y_1 = y_3 - 2y_2$. Substitute into the third-coefficient equation: $3(y_3 - 2y_2) + 9y_2 + 3y_3 = 6y_3 + 3y_2 = 0 \Rightarrow y_2 = -2y_3$. Then $y_1 = y_3 - 2(-2y_3) = 5y_3$. So $\mathbf{y} = y_3(5, -2, 1)^T$. Check: $5(1, 3, 3, 2) - 2(2, 6, 9, 7) + 1(-1, -3, 3, 4) = (5-4-1,\ 15-12-3,\ 15-18+3,\ 10-14+4) = (0, 0, 0, 0)$ ✓.
$$N(A^T) = \operatorname{span}\{(5, -2, 1)^T\}, \qquad \dim = m - r = 3 - 2 = 1.$$

**9.** (a) True: each pivot column holds exactly one pivot and each nonzero RREF row holds exactly one pivot, so both counts equal the number of pivots, which is the rank. (b) False: $m < n$ means more unknowns than equations, but consistency is not guaranteed — e.g. $A = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$, $\mathbf{b} = (0, 1)^T$: the second row reads $0 = 1$, no solution. (c) True: proved in §4.7 — $N(A)$ passes both halves of the subspace test.

**10.** Invertible $4 \times 4$: $\operatorname{rank}(A) = 4$.

- $\dim C(A) = 4$ (the whole $\mathbb{R}^4$), $\dim N(A) = 4 - 4 = 0$ (just $\{\mathbf{0}\}$).
- $\dim C(A^T) = 4$, $\dim N(A^T) = 4 - 4 = 0$.

Since $C(A) = \mathbb{R}^4$, every $\mathbf{b}$ is on the menu; since $N(A) = \{\mathbf{0}\}$, each recipe is unique. $Ax = \mathbf{b}$ has exactly one solution $\mathbf{x} = A^{-1}\mathbf{b}$ for every $\mathbf{b}$.
