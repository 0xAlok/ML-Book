# Solutions — Chapter 3: Linear systems, determinants, inverses

Worked solutions to the problem set in `chapters/03-linear-systems-determinants-inverses.md`. Same numbering, full steps.

**1.** The unknowns are $x, y$; the coefficient table and right-hand side are read straight off the equations:

$$A = \begin{pmatrix} 3 & 2 \\ 1 & -1 \end{pmatrix}, \qquad
\mathbf{x} = \begin{pmatrix} x \\ y \end{pmatrix}, \qquad
\mathbf{b} = \begin{pmatrix} 12 \\ 1 \end{pmatrix},$$

so $A\mathbf{x} = \mathbf{b}$ is $\begin{pmatrix} 3 & 2 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 12 \\ 1 \end{pmatrix}$. The augmented matrix glues $\mathbf{b}$ on as a last column:

$$[A \mid \mathbf{b}] = \left[\begin{array}{rr|r} 3 & 2 & 12 \\ 1 & -1 & 1 \end{array}\right].$$

**2.** Augmented matrix:
$$\left[\begin{array}{rrr|r}
1 & 1 & 1 & 5 \\
2 & -1 & 3 & 17 \\
1 & 2 & -1 & -4
\end{array}\right]$$

Step 1 — $R_2 \leftarrow R_2 - 2R_1$: $[2-2,\ -1-2,\ 3-2 \mid 17-10] = [0,\ -3,\ 1 \mid 7]$.

Step 2 — $R_3 \leftarrow R_3 - R_1$: $[1-1,\ 2-1,\ -1-1 \mid -4-5] = [0,\ 1,\ -2 \mid -9]$.

$$\left[\begin{array}{rrr|r}
1 & 1 & 1 & 5 \\
0 & -3 & 1 & 7 \\
0 & 1 & -2 & -9
\end{array}\right]$$

Step 3 — $R_2 \leftrightarrow R_3$ (nicer pivot):

$$\left[\begin{array}{rrr|r}
1 & 1 & 1 & 5 \\
0 & 1 & -2 & -9 \\
0 & -3 & 1 & 7
\end{array}\right]$$

Step 4 — $R_3 \leftarrow R_3 + 3R_2$: $[0,\ -3+3,\ 1-6 \mid 7-27] = [0,\ 0,\ -5 \mid -20]$, so $-5z = -20$ and $z = 4$.

Step 5 — back-substitution. Row 2: $y - 2z = -9$, so $y = -9 + 2(4) = -1$. Row 1: $x + y + z = 5$, so $x = 5 - (-1) - 4 = 2$.

Solution: $(x, y, z) = (2, -1, 4)$. Verify: $2 + (-1) + 4 = 5$ ✓; $2(2) - (-1) + 3(4) = 4 + 1 + 12 = 17$ ✓; $2 + 2(-1) - 4 = 2 - 2 - 4 = -4$ ✓.

**3.**
i) Infinitely many. The second equation is $3 \times$ the first ($3x + 6y = 3 \cdot 4 = 12$): one equation in disguise. Solutions: all $(x, y)$ with $x + 2y = 4$, e.g. $(x, y) = (4 - 2t,\ t)$. Check: $(4-2t) + 2t = 4$ ✓; $3(4-2t) + 6t = 12$ ✓.

ii) No solution. $3 \times$ the first equation demands $3x + 6y = 12$, but the second demands $3x + 6y = 10$: $12 \ne 10$, contradiction. (Parallel lines.)

iii) Unique solution. Subtract: $(x + 2y) - (x - y) = 4 - 1$, so $3y = 3$ and $y = 1$; then $x = 1 + y = 2$. Check: $2 + 2(1) = 4$ ✓; $2 - 1 = 1$ ✓. Solution: $(2, 1)$.

**4.**
i) $\det(A) = 3(1) - 1(2) = 1 \ne 0$. By the $2 \times 2$ formula (swap diagonal, negate off-diagonal, divide by $\det$):
$$A^{-1} = \frac{1}{1}\begin{pmatrix} 1 & -1 \\ -2 & 3 \end{pmatrix} = \begin{pmatrix} 1 & -1 \\ -2 & 3 \end{pmatrix}.$$
Verify:
$$AA^{-1} = \begin{pmatrix} 3 & 1 \\ 2 & 1 \end{pmatrix}\begin{pmatrix} 1 & -1 \\ -2 & 3 \end{pmatrix}
= \begin{pmatrix} 3-2 & -3+3 \\ 2-2 & -2+3 \end{pmatrix}
= \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I.$$
✓

ii) $\begin{pmatrix} x \\ y \end{pmatrix} = A^{-1}\mathbf{b} = \begin{pmatrix} 1 & -1 \\ -2 & 3 \end{pmatrix}\begin{pmatrix} 5 \\ 3 \end{pmatrix} = \begin{pmatrix} 5 - 3 \\ -10 + 9 \end{pmatrix} = \begin{pmatrix} 2 \\ -1 \end{pmatrix}$. Check: $3(2) + 1(-1) = 5$ ✓; $2(2) + 1(-1) = 3$ ✓.

**5.** $\det(A) = 2(2) - 4(1) = 4 - 4 = 0$. A matrix has an inverse only if its determinant is nonzero; here it is zero, so $A$ is singular — no inverse exists. (Equivalently: row 1 is $2 \times$ row 2, so the rows carry redundant information.)

**6.**
i) $\det\begin{pmatrix} 5 & -2 \\ 3 & 4 \end{pmatrix} = 5(4) - (-2)(3) = 20 + 6 = 26$.

ii) Expand along row 1 (signs $+$, $-$, $+$):
$$\begin{aligned}
\det &= 1\det\begin{pmatrix} 4 & 5 \\ 0 & 6 \end{pmatrix}
-2\det\begin{pmatrix} 0 & 5 \\ 1 & 6 \end{pmatrix}
+3\det\begin{pmatrix} 0 & 4 \\ 1 & 0 \end{pmatrix} \\
&= 1(24 - 0) - 2(0 - 5) + 3(0 - 4) \\
&= 24 + 10 - 12 = 22.
\end{aligned}$$

iii) Upper triangular: determinant = product of the diagonal $= 3 \cdot (-2) \cdot 4 = -24$.

**7.** $\det(2A) = 40$. Reason: $2A$ scales every row of $A$ by $2$; scaling one row by $t$ scales the determinant by $t$ (property vi), and there are $3$ rows, so $\det(2A) = 2 \cdot 2 \cdot 2 \cdot \det(A) = 8 \cdot 5 = 40$. (In general $\det(tA) = t^n \det(A)$ for $n \times n$ $A$.)

**8.** $A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$, $\mathbf{b} = \begin{pmatrix} 7 \\ 11 \end{pmatrix}$, $\det(A) = 2(3) - 1(1) = 5 \ne 0$.
- $A_1 = \begin{pmatrix} 7 & 1 \\ 11 & 3 \end{pmatrix}$: $\det(A_1) = 7(3) - 1(11) = 21 - 11 = 10$, so $x = 10/5 = 2$.
- $A_2 = \begin{pmatrix} 2 & 7 \\ 1 & 11 \end{pmatrix}$: $\det(A_2) = 2(11) - 7(1) = 22 - 7 = 15$, so $y = 15/5 = 3$.

Check: $2(2) + 3 = 7$ ✓; $2 + 3(3) = 11$ ✓.

**9.** From the first equation $x = -2y$; the second is $3 \times$ the first, hence redundant. Put $y = t$ (free): all solutions are $(x, y) = (-2t,\ t)$, $t \in \mathbb{R}$ — infinitely many. Why infinitely many: $\det\begin{pmatrix} 1 & 2 \\ 3 & 6 \end{pmatrix} = 6 - 6 = 0$, so the coefficient matrix is singular; a homogeneous system with singular coefficient matrix always has nontrivial solutions beyond $\mathbf{0}$.

**10.**
i) True — a zero row forces $\det = 0$ (property vii).
ii) True — $\det(A) \ne 0$ means $A$ is invertible, and then $\mathbf{x} = A^{-1}\mathbf{b}$ is the one and only solution.
iii) True — swapping two rows flips the sign (property iv).
iv) False. Counterexample: $A = B = I_2$. Then $\det(A + B) = \det\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = 4$, but $\det(A) + \det(B) = 1 + 1 = 2 \ne 4$.

**11.** The rows read $x + 2y = 4$ and $z = 3$. Leading $1$s are in columns 1 and 3, so $x$ and $z$ are dependent; column 2 has no leading $1$, so $y$ is independent. Set $y = t$: then $x = 4 - 2t$ and $z = 3$. All solutions: $(x, y, z) = (4 - 2t,\ t,\ 3)$, $t \in \mathbb{R}$ — infinitely many.
