# Solutions to Chapter 2: Vectors and matrices

1. $(5, -2, 4) + 3(1, 0, -1) - (2, 3, 0)$. Scale first, then add coordinatewise:
$$3(1, 0, -1) = (3, 0, -3),$$
$$(5, -2, 4) + (3, 0, -3) = (5 + 3,\ -2 + 0,\ 4 - 3) = (8, -2, 1),$$
$$(8, -2, 1) - (2, 3, 0) = (8 - 2,\ -2 - 3,\ 1 - 0) = (6, -5, 1).$$

2. $\mathbf{u} = (6, 8)$. Norm:
$$\lVert \mathbf{u} \rVert = \sqrt{6^2 + 8^2} = \sqrt{36 + 64} = \sqrt{100} = 10.$$
Unit vector in the same direction: divide by the length,
$$\hat{\mathbf{u}} = \frac{(6, 8)}{10} = \left(\frac{3}{5}, \frac{4}{5}\right).$$
Check: $\left\lVert \left(\frac{3}{5}, \frac{4}{5}\right) \right\rVert = \sqrt{\frac{9}{25} + \frac{16}{25}} = \sqrt{1} = 1$. ✓

3. $(3, -1, 2) \cdot (-2, 4, 5) = 3 \times (-2) + (-1) \times 4 + 2 \times 5 = -6 - 4 + 10 = 0$.
The dot product is zero, so $\cos\theta = 0$ and the angle between them is $90^\circ$ — the vectors are at right angles to each other (this gets the name $orthogonal$ in Chapter 5).

4. $\mathbf{u} = (1, 2)$, $\mathbf{v} = (2, 1)$:
$$\mathbf{u} \cdot \mathbf{v} = 1 \times 2 + 2 \times 1 = 2 + 2 = 4,$$
$$\lVert \mathbf{u} \rVert = \sqrt{1 + 4} = \sqrt{5}, \qquad \lVert \mathbf{v} \rVert = \sqrt{4 + 1} = \sqrt{5},$$
$$\cos\theta = \frac{4}{\sqrt{5} \cdot \sqrt{5}} = \frac{4}{5} = 0.8 \;\Rightarrow\; \theta = \cos^{-1}(0.8) \approx 36.87^\circ.$$

5. Want $(7, 5) = a(1, 2) + b(2, 1)$. Expanding the right side: $(a + 2b,\ 2a + b)$. Matching components:
$$\begin{cases} a + 2b = 7 \\ 2a + b = 5 \end{cases}$$
From the first, $a = 7 - 2b$. Substituting into the second: $2(7 - 2b) + b = 5$, i.e. $14 - 3b = 5$, so $3b = 9$ and $b = 3$. Then $a = 7 - 2 \times 3 = 1$.
Check: $1(1, 2) + 3(2, 1) = (1 + 6,\ 2 + 3) = (7, 5)$. ✓

6. $A = \begin{pmatrix} 2 & 0 & -1 \\ 1 & 3 & 4 \end{pmatrix}$ has 2 rows and 3 columns: shape $2 \times 3$. The $(2, 1)$-th entry is the row-2, column-1 entry: $A_{21} = 1$. Not square ($2 \ne 3$), hence not diagonal either.

7. $A + B$ entrywise:
$$\begin{pmatrix} 1 & -2 \\ 0 & 3 \end{pmatrix} + \begin{pmatrix} 4 & 1 \\ 2 & -1 \end{pmatrix} = \begin{pmatrix} 1+4 & -2+1 \\ 0+2 & 3+(-1) \end{pmatrix} = \begin{pmatrix} 5 & -1 \\ 2 & 2 \end{pmatrix}.$$
$2A = \begin{pmatrix} 2 & -4 \\ 0 & 6 \end{pmatrix}$, so
$$2A - B = \begin{pmatrix} 2-4 & -4-1 \\ 0-2 & 6-(-1) \end{pmatrix} = \begin{pmatrix} -2 & -5 \\ -2 & 7 \end{pmatrix}.$$

8. $3 \times 2$ times $2 \times 2$: inner dimensions match (2 = 2), so the result is $3 \times 2$. Each entry = row of left dotted with column of right:
- $(1,1)$: $(2, -1) \cdot (1, 3) = 2 \times 1 + (-1) \times 3 = 2 - 3 = -1$
- $(1,2)$: $(2, -1) \cdot (2, -1) = 2 \times 2 + (-1) \times (-1) = 4 + 1 = 5$
- $(2,1)$: $(0, 3) \cdot (1, 3) = 0 \times 1 + 3 \times 3 = 9$
- $(2,2)$: $(0, 3) \cdot (2, -1) = 0 \times 2 + 3 \times (-1) = -3$
- $(3,1)$: $(1, 1) \cdot (1, 3) = 1 \times 1 + 1 \times 3 = 4$
- $(3,2)$: $(1, 1) \cdot (2, -1) = 1 \times 2 + 1 \times (-1) = 1$
$$\begin{pmatrix} 2 & -1 \\ 0 & 3 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 2 \\ 3 & -1 \end{pmatrix} = \begin{pmatrix} -1 & 5 \\ 9 & -3 \\ 4 & 1 \end{pmatrix}.$$

9. $$AB = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix} = \begin{pmatrix} 0\cdot2 + 1\cdot0 & 0\cdot0 + 1\cdot3 \\ 1\cdot2 + 0\cdot0 & 1\cdot0 + 0\cdot3 \end{pmatrix} = \begin{pmatrix} 0 & 3 \\ 2 & 0 \end{pmatrix},$$
$$BA = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 2\cdot0 + 0\cdot1 & 2\cdot1 + 0\cdot0 \\ 0\cdot0 + 3\cdot1 & 0\cdot1 + 3\cdot0 \end{pmatrix} = \begin{pmatrix} 0 & 2 \\ 3 & 0 \end{pmatrix}.$$
$AB \ne BA$ — another concrete case of non-commutativity.

10. Row-times-column: row 1 $\cdot$ $(5, 6)$ = $1 \times 5 + 2 \times 6 = 5 + 12 = 17$; row 2 $\cdot$ $(5, 6)$ = $3 \times 5 + 4 \times 6 = 15 + 24 = 39$. So $A\mathbf{v} = (17, 39)$.

Columns view: the columns of $A$ are $\mathbf{a}_1 = (1, 3)$, $\mathbf{a}_2 = (2, 4)$, and
$$A\mathbf{v} = 5\mathbf{a}_1 + 6\mathbf{a}_2 = 5(1, 3) + 6(2, 4) = (5, 15) + (12, 24) = (17, 39).$$
Both views agree. ✓

11. False. The product of an $m \times n$ matrix and an $n \times p$ matrix is $m \times p$ — the $outer$ dimensions: rows from $A$, columns from $B$. ($p \times m$ is wrong in both order and meaning; e.g. a $2 \times 3$ times a $3 \times 4$ gives $2 \times 4$, not $4 \times 2$.)

12. Transpose flips rows and columns: $\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}^T = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$ — it equals itself, so yes, symmetric. And yes: a $1 \times n$ row vector's transpose is $n \times 1$, i.e. a column vector, directly from $(A^T)_{ij} = A_{ji}$.
