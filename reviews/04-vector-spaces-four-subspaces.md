# Review log — Chapter 4. Vector spaces and the four fundamental subspaces

## Sources used (all committed in the repo)

1. **Primary:** `sources/course materials/MLF-20261009T001559Z-1-001/MLF/PPT/Week 3/1. Four Fundamental Subspaces.pdf` (10 pages; scanned handwriting, read page-by-page as images). Covers: $C(A)$ as span of columns; $Ax = \mathbf{b}$ solvable iff $\mathbf{b} \in C(A)$; the $4 \times 3$ example (col 3 = col 1 + col 2, $C(A)$ a 2-D subspace of $\mathbb{R}^4$); $N(A) = \{\mathbf{x} \mid A\mathbf{x} = \mathbf{0}\}$ with the closure proof; the $(1,1,-1)^T$ example ($N(A)$ a line in $\mathbb{R}^3$); Remark 1 (invertible $\Rightarrow$ $N(A) = \{\mathbf{0}\}$, $C(A)$ whole space, unique $\mathbf{x} = A^{-1}\mathbf{b}$; else solutions $\mathbf{x} = \mathbf{x}_p + \mathbf{x}_n$); Remark 2 (Gaussian elimination for $N(A)$ on the $3 \times 4$ matrix, pivot cols 1 and 3, free vars $x_2, x_4$, basis vectors $(-2,1,0,0)^T$ and $(2,0,-2,1)^T$); rank = # pivot columns, nullity = # free variables, rank = $\dim C(A)$, nullity = $\dim N(A)$, rank + nullity = $n$; row space $R(A) = C(A^T)$; col rank = row rank; $N(A^T) = \{\mathbf{y} \mid A^T\mathbf{y} = \mathbf{0}\} = \{\mathbf{y} \mid \mathbf{y}^T A = \mathbf{0}\}$ as "a linear combination of rows leading to zero vector"; $\dim C(A^T) + \dim N(A^T) = m$, $\dim N(A^T) = m - r$; the $[1\ 1\ {-1}] \in N(A^T)$ example; the complete $2 \times 2$ example (all four subspaces as lines); the homework matrix $\begin{pmatrix} 1 & 3 & 3 & 2 \\ 2 & 6 & 9 & 7 \\ -1 & -3 & 3 & 4 \end{pmatrix}$ (used as problem 8).
2. `sources/course materials/MLF-20261009T001559Z-1-001/MLF/Transcripts/Week 3/1. Four Fundamental Subspaces.pdf` — lecture transcript; confirmed the PPT reading (column-space solvability question, "4 equations in 3 unknowns", null space as a line in $\mathbb{R}^3$).
3. `sources/course materials/MLF-20261009T001559Z-1-001/MLF/PPT/Week 3/Week 3 Tutorial 3.1.pdf` — four-subspace summary table ($C(A)$ subspace of $\mathbb{R}^m$, etc.); its GeoGebra column-space link was not reused.
4. **Foundations:** `sources/course materials/Maths2_LINEARALGEBRA.pdf` — vector space axioms (Def 3.2.1), subspace + two-condition subspace test (Def 3.4.1, Theorem 3.4.1), trivial subspaces $V$ and $\{\mathbf{0}\}$, linear combination / dependence / independence (Defs 4.2.1–4.2.3), worked examples ($\{(1,2,4),(2,-1,2),(5,0,8)\}$ dependent, $\{(1,1),(1,-1)\}$ independent), span (Def 4.3.1, x-axis and xy-plane examples), basis (Def 4.4.1, standard basis of $\mathbb{R}^n$), dimension (Def 4.5.1, $\dim \mathbb{R}^n = n$), basis-from-spanning-set example for $W = \operatorname{span}\{(1,0,0),(0,1,0),(3,5,0)\}$, "pivot columns of $A$ = basis of column space" (§6.5), rank-nullity theorem (§6.6). Also the non-$\mathbb{R}^n$ example $M_{2\times 3}(\mathbb{R})$ (Example 3.2.2).
5. Cross-check anchors (correctness only, nothing copied): standard references for the four-subspace dimensions table ($r$, $n-r$, $r$, $m-r$) and the RREF basis procedures (pivot columns of $A$ for $C(A)$, free-variable special solutions for $N(A)$).

## What was double-checked

- Every worked example was recomputed by hand a second time and verified numerically (column relations, null-space candidates multiplied against $A$, elimination steps, free-variable back-substitution, left-null combinations). All check out:
  - $4 \times 3$ matrix: col 3 = col 1 + col 2 ✓; $(1,1,-1)^T$ gives $A\mathbf{x} = \mathbf{0}$ ✓.
  - $3 \times 4$ matrix: elimination to $U = [[1,2,2,2],[0,0,2,4],[0,0,0,0]]$ ✓; pivots in cols 1, 3 ✓; $\mathbf{u} = (-2,1,0,0)^T$, $\mathbf{v} = (2,0,-2,1)^T$ both satisfy $A\mathbf{x} = \mathbf{0}$ ✓; $[1\ 1\ {-1}]A = \mathbf{0}$ entry by entry ✓.
  - $2 \times 2$ matrix: $C(A)$ line through $(1,3)^T$, $N(A)$ line through $(-2,1)^T$, row space line through $(1,2)^T$, $N(A^T)$ line through $(-3,1)^T$ — all verifications ✓.
  - Homework matrix (problem 8): elimination, basis vectors $(-3,1,0,0)^T$ and $(1,0,-1,1)^T$ for $N(A)$, left-null vector $(5,-2,1)^T$ — all verifications ✓.
  - Independence/dependence test arithmetic, span parametric forms, basis-finding reductions — all verified ✓.
- Terminology reconciled with Chapters 2–3: "dependent/independent variables" (§3.5) = pivot/free variables here; number of free variables = nullity = $\dim N(A)$; column/row-space solvability story connects directly to §3.3's "which recipe of columns makes $\mathbf{b}$?".

## Discrepancies / uncertainties

- None found between the PPT, the transcript, and Maths2 on the overlapping content (definitions of the four subspaces, rank-nullity, dimension counts all agree).
- The PPT does not define vector space / subspace / span / independence / basis / dimension — it assumes them as recap. These were taken from Maths2 (committed source, same repo). This is recorded here so the coordinator can confirm the sourcing choice.

## Deliberate omissions

- Orthogonality of the subspace pairs ($C(A) \perp N(A^T)$, $C(A^T) \perp N(A)$) is **not** stated or proved here — that is Chapter 5 material; Chapter 4 ends with a forward pointer only. The dimension-sum facts ($\dim C(A) + \dim N(A) = n$, $\dim C(A^T) + \dim N(A^T) = m$) are included since they are pure rank-nullity, from the source slides.
- No material from the other Week 3 decks (orthogonal vectors, projections, least squares) — per instructions.
- The "nonzero rows of the RREF = row-space basis" procedure was cross-checked against standard references but is not in the committed sources, so it is not stated as a general method; the row space is exhibited by inspection in the worked examples (as the lecture itself does).
- The tutorial's GeoGebra visualization link was not embedded (external interactive content; an original matplotlib diagram was drawn instead).
- No external links: none were needed.
