# Review log — Chapter 5. Orthogonality, projections, least squares

## Sources used (all committed in the repo)

1. **Primary:** `sources/course materials/MLF-20261009T001559Z-1-001/MLF/Transcripts/Week 3/`:
   - `2. Orthogonal Vectors and Subspaces.pdf` — length, orthogonality via $\mathbf{x}^T\mathbf{y} = 0$, Pythagoras link, $\mathbf{0}$ orthogonal to everything, mutually orthogonal nonzero $\Rightarrow$ independent (proof), orthonormal vectors ($(\cos\theta, \sin\theta)$ example), orthogonal subspaces def, row space $\perp$ null space (proof), column space $\perp$ left null space (via $A^T$), the $A = [[1,2],[2,4],[3,6]]$ example with dimension check $1+1=2$, $1+2=3$.
   - `3. Projections.pdf` — motivation (inconsistent systems, $b \notin C(A)$), projection onto a line $\hat{x} = \mathbf{a}^T\mathbf{b}/\mathbf{a}^T\mathbf{a}$, $\mathbf{p} = \hat{x}\mathbf{a}$, projection matrix $P = \mathbf{a}\mathbf{a}^T/\mathbf{a}^T\mathbf{a}$, the $\mathbf{a} = (1,1,1)^T$ example ($P = \frac13$ all-ones, $P^T = P$, $P^2 = P$, $C(P)$ = line, $N(P)$ = orthogonal plane, rank 1, scaling $\mathbf{a} \to 2\mathbf{a}$ leaves $P$ unchanged), Cauchy–Schwarz proof from $\lVert\mathbf{e}\rVert^2 \ge 0$.
   - `4. Least Squares and Projections onto a Subspace.pdf` — 1-D calculus route ($2x = b_1, 3x = b_2, 4x = b_3$ $\to$ $\hat{x} = (2b_1+3b_2+4b_3)/29 = \mathbf{a}^T\mathbf{b}/\mathbf{a}^T\mathbf{a}$), the "calculus = projection" bottom line, projection onto $C(A)$ via orthogonal-complement route and first-principles route, normal equations $A^TA\hat{\mathbf{x}} = A^T\mathbf{b}$ ("the most important equation"), $A^TA$ invertible when columns independent (proof left as exercise — I supplied the standard $\lVert A\mathbf{x}\rVert^2$ argument), $P = A(A^TA)^{-1}A^T$ with $P^T = P$, $P^2 = P$ checks, converse ($P^2 = P$ + symmetric $\Rightarrow$ projection onto $C(P)$), special cases ($\mathbf{b} \in C(A) \Rightarrow P\mathbf{b} = \mathbf{b}$; $\mathbf{b} \in N(A^T) \Rightarrow P\mathbf{b} = \mathbf{0}$; $A$ square invertible $\Rightarrow P = I$; rank-1 recovers the line case).
   - `5. Example of Least Squares.pdf` — the line-fit example $A = [[-1,1],[1,1],[2,1]]$, $\mathbf{b} = (1,1,3)^T$: Gaussian elimination showing inconsistency, $A^TA = [[6,2],[2,3]]$, $A^T\mathbf{b} = (6,5)^T$, solution $(4/7, 9/7)^T$, line $y = \frac47x + \frac97$, projections $5/7, 13/7, 17/7$, error vector $(2/7, -6/7, 4/7)^T$ orthogonal to both columns.
2. **Consistency:** `chapters/02-vectors-and-matrices.md` §2.5 (dot product, norm, cosine), `chapters/03-linear-systems-determinants-inverses.md` §3.1/§3.11 (normal-equations preview), `chapters/04-vector-spaces-four-subspaces.md` §4.9–4.10 (four subspaces, dimensions table, three outcomes of $A\mathbf{x} = \mathbf{b}$; Ch 4's figure is referenced, not reused).
3. PPT decks (`PPT/Week 3/` 2–5) were **not** read — the transcripts were complete and clean, and the task said to prefer transcripts if PPT extraction is garbled.
4. **Cross-check anchors (correctness only, nothing copied):** web search on the normal equations confirmed the sign convention $A^T(b - A\hat{\mathbf{x}}) = \mathbf{0} \Rightarrow A^TA\hat{\mathbf{x}} = A^T\mathbf{b}$, the uniqueness condition (full column rank), and $N(A^TA) = N(A)$. The MML book landing page was fetched but its chapters were not needed — the transcript derivations plus hand recomputation sufficed.

## What was double-checked (all recomputed by hand a second time)

- Least-squares example: $A^TA = [[6,2],[2,3]]$ ✓, $A^T\mathbf{b} = (6,5)^T$ ✓, $\hat{\boldsymbol\theta} = (4/7, 9/7)^T$ ✓, elimination gives $0 = 2$ (inconsistent) ✓, projections $(5/7, 13/7, 17/7)$ ✓, $\mathbf{e} = (2/7, -6/7, 4/7)^T$ with both column dot products $= 0$ ✓, $\lVert\mathbf{e}\rVert^2 = 56/49 = 8/7$ ✓.
- Projection-matrix check on the same example: $P = [[13/14, 3/14, -1/7],[3/14, 5/14, 3/7],[-1/7, 3/7, 5/7]]$ — **first computation contained arithmetic slips** (caught on re-verify: $P_{12}$ was $1/2$, should be $3/14$; $P_{13}$ was $2/7$, should be $-1/7$); corrected and confirmed $P\mathbf{b} = \mathbf{p}$ row by row and $\operatorname{tr}(P) = 2$ = rank ✓. This was the one real error caught by the re-read pass; fixed before commit.
- Line projection $\mathbf{b} = (1,2,3)$, $\mathbf{a} = (1,1,1)$: $\hat{x} = 2$, $\mathbf{p} = (2,2,2)$, $\mathbf{e} = (-1,0,1)$, $\mathbf{a}^T\mathbf{e} = 0$ ✓.
- $xy$-plane example (matches `ch05-least-squares-geometry.png`): $\mathbf{p} = (1,2,0)$, $P = \operatorname{diag}(1,1,0)$, $P^2 = P$, $P^T = P$ ✓; normal-equation cross-check with $A = [[2,0],[0,2],[0,0]]$ gives $\hat{\mathbf{x}} = (1/2, 1)^T$, $A\hat{\mathbf{x}} = \mathbf{p}$ ✓.
- All 12 problem-set answers verified numerically (residual ⟂ column checks everywhere; problem 7's $A^T\mathbf{b} = (5, 11)^T$ — note the easy-to-miss $c_2^T\mathbf{b} = 11$, not 12).
- Cauchy–Schwarz algebra, $P^T = P$ / $P^2 = P$ for both line and general case, the $P = I$ special case, the converse proof identity.

## Discrepancies / source issues

1. Transcript 5 has acknowledged instructor typos: the error norm was first stated as "$-2/7$" (impossible for a norm), then corrected mid-lecture to the components $(2/7, -6/7, 4/7)^T$ with "this should not be square." The final numeric value $\lVert\mathbf{e}\rVert^2 = 8/7$ is **my computation**, not stated in the lecture — noted in the chapter's step 6 without attribution to the source.
2. Transcript 4 says "any given vector is in either $C(A)$ or $N(A^T)$" — loose phrasing. The chapter states the correct version: dimensions add up, so every vector **decomposes uniquely** into a piece from each (that's what $\mathbf{b} = \mathbf{p} + \mathbf{e}$ is).
3. Transcript 2's example "$(2, 2, 2, -1, 2)$" is garbled in extraction (probably two vectors). I did not reuse it; used clean equivalents ($(\cos\theta, \sin\theta)$ is from the transcript itself).
4. Transcript 2's claim that orthogonal-subspace pairing holds was verified against Ch 4's dimension table — consistent.

## Uncertainties

- None material. The transcripts are self-contained; every formula used appears in them, and every number was recomputed twice.

## Deliberate omissions

- **Gram–Schmidt / QR:** not in the source decks; orthonormal bases are only *used*, never constructed. Omitted (likely Chapter 7 SVD material).
- **Pseudo-inverse** $A^\dagger = (A^TA)^{-1}A^T$: not named in the sources; the formula appears, the name doesn't. Omitted.
- **Oblique projections:** only $P^2 = P$ without symmetry gets a mention (problem 10a); not developed.
- **Stats view** of residuals (variance estimates): belongs to the Stats chapters, not here.
- **External links / reused diagrams:** none — the one figure (`assets/ch05-least-squares-geometry.png`) is original, drawn with matplotlib for this chapter, matching the §5.7 $xy$-plane example exactly.
- External web sources used only as cross-checks; nothing copied from them.
