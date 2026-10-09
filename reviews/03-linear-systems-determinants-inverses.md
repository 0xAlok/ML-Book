# Review log — Chapter 3: Linear systems, determinants, inverses

Branch: `draft/linear-systems-determinants-inverses` (from `master`). Not merged — awaiting independent review.

## Sources used

Primary (all under `sources/course materials/Maths1/Lecture PPT-Slides/`, read via the read tool with PDF-to-text conversion):
- Week 1: `3.System-of-linear-equations.pdf` — linear equation/system definitions, $Ax=b$ matrix form, augmented-matrix idea, the three solution outcomes with the grocery-buyer 2×2 examples (unique: $2x+y=215$, $3x+y=260$; infinite: $2x+y=215$, $4x+2y=430$; none: $2x+y=215$, $4x+2y=400$), geometric line interpretation.
- Week 1: `4.determinants-1.pdf` — determinant definition, $2\times2$ formula $ad-bc$ (worked: $\begin{smallmatrix}2&3\\6&10\end{smallmatrix}\to2$), $3\times3$ cofactor expansion (worked: $\begin{smallmatrix}2&4&1\\3&8&7\\5&6&9\end{smallmatrix}\to70$), $\det(I)=1$, $\det(AB)=\det(A)\det(B)$ with $2\times2$ proof, row-operation effects (swap flips sign; add-multiple unchanged; scale row scales).
- Week 1: `5.determinants-2.pdf` — triangular shortcut (worked: $\begin{smallmatrix}2&4&3\\0&8&7\\0&0&9\end{smallmatrix}\to144$), transpose definition and $\det(A^T)=\det(A)$, minors/cofactors, inductive definition, expansion along any row/column, property list, computational tips (zero row → 0; dependent row → 0; expand along sparsest row/column).
- Week 2: `6.Cramer_s-rule.pdf` — Cramer's rule statement $x_i=\det(A_{x_i})/\det(A)$ for $2\times2$, $3\times3$, $n\times n$.
- Week 2: `7.inv-coeff-matrix.pdf` — inverse definition $AB=BA=I$, $\det(A)\det(A^{-1})=1$, adjugate formula $A^{-1}=\frac{1}{\det(A)}\operatorname{adj}(A)$, solving via $\mathbf{x}=A^{-1}\mathbf{b}$ (worked $3\times3$ grocery example, solution $(45,125,150)$), homogeneous systems (unique trivial solution iff invertible; infinite solutions iff $\det=0$).
- Week 2: `8.row-echelon.pdf` — (reduced) row echelon form definition, reading solutions from RREF (zero row with $b_i\ne0$ → no solution; dependent/independent variables; free-variable parametrization).
- Week 2: `9.row-reduction.pdf` — the three elementary row operations with $R_i\leftrightarrow R_j$, $R_i\leftarrow cR_i$, $R_i\leftarrow R_i+cR_j$ notation, row-reduction algorithm, determinant-via-row-reduction with the operation-effect table.
- Week 2: `10.gauss-elim.pdf` — augmented matrix $[A\mid b]$, Gaussian elimination steps, homogeneous systems (trivial solution; more variables than equations → nontrivial solutions guaranteed), inverse via $[A\mid I]$ row reduction (mentioned in contents).
- Week 2: `determinants-3.pdf` — recap of properties 1–4 and computational tips (confirms property list).

Consistency (voice/notation): `chapters/01-sets-functions-preliminaries.md` (§1.6 $\mathbb{R}^d$, §1.8 bijectivity, §1.10 inverse functions — the inverse-as-undo echo), `chapters/02-vectors-and-matrices.md` (§2.7–2.11 — columns view of $A\mathbf{x}$ as linear combination of columns, identity matrix, non-commutativity). Chapter 1's `§1.15` already forward-points to matrix inverses in Chapter 3; Chapter 2's "Where this goes next" names Chapter 3 as linear systems — both match.

Cross-check anchors (correctness only, nothing copied): web search on the determinant's geometric meaning confirmed the standard statement used in §3.7 ($|\det(A)|$ = area/volume scaling factor; sign = orientation; $\det=0$ = collapse). MML Book ch. 2 and D2L linear-algebra appendix were listed as anchors; the geometric claim was verified against multiple standard references instead of copying any.

## What was double-checked

Every number in the chapter and solutions was recomputed by hand during writing, and again in the re-read pass:
- Trio: $(45,125)$ satisfies $2x+y=215$ ($90+125$) and $3x+y=260$ ($135+125$) ✓; $4x+2y=430$ is exactly $2\times$ the first equation ✓; $4x+2y=400$ contradicts $2\times$ first ($=430$) ✓.
- §3.2 check-example $(1,2,3)$: $1+4+3=8$, $2+2+3=7$, $1+2+6=9$ ✓.
- 3×3 elimination (§3.5): every intermediate matrix recomputed ($R_2-2R_1=[0,-1,-3\mid-11]$, $R_3-3R_1=[0,-4,-2\mid-14]$, $R_3+4R_2=[0,0,10\mid30]$); back-substitution $z=3$, $y=11-9=2$, $x=6-2-3=1$; all three original equations verified ✓.
- Inverse: $\det\begin{smallmatrix}2&1\\1&1\end{smallmatrix}=1$; $A^{-1}=\begin{smallmatrix}1&-1\\-1&2\end{smallmatrix}$; $AA^{-1}=I$ computed entry by entry ✓; $\mathbf{x}=A^{-1}(5,3)^T=(2,1)$, checked in both equations ✓.
- Determinants: $2\times2$ ($20-18=2$) ✓; $3\times3$ expansion ($60+32-22=70$) cross-checked via Sarrus ($144+140+18-40-84-108=70$) ✓; triangular ($2\cdot8\cdot9=144$) ✓; row-reduction det ($R_2-3R_1\to\begin{smallmatrix}1&2\\0&-2\end{smallmatrix}$, det $-2$; direct $4-6=-2$) ✓.
- Cramer (§3.8): $\det(A)=5$, $\det(A_1)=10$, $\det(A_2)=15$ → $(2,3)$; verified in both equations ✓.
- Every problem-set solution (1–11) recomputed: elimination $(2,-1,4)$ verified in all three equations; $3\times3$ det $=22$; triangular $-24$; $\det(2A)=40$; Cramer $(2,3)$; homogeneous family $(-2t,t)$; the $\det(A+B)$ counterexample ($4\ne2$) ✓.
- REF/RREF examples: $A_{\text{ref}}$ is REF-but-not-RREF (the $3$ above the column-3 leading $1$); $A_{\text{rref}}$ satisfies all four conditions; the staircase-violating $2\times3$ is correctly not REF.
- Diagram `assets/ch03-three-cases.png` is an original matplotlib figure (three panels, equations matching the §3.3 trio); intersection point $(45,125)$ annotated; no external source.

## Discrepancies found in sources

1. **REF vs RREF conflated.** `8.row-echelon.pdf` defines "(Reduced)Row echelon form" with all four conditions at once — leading entries required to be $1$ *and* the only nonzero in their column — which is really the RREF definition; the staircase-only REF is never separately defined. The chapter uses the standard split (REF = staircase; RREF = staircase + clean leading $1$s) and carries an explicit `Note:` flagging the difference. (Standard references, incl. the MML/D2L anchors, use the split version.)
2. **Garbled worked example avoided.** In `10.gauss-elim.pdf` ("Another example": $x_1+x_2+x_3=2$, $x_2-3x_3=1$, $2x_1+x_2+5x_3=0$) the slide's final augmented row renders as $[0\ 0\ 0\mid 3]$, but recomputation from the printed steps ($R_3-2R_1$ then $R_3+R_2$) gives $[0\ 0\ 0\mid -3]$. The outcome (no solution) is unaffected, but the chapter does not reuse this example — §3.5's $3\times3$ is original and fully verified.
3. **Cramer's-rule slide numbers inconsistent.** In `6.Cramer_s-rule.pdf` the $2\times2$ example's printed determinants ($\det(A_{x_1})=38$, $\det(A_{x_2})=76$) do not match direct computation from the printed matrices ($55-21=34$; $28-66=-38$). The rule statement itself is standard and correct; §3.8's worked example is original with verified arithmetic.
4. **Unreadable pages.** A few pages were images the read tool could not convert (title and "Thank you" slides: pages 1/47 of `3.System-of-linear-equations.pdf`, 1/35/48 of `4.determinants-1.pdf`, 1/52 of `5.determinants-2.pdf`). By the slide decks' pattern these are cover/thanks pages with no mathematical content; nothing was built on them.

## Uncertainties

None material. The geometric interpretation of the determinant (area/volume scaling, orientation sign) is not stated in the IITM slides but is standard, was verified against external references, and is marked as geometric intuition rather than a sourced theorem. The "elimination beats inversion in practice" line is standard numerical-linear-algebra consensus, kept to one honest line as instructed.

## Deliberate omissions (per task scope)

- The adjugate-matrix formula $A^{-1}=\frac{1}{\det A}\operatorname{adj}(A)$ (in `7.inv-coeff-matrix.pdf`) is not taught — the task scoped the inverse to the $2\times2$ formula; the adjugate is noted as existing in sources for a possible later pass.
- The slides' $3\times3$ inverse-via-adjugate grocery example (solution $(45,125,150)$) was not reproduced; its arithmetic was spot-checked ($A(45,125,150)^T=(1960,2215,1135)^T$ ✓, $\det=-188$ ✓) but the example is too heavy for this chapter's worked sets.
- Full reduction to RREF for the §3.5 $3\times3$ is not shown — REF + back-substitution is the cleaner pedagogical path and matches the task.
- The inductive $n\times n$ determinant definition is stated only in its $3\times3$ working form; LU/QR/SVD factorizations are not in the sources and not introduced.
- External links: none added; per instructions, links are reserved for dire situations, and the chapter is self-contained.
