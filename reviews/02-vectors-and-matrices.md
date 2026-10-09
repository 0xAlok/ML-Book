# Review log — Chapter 2: Vectors and matrices

## Sources used

- Primary: `sources/course materials/Maths1/Lecture PPT-Slides/Week 1/1.vectors.pdf` (vector as list, row/column, addition coordinatewise, scalar multiplication, grocery/Arun-Neela examples, parallelogram/head-to-tail addition, R^n = points in R^n, magnitude+direction physical view, (1,2)+(2,1)=(3,3)).
- Primary: `sources/course materials/Maths1/Lecture PPT-Slides/Week 1/2.matrices.pdf` (matrix definition, m×n, (i,j)-th entry, square/diagonal/scalar/identity matrices, entrywise addition and scalar multiplication, matrix multiplication formula (AB)_ij = Σ_k A_ik B_kj, shape rule A_{m×n}B_{n×p}=(AB)_{m×p}, scalar mult = cI multiplication, IA=A=AI, the full property list incl. AB≠BA, worked products [[1,2],[3,4]]×[[1,2,3],[3,4,5]] and [[1,2],[3,4]]×(5,6)).
- Secondary: `sources/course materials/Maths2_LINEARALGEBRA.pdf` — vectors/matrices portions only: §1.1–1.3 (same definitions/examples as above), §9.1–9.3 (dot product definition (x1,y1)·(x2,y2)=x1x2+y1y2, worked (2,4)·(3,5)=26 and (1,2,3)·(2,0,1)=5, length via Pythagoras, length = sqrt(v·v), angle formula cos θ = (u·v)/sqrt((u·u)(v·v))), Def 4.2.1 (linear combination Σ αi vi), symmetric matrices used as A = A^T (§7, line ~7499), A^T notation (line ~2207).
- Cross-check anchors (verify-only, never copied): https://mml-book.github.io/ ch.2 and https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html (§22.1.1–22.1.2: point vs direction views, head-to-tail addition, u·v = Σ ui vi, angle θ = arccos((v·w)/(||v|| ||w||)), cosine similarity, projection length ||v||cosθ).

## What was double-checked

- Every worked numeric computation recomputed by hand, twice: the grocery sum (227,110,49,125,48), the full 2×2×2×3 product (each of 6 entries: 7,10,13,15,22,29), the matrix-vector product (17,39) in both row-view and columns-view, all problem-set/solution numbers. Independent recomputation during the re-read found no errors.
- Angle examples: (1,0) vs (1,1) → 45° (matches the geometric picture); (4,3) vs (−4,3) → cosθ = −7/25, θ ≈ 106.3° (arccos(−0.28) ≈ 106.26° ✓); (1,2) vs (2,1) → cosθ = 4/5 ≈ 36.87° ✓.
- Norm/unit-vector checks: ||(3,4)||=5, (3/5,4/5) has norm 1; ||(6,8)||=10 ✓.
- Linear-combination solutions: (7,5)=1(1,2)+3(2,1) verified by substitution ✓.
- Non-commutativity counterexamples: AB=[[1,0],[3,0]] vs BA=[[1,2],[0,0]] (§2.11); solutions Q9: AB=[[0,3],[2,0]] vs BA=[[0,2],[3,0]] ✓.
- Zero matrix/transpose/symmetric definitions are standard and consistent with the sources' usage (A^T notation, symmetric as A=A^T); verified against d2l's transpose notation and MML ch.2 conventions during cross-check.
- The columns-view identity Ax = Σ xi ai verified numerically on the chapter's own example (17,39 both ways ✓) and illustrated in ch02-matvec-columns.png (2a1+a2=(4,2)+(1,1)=(5,3) ✓, matches the drawn arrows).

## Discrepancies / uncertainties

- None found in the primary sources. One garbled PDF extraction (the matrices.pdf matrix-addition example with 3×2 decimals) was NOT reused — a clean equivalent example was written instead.
- The sources number vector entries from 1 ((i,j)-th entry) and so does this chapter; consistent with Chapter 1's ordered-pair conventions.

## Deliberate omissions (things in sources that belong to later chapters)

- Linear systems / solving Ax=b (Maths1 3.System-of-linear-equations.pdf, Maths2 §1.3.2, §3–§8): → Chapter 3.
- Determinants and matrix inverses (Maths1 4/5 determinants slides, Maths2 §2, §5): → Chapter 4 (outline order).
- Vector spaces, subspaces, span, linear independence (Maths2 §4): → Chapter 4.
- Inner-product-space theory, Cauchy-Schwarz, Gram-Schmidt, orthogonal transformations (Maths2 §9.4–9.10): → Chapter 5 (orthogonality). Only the R^n dot product, norm, and angle were kept here, as Chapter 2's brief requires.
- Row-reduction / Gaussian elimination: → Chapter 3.
- Physics-flavoured material (velocity/force examples, plane-and-wind): kept to one sentence; the book is ML-flavoured by design.

## Content notes for the reviewer

- §2.5's cosine-similarity and projection mentions are marked as ML context / Chapter 5 preview, not derived claims — the angle formula itself is sourced (Maths2 §9.1.3/9.3, verified on d2l).
- The "columns as linear combination" view (§2.10) is a standard re-expression of the sourced (AB)_ij formula, verified on d2l and numerically in-chapter; the sources present only the row-times-column mechanics.
- One illustration is original matplotlib (ch02-matvec-columns.png); one is reused from d2l.ai with the source URL in the HTML comment above it.
- Projection ("how much of v lies along u") is intentionally a one-line preview; the full treatment belongs to Chapter 5.
