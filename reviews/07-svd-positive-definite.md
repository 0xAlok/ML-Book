# Review log — Chapter 7 (SVD and positive definite matrices)

## Sources used
- MLF Transcripts/Week 5/Lecture 5.6 — Singular Value Decomposition (SVD theorem, the $A^TA$ bridge: $\sigma_i = \sqrt{\lambda_i}$, $\mathbf{y}_i = A\mathbf{x}_i/\sigma_i$, $U$'s columns = eigenvectors of $AA^T$).
- MLF Transcripts/Week 5/Lecture 5.7 — Example of SVD ($A = [[\sqrt{2},1],[0,\sqrt{2}]]$: defective, $A^TA = [[2,\sqrt{2}],[\sqrt{2},3]]$, $\lambda = 4,1$, $\sigma = 2,1$, $\mathbf{x}_1 = (1,\sqrt{2})^T/\sqrt{3}$, $\mathbf{x}_2 = (\sqrt{2},-1)^T/\sqrt{3}$).
- MLF Transcripts/Week 6/1 — Positive Definiteness (PD function def, quadratic $ax^2+2bxy+cy^2$ conditions $a>0$, $ac>b^2$ via completing the square, semi-definite/saddle cases, $f = V^TAV$ connection).
- MLF Transcripts/Week 6/2 — Positive Definite Matrices (def $\mathbf{x}^TA\mathbf{x}>0$; equivalence with all-eigenvalues-$>0$, both directions proved via spectral theorem).
- PPT PDFs for Weeks 5–6 consulted for slide-level checks; transcripts preferred (PPT extraction garbled, as expected).
- NOT used: Week 6 decks 3–6 (PCA → Chapter 24); Week 5 decks 1–5 (complex/Hermitian/unitary theory — out of scope, one-sentence mention at most; chapter works over real matrices).
- Ch 6 (§6.11 spectral theorem, §6.12 bridge pointer) and Ch 5 (§5.4 orthonormality, $Q^TQ=I$) re-read for notation/voice continuity.

## Double-checked (hand recomputation)
- §7.4 example: $A^TA$ eigenvalues via $\lambda^2-5\lambda+4=0$ → $4,1$ ✓; eigenvectors $(1,\sqrt{2})^T$, $(\sqrt{2},-1)^T$ verified against $A^TA-\lambda I$ ✓; $\mathbf{y}_1 = (\sqrt{2/3}, 1/\sqrt{3})^T$, $\mathbf{y}_2 = (1/\sqrt{3}, -\sqrt{2/3})^T$ orthonormal ✓; $U\Sigma V^T$ reconstructed **entry by entry** — all four match $A$ ✓.
- §7.5 rectangular example: $A^TA$ char poly $\lambda(1-\lambda)(\lambda-3)$ re-derived ✓; during writing I caught and fixed an eigenvector error (initially wrote $(1,1,1)^T$ for $\lambda=3$; recheck showed $M(1,1,1)^T = (2,2,4)^T \ne 3(1,1,1)^T$; correct eigenvector is $(1,1,2)^T$, verified $(M-3I)(1,1,2)^T = 0$) ✓; $\mathbf{u}_1 = (1/\sqrt{2},1/\sqrt{2})^T$, $\mathbf{u}_2 = (1/\sqrt{2},-1/\sqrt{2})^T$ unit + orthogonal ✓; all six entries of $U\Sigma V^T$ re-verified against $A$ ✓; $V^TV=I$ for all three columns ✓.
- PD examples: $[[2,1],[1,2]]$ eigenvalues $3,1$ ✓, completing-the-square $2(x+y/2)^2 + \tfrac{3}{2}y^2$ ✓; $[[1,2],[2,1]]$ eigenvalues $3,-1$, $\mathbf{x}^TB\mathbf{x} = -2$ at $(1,-1)$ ✓; $[[4,2],[2,3]]$ eigenvalues $(7\pm\sqrt{17})/2 > 0$ ✓; $x^2+4xy+5y^2 = (x+2y)^2+y^2$ ✓; rank-1 approx $A_1 = [[2\sqrt{2}/3, 4/3],[2/3, 2\sqrt{2}/3]]$ ✓.
- §7.10 claim about $f = 2x^2+4xy+y^2$: Hessian $[[4,4],[4,2]]$, eigenvalues $3\pm\sqrt{17}$ (mixed signs) → saddle, consistent with $ac-b^2 = -2 < 0$. See discrepancy note below.
- Cross-check against MML book Ch. 4 anchors (via web search, verification only, nothing copied): $A = U\Sigma V^T$ with $\sigma_i = \sqrt{\lambda_i(A^TA)}$, $\mathbf{u}_i = A\mathbf{v}_i/\sigma_i$, and $U\Sigma V^T$ hand-check procedure — all consistent with the chapter's presentation.

## Discrepancies / uncertainties
1. **Transcript internal contradiction (Lecture 6.1):** the early remark on $f(x,y) = 2x^2+4xy+y^2$ says the second-derivative check "implies $f$ has minima at $(0,0)$", but the lecture's own later rigorous criterion ($a>0$, $ac>b^2$) gives $ac-b^2 = 2-4 = -2 < 0$ → **saddle**, and the Hessian $[[4,4],[4,2]]$ has mixed-sign eigenvalues ($3\pm\sqrt{17}$). The chapter follows the rigorous criterion (the later, proved result) and §7.10 explicitly works this as a saddle — likely a transcription slip or loose preview remark in the source. Flagging here; the math is unambiguous.
2. $2 \times 2$ PD test presented as $a>0$, $\det(A)>0$ — exactly the lecture's $a>0$, $ac-b^2>0$ (with $A = [[a,b],[b,c]]$); no general leading-principal-minor criterion, since the source never states one beyond $2 \times 2$.
3. "Best rank-$k$ approximation" optimality is stated (Frobenius/Eckart–Young sense) but not proved — source doesn't develop it; full treatment deferred to Ch 24 per task scope.
4. Second-derivative test: source checks second partials for the quadratic case and ties minima to definiteness; chapter gives the Hessian/minimum/saddle rule for that setting and forwards the general theory to Ch 10/12. No multivariate second-derivative theorem is proved (source doesn't).

## Re-read pass (fixes applied as a second commit)
- Geometry figure caption/suptitle said "§7.3 example" — the figure shows the §7.4 matrix; corrected to §7.4 in both the markdown alt text and the regenerated PNG suptitle.
- §7.4 note pointed at "(§7.5)" for the reflection remark — the geometry discussion is §7.6; corrected.
- §7.10 opened with "Return to $f = 2x^2+4xy+y^2$ from §7.8" but §7.8 had never introduced that specific $f$ (it was only in the source lecture). Added the source's motivating example as an `eg` block at the start of §7.8 (partials, stationary point, second partials), so §7.10's "Return to" lands.
- Independent NumPy re-verification: both $U\Sigma V^T$ reconstructions exact to machine precision ($\le 2.3\times10^{-16}$); $U^TU = I$, $V^TV = I$ confirmed; rank-1 approx entries match; all PD eigenvalues/determinants and the saddle-Hessian mixed signs confirmed.

## Deliberate omissions
- PCA material (Week 6 decks 3–6) → Chapter 24, per task.
- Complex/Hermitian/unitary theory (Week 5 decks 1–5) → not needed for real-matrix SVD/PD; mentioned nowhere (chapter is real throughout).
- Full SVD existence proof details beyond the lecture's level (e.g. orthonormal extension argument is summarized, not proved).
- Reduced vs full SVD kept brief (one paragraph + example note).
- No external links in the chapter (none needed); no stock imagery — the geometry figure is an original matplotlib illustration (`chapters/assets/ch07-svd-geometry.png`), generated from the §7.4 example's own computed vectors.
