# Review log — Chapter 27: Kernel methods and kernel regression

## Sources read (before writing)
- `MLT/PPT/Week 2/Slide 1.pdf` (11 pp) and `Slide 2.pdf` (12 pp) — image-only handwritten lecture notes; rendered at 80 dpi and read page by page: PCA's two issues (large $d$, non-linear structure), the circle example $(f_1-a)^2 + (f_2-b)^2 = r^2$, the dual form $w_k = X\alpha_k$, premultiplying to $K\alpha_k = (n\lambda_k)\alpha_k$, the kernel trick ("we managed to compute $\phi(x)^T\phi(x')$ without explicitly computing $\phi(x)$"), polynomial kernel $\kappa(x,x') = (x^Tx'+1)^p$ with the "valid function" / explicit-$\phi$ exercise for $p=3,4$, kernel centering, and the $O(n^3)$ eigen-decomposition note. Content already covered by §24.10; reused here, not re-derived.
- `MLT/PPT/Week 5/Letures_1-4.pdf` and `Lectures_5-7.pdf` (4 pp each) — handwritten regression lectures; read as images: supervised setup, squared loss $\sum_i(f(x_i)-y_i)^2$, the "memorizing gives zero training error / what we care about is test performance / impose structure" points (verbatim quotes used), normal equations $(X^TX)w = Xy$, the geometric projection view, gradient descent, and $O(d^3)$ inversion cost.
- `Live session slide - 2024 Sep/TA notes/Colab/Week 5 Programming Assignment/Week_5_Programming_Assignment.ipynb` — the kernel-regression assignment (Section 2): synthetic data $y = X^3 + \text{noise}$, polynomial kernel of degree 3 `ker = ((xi.T@xj)+1)**3`, the key relation $w = \phi(\mathbb{x})\alpha$ (Q16), $\alpha = \text{pinv}(K)y$, and predictions $\hat y_i = \sum_j \alpha_j\,\text{ker}(x_i,x_j)$ (Q17). Transcripts folder is empty — no MLT transcripts exist; not blocking.
- `chapters/24-pca.md` §24.10 (kernel notation $\kappa$, Gram $K$, Mercer, polynomial/RBF, centering), `chapters/23-linear-polynomial-regression.md` (§§23.1–23.10: loss, normal equations, polynomial regression, overfitting, the ridge remark with the $\lambda$-bookkeeping note), `chapters/26-k-means.md` (Part IV subscript convention).

## Numbers recomputed (twice each: exact fractions + numpy floats)
- §27.8 Gram matrix $K = \begin{pmatrix}4&1&0\\1&1&1\\0&1&4\end{pmatrix}$ from $\kappa(x,z) = (xz+1)^2$ on $(-1,0,1)$ — hand and numpy agree.
- $\lambda = 0$: $\alpha = (1/2,-1,1/2)^T$ (symmetry argument: unique solution of a symmetric system is symmetric); $f(x) = \frac12(1-x)^2 - 1 + \frac12(1+x)^2 = x^2$; numpy: $K\alpha = (1,0,1)^T$ exactly, residuals $0$; closed-form match on off-training points.
- $\lambda = 1$: $\alpha = (1/4,-1/4,1/4)^T$; exact fractions confirm $(K+I)\alpha = (1,0,1)^T$; $f(x) = \frac14 + \frac12x^2$; $f(\text{train}) = (3/4,1/4,3/4)$; MSE $= 1/16$; numpy agrees to all digits.
- §27.7 eg: RBF bump one unit away at $\sigma = 0.5$ is $e^{-2} \approx 0.1353$ — numpy.
- Problem 1: $(xz+1)^3$ binomial coefficients $(1,3,3,1)$; $\phi(x) = (x^3,\sqrt3\,x^2,\sqrt3\,x,1)^T$ verified by dot product.
- Problem 3: $K = \begin{pmatrix}1&1\\1&5\end{pmatrix}$; $\det(K+I) = 11$; $\alpha = (8/11,6/11)^T$ (exact fractions); $f(x) = \frac{12}{11}x + \frac{14}{11}$; $f(0) = 14/11$, $f(2) = 38/11$ — numpy agrees.

## Derivations double-checked
- Gradient of $J(w) = \frac12\|\Phi w - y\|^2 + \frac{\lambda}{2}\|w\|^2$: entrywise chain rule matches §23.3's; $\nabla J = \Phi^T(\Phi w - y) + \lambda w$ ✓.
- Representer step: $\nabla J = 0 \Rightarrow w = \Phi^T\alpha$ with $\alpha = (y - \Phi w)/\lambda$; substitution gives $\lambda\alpha = y - K\alpha$ using $\Phi\Phi^T = K$ (checked entrywise: $(\Phi\Phi^T)_{ij} = \phi(x_i)^T\phi(x_j)$) ✓.
- $K + \lambda I$ invertibility: PSD $K$ + $\lambda > 0$ ⇒ PD — same proof as §23.10's homework ✓.
- $\hat y = K\hat\alpha = y$ for $\lambda = 0$, $K$ invertible (Problem 4) ✓.

## Cross-references verified
- Script-checked every `§X.Y` in the chapter and solutions against actual `## X.Y` headers in `chapters/*.md`: all resolve (10.1, 12.9, 22.10, 23.1–23.10, 24.9–24.10).
- Forward chapter refs (28, 32–33) match OUTLINE.md (Part IV plan); same convention as §23.11's Chapter 28/41 refs.
- Figure `assets/27-kernel-regression-fits.png` exists, resolves relative to `chapters/` (ch25/26 convention), marked original in the HTML comment; viewed the rendered PNG — both panels legible, curves match the §27.8/§27.7 numbers.
- Notation: subscript $x_i$, $\kappa(x,x')$, $K_{ij}$, $\Phi$ ($n \times D$, rows = points) — consistent with §24.10 and the Part IV subscript convention (Ch 26's explicit note); the $\frac{\lambda}{2}$ vs §23.10's $\lambda$ is flagged in §27.4 as the same bookkeeping choice §23.1 notes.

## Thin / contradictory / beyond-lecture points (flagged, not invented)
1. **No model-selection procedure in the sources.** The lectures give no recipe for choosing $\kappa$ or $\lambda$ (no cross-validation, no marginal likelihood). §27.9 covers only: qualitative kernel guidance (shape of the function), §23.10's $\lambda$ remark, and §23.9's validation discipline as the course's only selection tool. Said explicitly in §27.9(i).
2. **No large-$n$ workaround in the sources.** The $O(n^3)$ solve is stated as the bottleneck; no low-rank (Nyström), iterative, or sparse approximation appears in these lectures — §27.9(iii) says so and stops.
3. **$\alpha = K^{-1}y$ vs $\alpha = (K+\lambda I)^{-1}y$.** The assignment uses unregularized `pinv(K)@y`; the chapter presents the regularized form as the general solution and the assignment's as its $\lambda \to 0$ limit. The "pinv because $K$ can be singular" note is grounded in the assignment's code (in their 400-point/degree-3 setup, $\operatorname{rank} K \le 4 \ll 400$, so $K$ is necessarily singular).
4. **No centering for kernel regression.** §24.10 centers the Gram matrix for PCA; the chapter states kernel regression uses $K$ as-is (§27.9(iv)) — a mathematical fact about the derived algorithm, not a source claim.
5. **"Polynomial kernel = polynomial regression" softened.** The function classes coincide (Problem 5), but the $\|\cdot\|$ being penalized differs from §23.8's monomial-coefficient norm; §27.7(iii) says "searches the same curves," not "is the same estimator."
6. **RBF→neuron bridge (§27.10(iii)).** Forward-looking intuition for Part VI, phrased as such — not a source claim.
7. No contradictions between the Week 2 slides, Week 5 lectures, and the programming assignment.

## Fixes made during self-review
- §27.7(iii): weakened "kernel regression with this kernel *is* polynomial regression" → "searches the same curves as §23.8's polynomial regression" (regularization norm differs).
- §27.7: added the missing "Basically, ..." (similarity-vote restatement) — every complex topic now has one.
- §27.8 Step 2b: clarified "training error $1/16$ per point" is the mean squared error (residuals $\pm 1/4$ each).
- Figure generation: fixed two matplotlib scripting bugs (label hack, unbraced `\frac`) before saving; final PNG verified by viewing.
