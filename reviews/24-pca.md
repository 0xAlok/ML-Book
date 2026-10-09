# Review log — Chapter 24 (PCA), draft/pca

Date: 2026-10-09. Reviewer: chapter worker (self-review pass before coordinator's independent review).

## Sources re-read (all, before writing)

- MLF Week 6 PPT slides 3–6 ("Principal Component Analysis", "(Contd.)", "PCA as maximizing variance", "PCA in higher dimensions") — image-only PDFs, text extraction yielded garbage; content taken from the transcripts instead.
- MLF Week 6 Transcripts 3–6 (Prof. Prashanth L A): reconstruction-error derivation (optimal $z_{ij} = x_i^Tu_j$, $\beta_j = \bar x^Tu_j$, $J^* = \sum_{j=m+1}^d u_j^TCu_j$); Lagrangian + quotient-rule derivations of $Cu = \lambda u$; top-$m$ eigenvectors rule; the $\{(-1,-1),(0,0),(1,1)\}$ example; the $d \gg n$ lecture ($C = \frac1nA^TA$, rank $\le n$, $\frac1nAA^T$ trick, SVD connection flagged).
- MLF Week 6 Tutorial Part 2 (PCA): 8-step algorithm, Example 1 (3-D, on-a-line data), Example 2 (four 2-D points). Tutorial Part 1 is positive-definiteness (Chapter 7's territory) — not used.
- MLT Week 2 Slide 1 + Slide 2 (handwritten slides, read as rendered images): "Issues with PCA" (time complexity $O(d^3)$, eigenfaces; data not in a linear subspace), dual form ($w_k = X\alpha_k$, $K\alpha_k = n\lambda_k\alpha_k$, $\alpha_k^T K \alpha_k = 1$, $\alpha_k = \beta_k/\sqrt{n\lambda_k}$), circle example with explicit $\phi$ to $\mathbb{R}^6$, kernel trick ($(x^Tx'+1)^2 = \phi(x)^T\phi(x')$ worked expansion), polynomial + RBF kernels, Mercer's theorem (informal), kernel centering ($K^c_{ij} = K_{ij} - \theta_i - \theta_j + P$), compressed representation $\sum_j \alpha_{kj}K^c_{ij}$.

## What was checked

1. **Every derivation re-read against the transcript/slide it came from.** Reconstruction-error algebra, variance $= u^TCu$, Lagrangian and quotient-rule routes, dual-form derivation, kernel-centering derivation — all match the sources.
2. **Every numeric recomputed by hand AND independently in numpy:**
   - eg 1 (lecture): $C = \frac23[[1,1],[1,1]]$, $\lambda_1 = 4/3$, $u_1 = (1,1)/\sqrt2$, projections $=$ data, $J^* = 0$ ✓.
   - eg 2 (new): $C = [[1.25,0.75],[0.75,1.25]]$, $\lambda = 2, 0.5$, reconstructions $(1.5,1.5),(1.5,1.5),(3.5,3.5),(3.5,3.5)$, $J^* = 0.5$, $80\%$ variance ✓.
   - eg 3 (tutorial): $\lambda_1 = 28/3$ (see fix 1), reconstructions $=$ data, $J^* = 0$ ✓.
   - Kernel eg: $K$, $K^c$, eigenvalues $8, 8/3, 0$ → $\lambda_1 = 8/3, \lambda_2 = 8/9$, $\alpha_1 = (-1/4,0,1/4)$, coords $(-2,0,2)$, variance $8/3$ ✓.
   - Solutions Problems 1, 2, 4, 7, 8 verified term by term (including $\alpha_1^TK^c\alpha_1 = 1$).
3. **Cross-references verified against the actual files** (§-style, chapter-23 scheme): §22.6, §22.11, §22.12, §22.14-promise, §23.9, §23.11, §5.4, §6.11, §7.9, §19.1 — all exist with the cited content.
4. **Notation:** lectures' subscript convention ($x_i$) adopted and flagged vs §22.6's superscripts; $m$ used for target dimension (MLF) with $k = m$ noted for the tutorial's notation.

## What was fixed during review

1. **Tutorial Example 1 eigenvalue error (source bug, corrected in chapter):** the slide claims $\lambda_1 = 14$ for $C = \frac23[[1,2,3],[2,4,6],[3,6,9]]$ — it dropped the $\frac23$ factor. True value: $\frac23\|(1,2,3)\|^2 = 28/3$ (numpy confirms). Eigenvectors/projections in the slide are correct; only the eigenvalue was wrong. Chapter presents $28/3$ and notes the slip.
2. **My own error in the "tutorial slip" Note (caught on re-read):** I first wrote the corrected $J$ as $6 = \lambda_2$ for Example 2 — wrong ($6 \ne \lambda_2 = 2$). Root cause: the tutorial projects *uncentered* points while using a *centered* $C$. Done consistently with the mean term, reconstructions are $(2,2),(0,2),(2,2),(4,2)$, errors $4,0,4,0$, $J^* = 2 = \lambda_2$ ✓ (numpy confirms). Note rewritten; the tutorial's arithmetic "$\frac14[0^2+2^2+2^2+2^2] = 24$" is wrong twice (bracket $= 12$; and the $x_3$ term is $16$, not $2^2$).
3. **Sign inconsistency in kernel eg:** $\beta_1 = (1,0,-1)/\sqrt2$ but $\alpha_1$ written as $(-1/4,0,1/4)$ — fixed to $\beta_1 = (-1,0,1)/\sqrt2$ so $\alpha_1 = \beta_1/\sqrt8 = (-1/4,0,1/4)$ holds exactly; compressed coords $(-2,0,2)$ unchanged. Solutions Problem 8 updated to match.
4. Transcript OCR garbles ($J = \frac1n\sum\|x_i - \bar x\|^2$ should be $\|x_i - \tilde x_i\|$; $\tilde x_i$ formula fragments) were cross-checked against the tutorial's clean Step 8 and not propagated.

## Thin / unsourced points (flagged, not invented)

- **"Explained variance" / scree plots:** neither term appears in the sources. §24.8 presents only the ratio of sourced quantities (projected variance $\sum_1^m\lambda_j$ ÷ trace) and explicitly notes the lectures never name it; the scree plot was omitted entirely.
- **Elbow method:** same status — presented as a way to read the eigenvalue tail, flagged as unnamed in the lectures.
- **Choosing $m$ by validation:** noted as inapplicable (unsupervised, §22.11) rather than inventing a procedure.
- **GMMs/EM:** kept to a one-paragraph pointer in §24.11; no content bleed into Chapter 25.
- **SVD-PCA connection:** the MLF lecture flags it without deriving; chapter says exactly that and points to Chapter 7.
- **RBF kernel's infinite-dimensional $\phi$:** kept to the slide's own informal framing ("technicalities aside").

## Style check

- No emojis; all math terms in LaTeX; "Basically, ..." simplification under every complex topic; punchy `=` definitions, i)/ii)/iii) points, `Note:` callouts, worked `eg` blocks with full steps — matches STYLE.md and chapter 23's voice.
- Three original matplotlib figures in `chapters/assets/` (marked original in HTML comments); no external URLs used, so no link verification was needed.
