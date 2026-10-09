# Review log — Chapter 23: Linear and polynomial regression

Branch: `draft/linear-polynomial-regression`. Reviewer: chapter author (self-review pass, 2026-10-09).
Sources re-read before writing: `PPT/Week 4/1. Linear and Polynomial Regression.pdf`
(scanned images — pdftotext yields nothing; content recovered from the transcript),
`Transcripts/Week 4/1.Linear and Polynomial Regression.pdf` (full read — the primary
authority), and `Notes/Notes by Sejal/Week_4.pdf` (corroborates the loss, feature matrix,
normal equations, full-rank closed form, and the least-squares = MLE line).

## What was checked

1. **Every worked example recomputed independently** (exact rational arithmetic + NumPy,
   all match the chapter): eg 1 — $\hat\theta_0 = 53/335 \approx 0.158$,
   $\hat\theta_1 = 649/335 \approx 1.937$, squared residual sum $306/1675 \approx 0.1827$,
   $L = 0.0365 < 0.064 = L(f{=}2x)$; eg 2 — $\det(A^TA) = 6$ vs $\det(B^TB) = 0$;
   eg 3 — $A^TA = [[3,3,5],[3,5,9],[5,9,17]]$, $A^TY = (11,17,31)^T$, $(1,1,1)$ verified
   against all three equations; eg 4 — train/test MSE table (2.863/0.977, 0.0229/0.0030,
   0.0000/0.0191) reproduced exactly. Solutions: P1 $L = 0$; P2 normal-equation form
   checked; P4 $\hat\theta = (1/6, 3/2)^T$, squared error $1/6$; P6
   $\tilde A^T\tilde A\theta = (5,5,9)^T$ solved by $(2,-2,1)^T$; P8 — det $6$, rank $2$,
   det $0$.
2. **Every § cross-reference verified programmatically** (script extracted all `## X.Y`
   headers from `chapters/*.md` and all § refs from the chapter and solutions file):
   43 refs in the chapter + 11 in solutions, 0 missing — §4.6/§4.7/§4.8, §5.7/§5.8,
   §10.1/§10.7/§10.8/§10.9, §12.7/§12.9, §20.8, §22.5/§22.6/§22.8/§22.10/§22.14, and
   all §23.x self-refs real. Forward refs to Chapters 24/28/41 are plain prose,
   matching the ch-22 review convention (those chapters aren't written yet).
   Problem set (10) and solutions file (10) numbering match.
3. **Scope discipline**: no full ridge/lasso treatment (§23.10 is the lecture's own
   5-minute remark kept as a pointer; Chapter 28 does the real work), no PCA (Ch 24),
   no gradient-descent section (sources don't cover GD for least squares — only a
   prose pointer in §23.11 to §10.7/§10.9). Overfitting section (§23.9) is built from
   the lecture's overfitting remark + §22.10's train/validation/test discipline —
   synthesis, not lecture content, and framed as such.
4. **Derivations re-checked by hand**: the component-wise gradient
   ($\partial L/\partial\theta_k = [A^T(A\theta-Y)]_k$), the null-space proof
   (both directions), the PSD-Hessian global-minimum argument, the MLE→least-squares
   reduction, and the ridge homework ($\|Az\|^2 + \lambda\|z\|^2 > 0$).
5. **Figures inspected**: `assets/23-least-squares-fit.png` (5 points, fitted line,
   dashed residuals) and `assets/23-polynomial-overfit.png` (degree 1/2/4 fits).
   Both original matplotlib, marked as original in HTML comments. No external image
   URLs used (nothing needed from d2l.ai/mml-book here).
6. **Style**: no emojis; all math terms in LaTeX; `=` definitions, i)/ii)/iii) points,
   `Note:` callouts, worked `eg` blocks, 11 "Basically, ..." simplifications.

## What was fixed during review

- §23.1 Note: bias "becomes $\theta_1$" → "$\theta_0$" (chapter uses 0-based indexing).
- eg 2(ii): "$\theta_1 + 2\theta_2$ fixed" → "$\theta_0 + 2\theta_1$ fixed" (same reason).
- §23.9: fixed validation/test conflation — eg 4's held-out data is now labeled as
  playing the validation role, with a third test split called out as the real-deployment step.
- §23.10: the $\lambda$ vs $2\lambda$ inconsistency resolved — objective keeps the
  lecture's $+\lambda\|\theta\|^2$, and a one-line reparameterization note recovers the
  lecture's $(A^TA + \lambda I)\theta_{\text{reg}} = A^TY$.

## Thin / contradictory source points

- **No numeric worked examples in the lecture.** The transcript contains zero hand-worked
  numbers — every `eg` block and problem number in this chapter is original and was
  verified by exact arithmetic (above), not taken from any source.
- **Ridge factor-of-2 wrinkle.** The transcript states the regularized objective with
  $+\lambda\|\theta\|^2$ but the resulting equations as $(A^TA + \lambda I)\theta = A^TY$
  (which strictly needs $+(\lambda/2)\|\theta\|^2$). Handled in-text by absorbing the 2
  into $\lambda$; the lecture's equation is preserved verbatim.
- **Overfitting coverage is one sentence.** The lecture mentions overfitting only to
  motivate the ridge remark ("too small $\lambda$ → overfitting, too large $\lambda$ →
  underfitting"). §23.9's degree-ladder demo is an original illustration of that remark
  combined with §22.10 — no claim that the lecture showed it.
- **Week 4 practice-assignment Q14 (polynomial regression) is unusable**: only the
  solutions PDF (`MLF_PAS_W04.pdf`) exists in the dump; the question deck is missing,
  and the extracted matrix is too garbled to reconstruct the data reliably. Not used.
- **Gradient descent for least squares is absent from Week 4 sources** — deliberately
  kept as a prose pointer (§23.11) rather than a section.
