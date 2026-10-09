# Review log — Chapter 35: Loss functions for classification

## Sources actually used (verified by content, not folder name)

- **MLT Week-12 PPT slides** (`MLT-20261009T001607Z-1-001/MLT/PPT/Week 12/slide1.pdf`, 5 pp.; `slide2.pdf`, 2 pp.): both image-only (Xournal++ exports), rendered with `pdftoppm -png -r 100` and read as images (/tmp/w12/slide1-*.png, slide2-*.png). Contents used: the loss-function view (performance measure → 0/1 minimization is **NP-HARD**); squared loss $(g(x)-y)^2 = (g(x)y-1)^2$ as "Alg 1: using regression for classification"; the 0/1-vs-squared plot; the SVM derivation $\xi_i = \max(0, 1-(\hat w x_i)y_i)$ → $\min_w \tfrac12\lVert w\rVert^2 + C\sum_i \max(0, 1-(\hat w x_i)y_i)$ ("regularization + hinge loss", model-dependent + data-dependent); the four-curve plot (0/1, squared, hinge, logistic); the closing conclusions ("0-1 loss is NP-hard to minimize", "different algorithms use different 'surrogate' loss", "surrogates are convex and hence easy to minimize"); the logistic-loss derivation with $z_i\in\{0,1\}\!\leftrightarrow\!y_i\in\{\pm1\}$ collapsing both cases to $\boxed{\log(1+e^{-\hat w_i x_i y_i})}$; the perceptron subgradient ($-xy$ on mistakes, $0$ elsewhere, $[-1,0]xy$ at the kink) and "perceptron can be interpreted as SGD with modified hinge loss with step size = 1"; the boosting bullet $\text{Loss} = e^{-y\,h(x)}$ ("exponential loss"); slide 2's neural-net diagram ending in "Cross-Entropy Loss" (used only as the §35.13 forward pointer to Chapter 41).
- **MLT Week-12 practice assignment** (`MLT-.../MLT/Practice Assignment/Week_12.pdf`): Q1 (0/1 loss on a dataset, answer 0), Q2 (squared loss, answer 7) — cited in §35.5; Q3–Q8 are neural-net material for Chapter 41, not used.
- **Ashish Tendulkar Week-12 slides** (`MLT Week 12 Slides ANN.pdf`): neural-networks content — out of scope for this chapter (belongs to Chapter 41), not used.
- **MITx 6.036 notes** (logistic regression): via Chapter 31's §§31.9–31.10 ($L_{\mathrm{nll}}$ definition, cross-entropy = NLL, the $(g-y)$ gradient, separable-data weight blow-up without regularization) — re-derived in margin form from the MLT slide, not copied.
- **ESL ch. 14** (this dump's edition numbers it Chapter 14, "Boosting and Additive Trees"): §14.2 (additive expansion), §14.4 (AdaBoost.M1 ≡ forward stagewise additive modeling with exponential loss (14.8); the (14.9)–(14.15) derivation: $w_i^{(m)} = e^{-y_i f_{m-1}(x_i)}$, weighted-error minimization (14.10)–(14.11), $\beta_m = \tfrac12\log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}$ (14.12), $w_i^{(m+1)} = w_i^{(m)}e^{-\beta_m y_i G_m(x_i)}$ (14.14), the $\alpha_m = 2\beta_m$ rewrite (14.15)); the Figure-14.3 note ("AdaBoost is not optimizing training-set misclassification error; the exponential loss is more sensitive to changes in the estimated class probabilities"); §14.5 (population minimizer $f^\star = \tfrac12\log\frac{P(Y=1|x)}{P(Y=-1|x)}$ (14.16) — proof cited to Friedman et al. 2000, not shown in ESL; binomial deviance (14.18) shares the minimizer; "$e^{-Yf}$ itself is not a proper log-likelihood, since it is not the logarithm of any probability mass function"); §14.6 (margin framing: "any loss criterion … should penalize negative margins more heavily than positive ones"; exponential vs deviance = "monotone continuous approximations to misclassification loss"; deviance grows linearly vs exponential for large negative margins; deviance "far more robust in noisy settings … misspecification of the class labels"; "the performance of AdaBoost has been empirically observed to dramatically degrade in such situations"; squared error "is not a monotone decreasing function of increasing margin … not a good surrogate for misclassification error").
- Earlier book chapters as cited: §22.9 (0/1 loss def, sign(0)=+1 convention in eg 9), §22.10 (training loss ≠ goal), §30.11, §31.3 (perceptron loss, non-differentiability), §31.4 (update rule), §31.7 (flat loss, no certainty), §31.9–31.10, §31.15(iii), §33.3 ($\xi_i$ = hinge), §33.10, §34.6 (AdaBoost updates, the $\tfrac12$ Note), §34.10(iii).

## Numbers recomputed (twice: by hand + numpy)

Script: the verification run in this session (numpy; all values below confirmed to 10 s.f.).

- **§35.10 comparison table** — margins $m = [2.0, 0.5, 0.5, -1.0, -3.0]$:
  - 0/1: $[0,0,0,1,1]$, total $2$, mean $0.4$.
  - hinge: $[0, 0.5, 0.5, 2, 4]$, total $\boxed{7}$ (my first hand-add said $6.5$ — caught and corrected; see Fixes), mean $1.4$.
  - logistic: $[0.126928011, 0.4740769842, 0.4740769842, 1.3132616875, 3.0485873516]$, total $5.4369310185$, mean $1.0873862037$.
  - squared: $[1, 0.25, 0.25, 4, 16]$, total $21.5$, mean $4.3$.
  - exponential: $[0.1353352832, 0.6065306597, 0.6065306597, 2.7182818285, 20.0855369232]$, total $24.1522153543$, mean $4.8304430709$.
- **eg 1**: $1/3$. **eg 2**: $0, 1.2, 0.7$. **eg 3**: squared $2.25, 4, 0.64$ vs hinge $0, 2, 0.8$. **eg 4**: $-\log 0.8 = 0.2231435513$; $\sigma^{-1}(0.8) = \log 4 = 1.3862943611$; $e^{-1.3863} = 0.25$ exactly (so $\log 1.25 = -\log 0.8$ algebraically, not just numerically). **eg 5**: $e^{-1.5} = 0.2231301601$, $e^{0.2} = 1.2214027582$, $e^{-0.3} = 0.7408182206$.
- **Problem 1**: margins $1.5, -0.4, 2.2, -0.8, 0.2$; 0/1 $= 2/5 = 0.4$. **Problem 2**: hinge $0, 1.4, 0, 1.8, 0.8$, total $4.0$; only $E$ correct-but-charged. **Problem 3**: as eg 3. **Problem 4**: both $0.2231435513$; margins both $1.3862943611$. **Problem 7(ii)**: $\log(1+e^{0.5}) = 0.9740769842 < 1$ ✓ (counterexample stands).
- **Upper-bound grid check** (161 margins in $[-4,4]$): hinge $\ge$ 0/1 everywhere ✓; squared $\ge$ 0/1 everywhere ✓; exponential $\ge$ 0/1 everywhere ✓; logistic $\ge$ 0/1 **fails** (False) — the §35.9 Note's claim verified, not assumed.

## Derivations double-checked

- Squared-loss margin form: $(g-y)^2 = g^2+y^2-2gy = (gy)^2+1-2gy = (gy-1)^2$ using $y^2 = 1$ ✓ (matches the slide's boxed $(g(x)\cdot y - 1)^2$).
- Logistic $z_i = 0$ case: $-\log(1-\sigma(s)) = -\log\frac{e^{-s}}{1+e^{-s}} = s + \log(1+e^{-s}) = \log(1+e^{s})$ ✓ collapses with the $z_i = 1$ case to $\log(1+e^{-y_i s_i})$ ✓.
- Logistic convexity: $\ell''(m) = \sigma(-m)(1-\sigma(-m)) > 0$ ✓ (supports §35.9(iii)).
- ESL (14.11) split: $e^{-\beta}\sum_{\text{correct}} + e^{\beta}\sum_{\text{wrong}} = e^{-\beta}\sum_{\text{all}} + (e^{\beta}-e^{-\beta})\sum w_i\mathbf{1}(\text{wrong})$ ✓.
- $\beta^\star$ calculus: $C(\beta) = e^{-\beta}W(1-\mathrm{err}) + e^{\beta}W\,\mathrm{err}$, $C' = 0 \iff e^{2\beta} = (1-\mathrm{err})/\mathrm{err}$ ✓.
- Course-vs-ESL $\alpha$: ESL's $\alpha_m = 2\beta_m$ with update $\times e^{\alpha_m\mathbf{1}(\text{wrong})}$ (common $e^{-\beta_m}$ normalizes away) ≡ course's $\times e^{\pm\alpha_m^{\text{course}}}$ with $\alpha_m^{\text{course}} = \beta_m$ — same wrong/correct ratio $e^{2\beta_m}$ ✓ (consistent with §34.6's Note).

## Cross-references verified against the actual files

- Script-checked: every `§x.y` cited in the chapter resolves to a real `## x.y` header in the book (no missing sections).
- The four forward promises confirmed present: §30.11 ("Classification losses get their own chapter (Chapter 35)"), §31.15(iii) ("Classification losses get their own chapter (Chapter 35); $L_{\mathrm{nll}}$ was the first specimen"), §33.10 ("Chapter 35 (loss functions) gives the hinge loss $\max(0, 1 - y\cdot\text{score})$ — the $\xi_i$ of §33.3"), §34.10(iii) (exponential-loss story "belongs with the loss functions, next chapter"). All four are marked kept in §35.13.
- Quotes verified against source files: §31.7's "have the same $J$ value" / "which makes it difficult to design an algorithm that searches through the space of hypotheses for a good one"; §31.4's "learn only when wrong" (section title); §34.6's update formulas and the $\tfrac12$ Note; ESL quotes re-checked against the extracted PDF text.

## Thin / contradictory source points (flagged, not invented)

1. **NP-hardness of 0/1 minimization** is the MLT slide's bare claim (red "NP-HARD" annotation) — no proof in the sources; the chapter reports it as the slides' claim (§35.1(iii), §35.12(iii)), not a theorem.
2. **0/1 at $m = 0$**: the MLT slide uses strict $\mathbf{1}(w^Txy < 0)$; §22.9's eg 9 uses $\operatorname{sign}(0) = +1$. The chapter follows the slide and carries an explicit `Note:`; no worked example has $m = 0$, so they agree everywhere used.
3. **The $f^\star = \tfrac12\log$-odds proof** is cited by ESL to Friedman et al. (2000), not shown — the chapter states the result with the citation and marks the gap honestly (§35.11(i)).
4. **Logistic loss is not an upper bound of 0/1** (dips to $0.9741$ at $m=-0.5$) — the MLT slide's plot suggests "surrogates sit above the step"; the chapter corrects this with a computed counterexample (§35.9 Note, Problem 7(ii)). Verified numerically, not asserted.
5. **Perceptron update factor of 2**: §31.4's rule gives $w \pm 2\alpha\phi(x)$ on mistakes; the MLT slide's SGD derivation gives $w + x_i y_i$. The chapter notes the $2$ is absorbed into the step size (§35.8, solution P5) rather than pretending they are textually identical.
6. **AdaBoost noise sensitivity** was deliberately *not* claimed in Chapter 34 (no source had been read); ESL §14.6 *does* document it ("performance of AdaBoost has been empirically observed to dramatically degrade" under label misspecification), so this chapter states it with the ESL citation (§35.7, §35.12(ii)) — no conflict, the source base grew.
7. **ESL Figure-14.4's caption** writes the hinge as "$(1-yf)\cdot\mathbf{1}(yf > 1)$" — almost certainly a typo for $\mathbf{1}(yf < 1)$; the chapter uses $\max(0, 1-m)$ per the MLT sources and does not quote the caption.
8. **Figures**: `chapters/assets/35-loss-curves.png` is an original matplotlib plot (script in /tmp: `fig_loss_curves.py`); squared/exponential clipped at $8$ for display (noted in code); comparison margins marked on the hinge curve. HTML comment in the chapter marks it original.

## Fixes made during self-review

1. **Hinge total in §35.10**: first hand-addition gave $6.5$; numpy said $7.0$ ($0+0.5+0.5+2+4$). Corrected to $7$ (mean $1.4$) before writing.
2. **§35.7 robustness paragraph**: first draft contained a mid-sentence self-correction ("It is therefore far more robust" — no: the other way). Rewrote as a clean sourced statement: exponential concentrates influence on large negative margins; the *deviance* is the robust one; AdaBoost empirically degrades under label misspecification.
3. **§35.2 overclaim**: "all agree positive good, negative bad" ignored squared loss's non-monotonicity; softened to "all agree negative margins are mistakes; they disagree on how to price size, and even on whether bigger positive is always better (§35.5's squared loss is the rebel)."
4. **§35.4 Note**: "the standard Chapter-12-style fix" — Chapter 12 never mentions subgradients; rephrased to "the standard replacement for a derivative at a convex kink" with no chapter claim.
5. **Figure annotations**: first version had overlapping "decision boundary m=0" / "margin m=1" text; rotated and repositioned them.

## Coordinator review fixes (pre-merge, 2026-10-09)

1. **§35.4 Note — hinge subgradient was the perceptron's.** The Note printed the slide's "$-xy$ on mistakes ($w^Txy < 0$), $0$ elsewhere, kink at $w^Txy = 0$" bullet as the hinge's subgradient. But that bullet was the slide's *perceptron* (modified-hinge) statement — and it contradicts §35.4's own Def $\max(0, 1-m)$, whose kink is at $m = 1$ and whose active region is $m < 1$ (including correct-but-shaky points). Corrected to: subgradient $-xy$ when $w^Txy < 1$, $0$ when $w^Txy > 1$, interval $[-xy, 0]$ at the kink $w^Txy = 1$; attribution corrected. (Solutions' P7(i) already had the kink at $m = 1$ — no change needed there.)
2. **§35.13 "Part V picks the thread straight back up" → "Part VI".** Chapter 41 (artificial neuron/MLPs from scratch, per OUTLINE.md) is in Part VI, not Part V (which is the MLP-code-practice part, chs 36–40). The cross-entropy forward pointer itself is correct; only the part label was wrong.
