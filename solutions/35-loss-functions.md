# Solutions — Chapter 35: Loss functions for classification

*Full worked solutions to the Chapter 35 problem set. Every number recomputed independently (hand algebra cross-checked in numpy; see the review log).*

## Problem 1 — Margin and 0/1

$m = ys$:

**(i)** $A$: $(+1)(1.5) = \boxed{1.5}$; $B$: $(+1)(-0.4) = \boxed{-0.4}$; $C$: $(-1)(-2.2) = \boxed{2.2}$; $D$: $(-1)(0.8) = \boxed{-0.8}$; $E$: $(+1)(0.2) = \boxed{0.2}$.

**(ii)** $\ell_{0/1} = \mathbf{1}(m < 0)$: $A: 0$, $B: 1$, $C: 0$, $D: 1$, $E: 0$. Average $= \boxed{2/5 = 0.4}$.

## Problem 2 — Hinge's margin demand

**(i)** $\max(0, 1-m)$: $A$: $\max(0, -0.5) = \boxed{0}$; $B$: $\max(0, 1.4) = \boxed{1.4}$; $C$: $\max(0, -1.2) = \boxed{0}$; $D$: $\max(0, 1.8) = \boxed{1.8}$; $E$: $\max(0, 0.8) = \boxed{0.8}$. Total $= 0 + 1.4 + 0 + 1.8 + 0.8 = \boxed{4.0}$ (mean $0.8$).

**(ii)** $\boxed{E}$: $m = 0.2 > 0$ (correct) but charged $0.8$.

**(iii)** The SVM doesn't ask "right side?" — it asks "margin $\ge 1$?"; anything short pays the $\xi_i$ bribe of §33.3.

## Problem 3 — Squared loss's confidence tax

**(i)**
- $m = 2.5$: $(2.5-1)^2 = \boxed{2.25}$; hinge $\max(0, -1.5) = \boxed{0}$.
- $m = -1$: $(-1-1)^2 = \boxed{4}$; hinge $\max(0, 2) = \boxed{2}$.
- $m = 0.2$: $(0.2-1)^2 = \boxed{0.64}$; hinge $\max(0, 0.8) = \boxed{0.8}$.

**(ii)** Squared charges more at $\boxed{m = 2.5}$ ($2.25 > 0$) and $\boxed{m = -1}$ ($4 > 2$); at $m = 0.2$ it charges less ($0.64 < 0.8$).

**(iii)** Squared loss is a regression loss: it wants the score to *equal* the label, so $m = 2.5$ ("too right") counts as an error.

## Problem 4 — Logistic, two faces

**(i)** $L_{\mathrm{nll}}(g, y) = -\big(y\log g + (1-y)\log(1-g)\big)$:
- $(0.8, 1)$: $-\log 0.8 = \boxed{0.2231}$ (to 4 s.f.).
- $(0.2, 0)$: $-\log(1 - 0.2) = -\log 0.8 = \boxed{0.2231}$.

**(ii)** $s = \sigma^{-1}(g) = \log\frac{g}{1-g}$:
- Case 1: $s = \log\frac{0.8}{0.2} = \log 4 = 1.3863$; $y = +1$ so $m = 1.3863$; $\log(1+e^{-1.3863}) = \log(1+0.25) = \log 1.25 = \boxed{0.2231}\ \checkmark$.
- Case 2: $s = \log\frac{0.2}{0.8} = -\log 4 = -1.3863$; $y = -1$ so $m = (-1)(-1.3863) = 1.3863$; same $\boxed{0.2231}\ \checkmark$.

Both notations bill identically — the margin form absorbs the $\{0,1\}\!\leftrightarrow\!\{\pm 1\}$ conversion.

## Problem 5 — Perceptron = modified-hinge SGD

**(i)** Misclassified $\Rightarrow -y\,w^T x > 0$, so the $\max$ drops: $\ell(w) = -y\,w^T x$. Then $\boxed{\nabla_w \ell = -yx}$.

**(ii)** One SGD step with step size $1$:
$$w \gets w - 1\cdot\nabla_w\ell = w - (-yx) = \boxed{w + yx},$$
the perceptron update (the MLT slide's form; §31.4's $(y^{(i)} - \hat y^{(i)}) = \pm 2$ on mistakes folds the factor of $2$ into the step size).

**(iii)** Correctly classified $\Rightarrow -y\,w^T x \le 0 \Rightarrow \ell = 0 \Rightarrow$ gradient $0 \Rightarrow$ **no update** — §31.4's "learn only when wrong."

## Problem 6 — AdaBoost weights from exponential loss

**(i)** Split on $y_i G(x_i) = \pm 1$; with normalized weights the correct mass is $1-\mathrm{err}$, the wrong mass $\mathrm{err}$:
$$C(\beta) = e^{-\beta}(1-\mathrm{err}) + e^{\beta}\,\mathrm{err}.$$
$$\frac{dC}{d\beta} = -e^{-\beta}(1-\mathrm{err}) + e^{\beta}\,\mathrm{err} = 0 \iff e^{2\beta} = \frac{1-\mathrm{err}}{\mathrm{err}} \iff \boxed{\beta^\star = \tfrac12\ln\frac{1-\mathrm{err}}{\mathrm{err}}}.$$

**(ii)** $w_i \gets w_i\,e^{-\beta^\star y_i G(x_i)}$: correct ($y_iG = +1$) gets $\times e^{-\beta^\star}$; wrong ($y_iG = -1$) gets $\times e^{+\beta^\star}$. The course's $\alpha_m$ (§34.6(iii)) equals $\beta^\star$, so this is exactly **misclassified $\times e^{+\alpha_m}$, correct $\times e^{-\alpha_m}$** — §34.6(iv). $\checkmark$

**(iii)** Step 2 of §35.11: for any fixed $\beta > 0$, the exponential-loss criterion over $G$ equals $(e^{\beta}-e^{-\beta})\sum_i w_i\mathbf{1}(y_i \ne G(x_i))$ plus a constant — so minimizing it over $G$ *is* minimizing the weighted error.

## Problem 7 — True or false

**(i)** **False.** Kink at $m = 1$: the subgradient there is the whole interval $[-1, 0]xy$ (MLT slide), not a single derivative.

**(ii)** **False.** Counterexample $m = -0.5$: $\log(1+e^{0.5}) = \log(2.6487) \approx \boxed{0.9741} < 1 = \ell_{0/1}$. (Hinge, squared, and exponential *do* upper-bound 0/1 everywhere — verified on a grid in the review log.)

**(iii)** **False.** $\max(0, 1-1) = \boxed{0}$ — margin exactly $1$ is precisely where the hinge turns off.

**(iv)** **True.** ESL (14.16): $f^\star(x) = \tfrac12\log\frac{P(Y=1\mid x)}{P(Y=-1\mid x)}$ — half the log-odds — so $\operatorname{sign}(f_M)$ is a legitimate classification rule.

**(v)** **True** — as stated (not proved) in the MLT Week-12 slides.

**(vi)** **False.** ESL §14.5: $e^{-Yf}$ "is not a proper log-likelihood, since it is not the logarithm of any probability mass function."
