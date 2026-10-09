# Review log — Chapter 31: Perceptron and logistic regression

Branch: `draft/perceptron-logistic`. Reviewer: chapter-writing subagent (self-review, two passes).

## Sources actually read (content-verified, not folder-name-verified)

1. `MLT/Slides - Ashish Tendulkar/Week 4/MLT Week 4 Slides - Perceptron.pdf` — full text extracted via `pdftotext` (303 lines). Contains: Rosenblatt 1958 neuron motivation; $y \in \{-1,+1\}$; model $\hat y = \operatorname{sign}(w^T\phi(x))$ with sign/threshold activation; loss $e^{(i)} = \max(0, -w^T\phi(x^{(i)})y^{(i)})$, "not differentiable in $w$"; update rule $w^{(t+1)} := w^{(t)} + \alpha(y^{(i)}-\hat y^{(i)})\phi(x^{(i)})$ with the three cases worked out; "Linear separable examples lead to convergence of the algorithm with zero training loss, else it oscillates"; confusion matrix / precision / recall / accuracy / F1 evaluation. (Outline said "MLT W9"; real content was in the "Week 4" folder — week numbering unreliable, as warned.)
2. `MLT/Slides - Ashish Tendulkar/Week 4/MLT Week 4 Slides - Models of Classification.pdf` — full text extracted (677 lines). Contains: generative-vs-discriminative definitions; linear discriminant $y = w_0 + w^Tx$, boundary $w_0 + w^Tx = 0$; $w$ = orientation (orthogonal to surface), $w_0$ = location (normal distance $-w_0/\lVert w\rVert$); dummy-$x_0=1$ bias trick; one-vs-rest / one-vs-one and their ambiguity regions; the $k$-discriminant fix with boundary $(w_{k0}-w_{j0}) + (w_k-w_j)^Tx = 0$.
3. `MLT/Practice Assignment/Week_9.pdf` — full text extracted. Q1 (linear separability), Q2 (perceptron oscillates forever on non-separable 3-point data — "will keep oscillating and repeating these weights, and will never converge"), Q3 (squared-length growth: $R=4$, $\lVert w\rVert^2=36$ → answer 45, i.e. next $\le 36+16=52$), Q4 (mistake bound: $R=4$, $\gamma=1$ → at most 16 mistakes, answer (c)), Q5 (16 labelings of unit-square corners, 14 linearly separable), Q6 (negate $w$ to flip all predictions), Q7 (XOR not separable), Q8 (none separable), Q9 (logistic regression: $z=10$ vs $z=-10$ → first probability "much higher").
4. `MLT/Notes/MITx - 6.036 Machine Learning/book.pdf` — logistic-classification + gradient-descent chapters extracted. Contains: $h(x;\theta,\theta_0) = \sigma(\theta^Tx+\theta_0)$, $\sigma(z)=1/(1+e^{-z})$; motivation (categorical predictions can't express certainty; same-$J$ problem); NLL per-example $L_{\mathrm{nll}} = -(\text{actual}\cdot\log\text{guess} + (1-\text{actual})\cdot\log(1-\text{guess}))$ with the $\{0,1\}$-label requirement; objective $J_{\mathrm{lr}} = \frac1n\sum L_{\mathrm{nll}} + \frac{\lambda}{2}\lVert\theta\rVert^2$; gradients $\nabla_\theta J = \frac1n\sum(g^{(i)}-y^{(i)})x^{(i)} + \lambda\theta$, $\partial J/\partial\theta_0 = \frac1n\sum(g^{(i)}-y^{(i)})$; GD update; study question on $\sigma=0.5$ being a line; study question on separable data with $\lambda=0$ vs large $\lambda$; softmax definition; multiclass $L_{\mathrm{nllm}} = -\sum_k \text{actual}_k\log\text{guess}_k$ (+ study question: $K=2$ recovers $L_{\mathrm{nll}}$).
5. Book's own chapters: `30-naive-bayes.md` (intro promise, §§30.1–30.3, 30.5, 30.7, 30.11–30.12), `20-estimation-mle-map.md` §20.12(i) (logistic regression as MLE example), `22-what-is-ml.md` §§22.9–22.10, `10-unconstrained-optimization-gradient-descent.md` §§10.4–10.7, `28-regularization.md` (exists, for the $\lambda$ term).

## Every number recomputed (hand × numpy)

- **§31.6 perceptron example** ($m_1=(1,-2,1),-1$; $m_2=(1,1,-2),-1$; $p_1=(1,2,2),+1$; $\alpha=1$): hand trace → updates $(0,0,0)\to(-2,4,-2)\to(-4,2,2)$, sweep-2 scores $(-6,-6,4)$, boundary $x_1+x_2=2$. Numpy independently: identical weight history, final scores `[-6. -6. 4.]`. **Caught and fixed:** my first hand pass scored $m_1$'s sweep-2 $z$ as $-10$; the numpy recompute showed $-6$ ($-4(1)+2(-2)+2(1) = -6$). Chapter text carries the corrected $-6$.
- **§31.11 GD step** ($x\in\{0,2\}$, $y\in\{0,1\}$, $\theta_0=\theta=0$, $\eta=1$): hand → $J_0=\log2\approx0.6931$, grads $(0,-1/2)$, new $(0,1/2)$, $g=(0.5,0.7311)$, $J_1\approx0.5032$. Numpy: $J_0=0.69314718$, grad $(0.0,-0.5)$, $J_1=0.50320443$. Match.
- **Problem 1** ($a=(1,0,2),+1$; $b=(1,2,0),-1$; $c=(1,-1,-1),-1$): hand and numpy → updates $(-2,-4,0)$, $(-4,-2,2)$; converged sweep 2; scores $(0,-8,-4)$; boundary $x_2=x_1+2$; 2 mistakes. Match.
- **Problem 5** ($(1,0),(3,1)$, $\eta=1$): hand and numpy → $J_0=0.69314718$, grad $(0,-0.5)$, new $(0,0.5)$, $g=(0.6225,0.8176)$, $J_1=0.58774513$. Match.
- **Sigmoid spot values:** $\sigma(10)\approx0.99995$, $\sigma(-10)\approx4.54\times10^{-5}$, $\sigma(0.5)\approx0.62246$, $\sigma(1.5)\approx0.81757$, $\log9\approx2.197$ — numpy-confirmed; chapter rounds consistently.

## Derivations double-checked

- $\sigma'(z)=\sigma(z)(1-\sigma(z))$: re-derived by quotient rule; the $e^{-z}/(1+e^{-z})^2$ factorization verified.
- $\partial L_{\mathrm{nll}}/\partial z = g-y$: the $g(1-g)$ cancellation re-done by hand; matches the 6.036 book's stated gradient exactly (both $\nabla_\theta J$ and $\partial J/\partial\theta_0$).
- $\sigma(-z)=1-\sigma(z)$ and $\sigma(z)>1/2\iff z>0$: re-proved (used in §31.8 and Problem 4).
- NB log-odds vs LR sigmoid → same "linear score $>0$" rule (Problem 6): both directions re-checked.

## Cross-references verified against the actual files

- §30 intro/"generative counterpart"/"Chapter 31 is the discriminative half" ✓ (ch30 line 1); §30.1 (discriminative def) ✓; §30.3(ii) ($2d+1$ params) ✓; §30.5 ($w^Tx+b>0$) ✓; §30.7 (counting) ✓; §30.11 (Ch 35 losses) ✓; §30.12 (SVMs Ch 32–33, ensembles Ch 34) ✓.
- §20.12(i) lists logistic regression as MLE example ✓; §20.12(ii) ridge = MAP/Gaussian prior ✓; §15.11 Bernoulli ✓ (via ch30's citation); §§22.9 (classification), 22.10 (test data) ✓; §§10.4 (step size), 10.5/10.7 (GD) ✓; ch28 exists ✓; Part VI = neural nets ✓ (per §20.12(i) itself).
- Figure paths `assets/31-perceptron-boundary.png`, `assets/31-sigmoid.png` resolve relative to `chapters/` (same pattern as ch30) and both files exist; HTML comments mark them original.

## Thin / contradictory source points (flagged, not invented)

1. **Logistic regression is thin in the IITM MLT slides** — no dedicated slides found (full-repo grep for "logistic" hit only the practice Q9 one-liner plus the two books). The chapter's logistic-regression core (§§31.7–31.12) is therefore sourced from the **MITx 6.036 book in the MLT Notes folder** — a course-supplied reference, not the lecture slides. This sourcing is stated in the chapter intro.
2. **No formal perceptron convergence proof in the slides** — only the statement (separable → convergence with zero training loss; else oscillation) and the assignment's quantitative bound $(R/\gamma)^2$. The chapter presents exactly this and no more (see §31.5's scope Note).
3. **Generative-vs-discriminative data-efficiency claim omitted** — the Ng & Jordan "generative needs less data / discriminative has lower asymptotic error" comparison is standard but appears nowhere in the sources, so §31.13 restricts itself to sourced contrasts (what is estimated, counting vs GD, parameter counts, where the boundary comes from).
4. **Q3's squared-length bound** is presented in the assignment's intended (standard) reading — per-update growth $\le R^2$ — since the slides don't derive it; consistent with the assignment's answer (45, not 55/60/65).
5. **"Logistic regression fails on XOR"** (§31.14) is my inference from the sourced linear-boundary form, not a sourced sentence — mathematically immediate, flagged here.
6. **The bias-as-augmented-feature construction** in the perceptron worked examples is my (standard) synthesis of the perceptron slides' $\phi(x)$ with the Models-of-Classification slides' dummy-$x_0$ trick — the perceptron slides have no explicit bias term.

## Fixes made during self-review

- Corrected the §31.6 sweep-2 $m_1$ score $-10 \to -6$ (numpy recompute; see above).
- Reworded §31.12's "§31.1's source" parenthetical for accuracy (the $k$-discriminant material comes from the Models of Classification slides introduced in §31.1).
- Kept §31.10's separable-data claim to "weights blow up / no finite minimizer" (the study question's $\lambda=0$ case) without asserting the book's unstated $\lambda\to\infty$ answer beyond "regularization keeps the optimum finite".
- Verified no emojis, all math in LaTeX, every complex topic has a "Basically, ..." line, no external links (none needed).
