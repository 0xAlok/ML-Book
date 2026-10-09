# Solutions — Chapter 31: Perceptron and logistic regression

## Problem 1 — Perceptron by hand

Data (augmented, $y \in \{-1,+1\}$), $\alpha = 1$, $w^{(0)} = (0,0,0)$, cycling $a, b, c$:

| | $\phi(x)$ | $y$ |
|---|---|---|
| $a$ | $(1, 0, 2)$ | $+1$ |
| $b$ | $(1, 2, 0)$ | $-1$ |
| $c$ | $(1, -1, -1)$ | $-1$ |

**Sweep 1.**
- $a$: $z = 0 \to \hat y = +1 = y$. Correct, no change.
- $b$: $z = 0 \to \hat y = +1 \ne -1$. Mistake: $w := (0,0,0) + 1\cdot(-1-1)\cdot(1,2,0) = \boxed{(-2,\, -4,\, 0)}$.
- $c$: $z = (-2)(1) + (-4)(-1) + 0(-1) = -2 + 4 + 0 = 2 \ge 0 \to \hat y = +1 \ne -1$. Mistake: $w := (-2,-4,0) + (-2)(1,-1,-1) = (-2-2,\ -4+2,\ 0+2) = \boxed{(-4,\, -2,\, 2)}$.

**Sweep 2** (verify, $w = (-4,-2,2)$).
- $a$: $z = -4(1) + (-2)(0) + 2(2) = -4 + 0 + 4 = 0 \ge 0 \to +1$ ✓.
- $b$: $z = -4(1) + (-2)(2) + 2(0) = -4 - 4 + 0 = -8 < 0 \to -1$ ✓.
- $c$: $z = -4(1) + (-2)(-1) + 2(-1) = -4 + 2 - 2 = -4 < 0 \to -1$ ✓.

No update fires in sweep 2 — converged (independently confirmed in numpy: same update sequence, final scores $(0, -8, -4)$).

(i) Weight history: $(0,0,0) \to (-2,-4,0) \to (-4,-2,2)$.
(ii) Boundary: $-4 - 2x_1 + 2x_2 = 0$, i.e. $\boxed{x_2 = x_1 + 2}$.
(iii) Total mistakes (updates): $\boxed{2}$.

## Problem 2 — Mistake bound

(i) $\text{\#mistakes} \le (R/\gamma)^2 = (3/0.5)^2 = 6^2 = \boxed{36}$.

(ii) One update can increase $\lVert w \rVert^2$ by at most $R^2 = 4$, so the next squared length is at most $20 + 4 = 24$. $\boxed{21}$ and $\boxed{23}$ are possible; $25 > 24$ is not.

(iii) The bound applies only to linearly separable data — for non-separable data the perceptron makes unboundedly many updates: it oscillates forever (§31.5).

## Problem 3 — Flip the labels

Negate the weight vector: $\boxed{w_{\text{new}} = -w}$. Proof: $\hat y_{\text{new}} = \operatorname{sign}(-w^T\phi(x)) = -\operatorname{sign}(w^T\phi(x)) = -\hat y_{\text{old}}$ whenever $w^T\phi(x) \ne 0$ (given: no point sits exactly on the boundary, so the sign is never ambiguous). Every prediction flips, exactly undoing the label swap. (This is the Week 9 practice assignment's Q6, whose answer is the negation.)

## Problem 4 — Sigmoid arithmetic

(i) $\sigma(10) = 1/(1+e^{-10})$: $e^{-10} \approx 4.5400 \times 10^{-5}$, so $\sigma(10) \approx 1/(1.0000454) \approx \boxed{0.99995}$ (4 s.f.: $0.99995$). $\sigma(-10) = 1/(1+e^{10}) \approx e^{-10} \approx \boxed{4.540 \times 10^{-5}}$.

(ii) $\sigma(z) = 0.9 \iff 1/(1+e^{-z}) = 0.9 \iff 1 + e^{-z} = 10/9 \iff e^{-z} = 1/9 \iff \boxed{z = \log 9 \approx 2.197}$.

(iii) $\sigma(-z) = \frac{1}{1+e^{z}} = \frac{e^{-z}}{e^{-z}(1+e^{z})} = \frac{e^{-z}}{1+e^{-z}} = 1 - \frac{1}{1+e^{-z}} = \boxed{1 - \sigma(z)}$. Then $g > 1/2 \iff \sigma(z) > 1/2 \iff 1/(1+e^{-z}) > 1/2 \iff 1 + e^{-z} < 2 \iff e^{-z} < 1 \iff \boxed{z > 0}$ (using monotonicity of $\exp$ and $\log$).

## Problem 5 — One GD step, new numbers

$(x^{(1)}, y^{(1)}) = (1, 0)$, $(x^{(2)}, y^{(2)}) = (3, 1)$; $\theta_0 = 0$, $\theta = 0$, $\eta = 1$.

(i) $z^{(i)} = 0$, $g^{(i)} = \sigma(0) = 1/2$ both. $J = \tfrac12[-\log(1-\tfrac12) - \log(\tfrac12)] = \boxed{\log 2 \approx 0.6931}$.

(ii) Errors: $g^{(1)} - y^{(1)} = 1/2$, $g^{(2)} - y^{(2)} = -1/2$.
$$\frac{\partial J}{\partial\theta_0} = \tfrac12\left(\tfrac12 - \tfrac12\right) = \boxed{0}, \qquad \frac{\partial J}{\partial\theta} = \tfrac12\left(\tfrac12\cdot 1 + \left(-\tfrac12\right)\cdot 3\right) = \tfrac12\left(\tfrac12 - \tfrac32\right) = \boxed{-\tfrac12}.$$

(iii) $\theta_0 := 0 - 0 = \boxed{0}$; $\theta := 0 - (-\tfrac12) = \boxed{\tfrac12}$.

(iv) New scores: $z^{(1)} = \tfrac12 \to g^{(1)} = \sigma(0.5) \approx 0.6225$; $z^{(2)} = \tfrac32 \to g^{(2)} = \sigma(1.5) \approx 0.8176$.
$$J_{\text{new}} = \tfrac12\bigl[-\log(1-0.6225) - \log(0.8176)\bigr] = \tfrac12\bigl[-\log(0.3775) - \log(0.8176)\bigr] \approx \tfrac12(0.9741 + 0.2014) \approx \boxed{0.5877} < 0.6931\ \checkmark.$$
(All numbers cross-checked in numpy: $J_0 = 0.69314718$, gradient $(0, -0.5)$, $J_1 = 0.58774513$.)

## Problem 6 — Same wall, two derivations

Naive Bayes: decide $1$ iff posterior odds exceed $1$, i.e. $\log \frac{P(y=1\mid x)}{P(y=0\mid x)} > 0$. By §30.5 the log-odds equals $w^T x + b$, so: predict $1 \iff \boxed{w^T x + b > 0}$.

Logistic regression: decide $1$ iff $P(y=1\mid x) = \sigma(\theta^T x + \theta_0) > 1/2$. By Problem 4(iii), $\sigma(z) > 1/2 \iff z > 0$, so: predict $1 \iff \boxed{\theta^T x + \theta_0 > 0}$.

The genuine difference: NB's $(w, b)$ are *derived* from counted generative parameters (priors and class-conditionals, closed-form MLE, §30.7); logistic regression's $(\theta, \theta_0)$ are *fitted directly* by iterative gradient descent on the NLL of $P(y\mid x)$ (§31.10). Same wall shape, opposite estimation philosophies.

## Problem 7 — Count the knobs

(i) Naive Bayes (binary): $\boxed{2d+1}$ — one Bernoulli parameter per feature per class ($2d$) plus the prior (§30.3(ii)).

(ii) Logistic regression (binary): $\boxed{d+1}$ — the weight vector $\theta \in \mathbb{R}^d$ and the scalar bias $\theta_0$ (§31.8).

(iii) Naive Bayes is **closed-form**: training is counting (§30.7), one pass over the data, no hyperparameters. Logistic regression is **iterative**: gradient descent needs a step size $\eta$ (§10.4), an initialization, and a stopping criterion — and has no finite closed-form solution.
