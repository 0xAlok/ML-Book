# Solutions — Chapter 30: Naive Bayes: generative vs discriminative models

## Problem 1 — Classify $[0, 0, 1]$ (§30.9's data, no smoothing)

Estimates from §30.9: $\hat p(y{=}1) = \hat p(y{=}0) = 1/2$; $\hat p^1 = (1, 2/3, 1/3)$; $\hat p^0 = (1/3, 2/3, 2/3)$.

(i) For $x = [0, 0, 1]$, take $(1 - \hat p_j)$ for the $0$-valued features and $\hat p_j$ for the $1$-valued one:
$$P(x \mid y{=}1) = \underbrace{(1 - 1)}_{F_1=0}\cdot\underbrace{\left(1 - \tfrac23\right)}_{F_2=0}\cdot\underbrace{\tfrac13}_{F_3=1} = 0\cdot\tfrac13\cdot\tfrac13 = \boxed{0},$$
$$P(x \mid y{=}0) = \underbrace{\left(1 - \tfrac13\right)}_{F_1=0}\cdot\underbrace{\left(1 - \tfrac23\right)}_{F_2=0}\cdot\underbrace{\tfrac23}_{F_3=1} = \tfrac23\cdot\tfrac13\cdot\tfrac23 = \boxed{\tfrac{4}{27} \approx 0.148}.$$

(ii) Unnormalized scores ($\times$ prior $1/2$): $y{=}1$: $0$; $y{=}0$: $\tfrac12\cdot\tfrac{4}{27} = \tfrac{2}{27} \approx 0.074$. Evidence $= 2/27$. Posteriors: $\boxed{P(y{=}1 \mid x) = 0}$, $\boxed{P(y{=}0 \mid x) = 1}$.

(iii) $\boxed{\text{Predict } y = 0.}$

(iv) The zero factor is $(1 - \hat p_1^1) = 1 - 1 = 0$: $F_1 = 0$ never occurs with class $1$ in training, so the MLE declares it impossible and the whole product dies. This is §30.8's **zero-count problem** — one unseen (feature, class) combination vetoes the class entirely, which is why Laplace smoothing exists.

## Problem 2 — Parameter counting ($k = 4$, $d = 5$ binary features)

(i) The full generative story needs one probability per (feature-vector, label) pair: $k\cdot 2^d - 1 = 4\cdot 2^5 - 1 = 4\cdot 32 - 1 = \boxed{127}$.

(ii) Naive Bayes (TA-notes style, priors counted straight): $k$ priors $+ k\cdot d$ conditionals $= 4 + 4\cdot 5 = \boxed{24}$.

(iii) The $k$ priors sum to $1$, so one is redundant: exact free parameters $= (k - 1) + kd = 3 + 20 = \boxed{23}$. (Each Bernoulli conditional is one free number — $\hat p$ determines $1 - \hat p$ — so no further redundancy.)

(iv) Full generative: $\boxed{\text{exponential in } d}$ — doubles with each added feature. Naive Bayes: $\boxed{\text{linear in } d}$ — $k$ more numbers per feature. That exponential-to-linear collapse is what the independence assumption buys (§30.3(ii)).

## Problem 3 — Laplace smoothing on $x = [0, 1, 1]$

(i) Unsmoothed, from §30.9's estimates:
$$P(x \mid y{=}1) = \underbrace{(1-1)}_{F_1=0}\cdot\underbrace{\tfrac23}_{F_2=1}\cdot\underbrace{\tfrac13}_{F_3=1} = \boxed{0}.$$
The $y{=}1$ score is $0$ no matter what the other features say; $P(x \mid y{=}0) = \tfrac23\cdot\tfrac23\cdot\tfrac23 = 8/27$, so the prediction is forced to $\boxed{0}$ (posteriors $0$ and $1$).

(ii) With $c = 1$ (§30.8): class $1$ ($n = 3$): $\hat p_1^1 = \frac{3+1}{3+2} = \frac45 = 0.8$, $\hat p_2^1 = \frac{2+1}{5} = \frac35 = 0.6$, $\hat p_3^1 = \frac{1+1}{5} = \frac25 = 0.4$. Class $0$ ($n = 3$): $\hat p_1^0 = \frac{1+1}{5} = \frac25 = 0.4$, $\hat p_2^0 = \frac{2+1}{5} = \frac35 = 0.6$, $\hat p_3^0 = \frac{2+1}{5} = \frac35 = 0.6$.

(iii) Smoothed likelihoods:
$$P(x \mid y{=}1) = 0.2\cdot 0.6\cdot 0.4 = \boxed{0.048}, \qquad P(x \mid y{=}0) = 0.6\cdot 0.6\cdot 0.6 = \boxed{0.216}.$$
Scores ($\times$ prior $1/2$): $0.024$ vs $0.108$; evidence $= 0.132$. Posteriors: $\boxed{P(y{=}1\mid x) = 0.024/0.132 \approx 0.182}$, $\boxed{P(y{=}0\mid x) = 0.108/0.132 \approx 0.818}$. $\boxed{\text{Predict } 0}$ — same verdict as unsmoothed here, but now class $1$ keeps a nonzero $18\%$ instead of being vetoed outright.

## Problem 4 — Spam posterior (practice-assignment numbers)

Numerator (spam, prior $0.2$):
$$0.2 \times 0.7 \times 0.2 \times 0.01 \times 0.3 = 0.2 \times 0.00042 = 0.000084.$$
Non-spam term (prior $0.8$):
$$0.8 \times 0.01 \times 0.02 \times 0.01 \times 0.1 = 0.8 \times 2\times 10^{-7} = 1.6\times 10^{-7}.$$
Evidence $= 0.000084 + 0.00000016 = 0.00008416$.
$$P(\text{spam} \mid \text{mail}) = \frac{0.000084}{0.00008416} \approx 0.99810 = \boxed{1.00}\ \text{(two decimals)}.$$
Every one of the four "spammy" words multiplies the spam case up and the ham case down; the $0.2$ prior is swamped by the likelihood ratio.

## Problem 5 — Gaussian NB boundary

(i) Class $0$: $\hat\mu_0 = \frac{1+2+3}{3} = \boxed{2}$; $\hat\sigma_0^2 = \frac{(1-2)^2 + 0 + (3-2)^2}{3} = \boxed{\tfrac23}$. Class $1$: $\hat\mu_1 = \frac{5+6+7}{3} = \boxed{6}$; $\hat\sigma_1^2 = \frac{(5-6)^2 + 0 + (7-6)^2}{3} = \boxed{\tfrac23}$.

(ii) Equal priors and equal variances: $p(x\mid 1) = p(x\mid 0) \iff (x-6)^2 = (x-2)^2$. Expanding: $x^2 - 12x + 36 = x^2 - 4x + 4$, so $-12x + 36 = -4x + 4$, $-8x = -32$, $\boxed{x = 4}$ — again the midpoint of the means.

(iii) $x = 3$: $(3-6)^2 = 9 > (3-2)^2 = 1$ → class $0$'s density is taller → $\boxed{\text{predict } 0}$. $x = 5$: $(5-6)^2 = 1 < (5-2)^2 = 9$ → $\boxed{\text{predict } 1}$.

## Problem 6 — Generative or discriminative?

- **Decision trees** — **discriminative**: model $P(y \mid x)$ directly (TA notes: $P(y{=}1 \mid x) = 1$ if $x$'s leaf is $1$).
- **KNN** — **discriminative**: the majority vote of the neighbours estimates $P(y \mid x)$ straight from the data, no model of $P(x, y)$.
- **Naive Bayes** — **generative**: models the pair $P(x, y) = P(x \mid y)P(y)$ — class priors plus per-class feature distributions — then flips with Bayes' rule.
- **Gaussian Naive Bayes** — **generative**: same as NB, with Gaussian class-conditional densities.

## Problem 7 — Why log space

(i) $P(y_c \mid x) = \dfrac{P(x \mid y_c)P(y_c)}{P(x)}$. For the $\operatorname*{argmax}$ over $y_c$, the denominator $P(x)$ is constant, so
$$\operatorname*{argmax}_{y_c} P(y_c \mid x) = \operatorname*{argmax}_{y_c}\big[P(x \mid y_c)P(y_c)\big].$$
With the naive factorization $P(x \mid y_c) = \prod_j p(x_j \mid y_c)$ and $\log$ strictly increasing (so it preserves argmax):
$$= \operatorname*{argmax}_{y_c}\left[\log\!\left(\prod_j p(x_j \mid y_c)P(y_c)\right)\right] = \operatorname*{argmax}_{y_c}\left[\sum_{j=1}^{m}\log p(x_j \mid y_c) + \log p(y_c)\right]. \qquad \blacksquare$$

(ii) Because a product of hundreds of small probabilities underflows to floating-point zero, while a sum of logs stays in a safe range — same argmax, no underflow.
