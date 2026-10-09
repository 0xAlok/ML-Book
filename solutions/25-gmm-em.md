# Solutions — Chapter 25: GMMs and the EM algorithm

**1. Trimodal arithmetic.**

(i) $f_X(x) = 0.4\,N(x\mid-4,0.5) + 0.3\,N(x\mid 0,1) + 0.3\,N(x\mid 5,1)$.

At $x = -4$:
- red: $0.4\cdot\frac{1}{\sqrt{2\pi\cdot0.5}}e^{0} = 0.4\cdot 0.564190 = 0.225676$,
- green: $0.3\cdot\frac{1}{\sqrt{2\pi}}e^{-(-4)^2/2} = 0.3\cdot 0.398942\cdot e^{-8} = 0.119683\cdot 0.000335 = 0.000040$,
- blue: $0.3\cdot\frac{1}{\sqrt{2\pi}}e^{-(-9)^2/2} \approx 0$,
$$\boxed{f_X(-4) \approx 0.2257}.$$

At $x = 5$:
- blue: $0.3\cdot\frac{1}{\sqrt{2\pi}}e^{0} = 0.119683$,
- green: $0.3\cdot\frac{1}{\sqrt{2\pi}}e^{-25/2} = 0.119683\cdot e^{-12.5} = 0.119683\cdot 3.73\times10^{-6} \approx 0.000000$,
- red $\approx 0$,
$$\boxed{f_X(5) \approx 0.1197}.$$

(ii) Tallest peak at $x = -4$ ($0.2257 > 0.1197$). Each peak's height is $\approx \pi_k/\sqrt{2\pi\sigma_k^2}$ (the other components contribute negligibly at a peak): red $\frac{0.4}{\sqrt{\pi}} = 0.2257$, green $\frac{0.3}{\sqrt{2\pi}} = 0.1197$, blue the same $0.1197$. Red wins twice over: largest weight ($0.4$) *and* narrowest bell ($\sigma^2 = 0.5$ concentrates its mass into a taller spike). This is §22.13's "mass spent in one place is mass denied elsewhere" — the narrow red bell spends its $0.4$ of mass in a tight neighbourhood, so its density there is high.

**2. Labels don't matter.**

Set A: $f_A(x) = 0.5\,N(x\mid 1,1) + 0.5\,N(x\mid -1,1)$.

Set B: $f_B(x) = 0.5\,N(x\mid -1,1) + 0.5\,N(x\mid 1,1)$.

For every $x$, $f_B(x) - f_A(x) = 0$ by commutativity of addition: the two summands are identical, only their order differs. Hence $f_A \equiv f_B$ as functions — no dataset, however large, can distinguish the two parameter vectors. This is §25.4's unidentifiability.

**3. Why direct MLE is ugly.**

(i) $N(x\mid\mu,1) = \frac{1}{\sqrt{2\pi}}e^{-(x-\mu)^2/2}$, so
$$R(\mu_1,\mu_2) = -\log\!\left(\tfrac12 N(a\mid\mu_1,1)+\tfrac12 N(a\mid\mu_2,1)\right) - \log\!\left(\tfrac12 N(b\mid\mu_1,1)+\tfrac12 N(b\mid\mu_2,1)\right).$$

(ii) $\frac{\partial}{\partial\mu_1}\log N(x\mid\mu_1,1) = x - \mu_1$. By the chain rule on each log-of-sum, with $D_i = \tfrac12 N(x_i\mid\mu_1,1)+\tfrac12 N(x_i\mid\mu_2,1)$ ($x_1 = a$, $x_2 = b$):
$$\frac{\partial R}{\partial\mu_1} = -\sum_{i=1}^{2}\frac{\tfrac12\,\partial_{\mu_1}N(x_i\mid\mu_1,1)}{D_i} = -\sum_{i=1}^{2}(x_i-\mu_1)\,\frac{N(x_i\mid\mu_1,1)}{N(x_i\mid\mu_1,1)+N(x_i\mid\mu_2,1)}.$$
The fraction is exactly the responsibility (with $\pi_1 = \pi_2$ cancelling):
$$\boxed{\frac{\partial R}{\partial\mu_1} = -(a-\mu_1)\,\gamma(z_{11}) - (b-\mu_1)\,\gamma(z_{12})}.$$

(iii) Setting to zero: $\gamma(z_{11})(a-\mu_1) + \gamma(z_{12})(b-\mu_1) = 0$, i.e.
$$\mu_1 = \frac{\gamma(z_{11})\,a + \gamma(z_{12})\,b}{\gamma(z_{11}) + \gamma(z_{12})}.$$
Not a closed-form solution: $\gamma(z_{i1}) = N(x_i\mid\mu_1,1)/[N(x_i\mid\mu_1,1)+N(x_i\mid\mu_2,1)]$ itself contains $\mu_1$ (and $\mu_2$) inside the Gaussian densities — $\mu_1$ appears on both sides of the equation. It is a *fixed-point equation*, not a formula. This is precisely the lecture's point: you can write the gradient, but "you cannot set it to $0$ and solve for it". (Note the silver lining: the fixed-point equation *is* the M-step update — EM solves it by iteration, freezing $\gamma$ in the E-step and solving for $\mu_1$ in the M-step.)

**4. E-step arithmetic.**

Numerators $\pi_k P(x\mid z=k)$:
- $k=1$: $0.4\times 0.2 = 0.08$,
- $k=2$: $0.3\times 0.3 = 0.09$,
- $k=3$: $0.3\times 0.5 = 0.15$.

Denominator: $0.08+0.09+0.15 = 0.32$. Posteriors:
$$\boxed{P(z=1\mid x) = \tfrac{0.08}{0.32} = 0.25},\quad \boxed{P(z=2\mid x) = \tfrac{0.09}{0.32} = 0.28125},\quad \boxed{P(z=3\mid x) = \tfrac{0.15}{0.32} = 0.46875}.$$
Component 3 has the highest posterior — most likely to have generated $x = 6$. (Note: the largest *likelihood* alone doesn't decide; the posterior weights it by $\pi_k$. Here they agree, but in general the $\pi_k$ factor matters.)

**5. Fill the responsibilities.**

(i) Rows must sum to 1:
- Row 1: $0.3 + a + 0.5 = 1 \Rightarrow \boxed{a = 0.2}$.
- Row 2: $b + 1 + b = 1 \Rightarrow \boxed{b = 0}$.
- Row 3: $c + 0.5 + c = 1 \Rightarrow \boxed{c = 0.25}$.
- Row 4: $0.1 + 0.4 + d = 1 \Rightarrow \boxed{d = 0.5}$.

$$R = \begin{pmatrix} 0.3 & 0.2 & 0.5 \\ 0 & 1 & 0 \\ 0.25 & 0.5 & 0.25 \\ 0.1 & 0.4 & 0.5 \end{pmatrix}.$$

(ii) M-step means $\mu_k = \sum_i R_{ik}x_i / \sum_i R_{ik}$ with $X = [-3,-1,1,3]$:
- $N_1 = 0.3+0+0.25+0.1 = 0.65$; $\mu_1 = \frac{0.3(-3)+0(-1)+0.25(1)+0.1(3)}{0.65} = \frac{-0.9+0+0.25+0.3}{0.65} = \frac{-0.35}{0.65} = \boxed{-0.5385}$,
- $N_2 = 0.2+1+0.5+0.4 = 2.1$; $\mu_2 = \frac{0.2(-3)+1(-1)+0.5(1)+0.4(3)}{2.1} = \frac{-0.6-1+0.5+1.2}{2.1} = \frac{0.1}{2.1} = \boxed{0.0476}$,
- $N_3 = 0.5+0+0.25+0.5 = 1.25$; $\mu_3 = \frac{0.5(-3)+0(-1)+0.25(1)+0.5(3)}{1.25} = \frac{-1.5+0+0.25+1.5}{1.25} = \frac{0.25}{1.25} = \boxed{0.2}$.

$$\boxed{\mu_1+\mu_2+\mu_3 = -0.5385 + 0.0476 + 0.2 = -0.2909 \approx -0.29}.$$
(The MLT live-session slide reports $-0.53 + 0.04 + 0.2 = -0.29$ — same answer up to its one-decimal rounding.)

**6. M-step arithmetic.**

$\lambda_{3i} = [0.2, 0.3, 0.1, 0.6, 0.1, 0.5, 0.4, 0.7, 0.1, 0.2]$, $x = [2,1,3,2,1,2,0,-3,0,2]$, $n = 10$.

Effective count: $N_3 = \sum_i\lambda_{3i} = 0.2+0.3+0.1+0.6+0.1+0.5+0.4+0.7+0.1+0.2 = \boxed{3.2}$.

$$\boxed{\pi_3 = N_3/n = 3.2/10 = 0.32}.$$

Weighted sum: $\sum_i\lambda_{3i}x_i = 0.2(2)+0.3(1)+0.1(3)+0.6(2)+0.1(1)+0.5(2)+0.4(0)+0.7(-3)+0.1(0)+0.2(2)$
$= 0.4+0.3+0.3+1.2+0.1+1.0+0-2.1+0+0.4 = \boxed{1.6}$.

$$\boxed{\mu_3 = 1.6/3.2 = 0.5}.$$

**7. The $\pi_k$ update, derived.**

Maximize $J(\pi) = \sum_{i=1}^{n}\sum_{k=1}^{K}\gamma(z_{ik})\log\pi_k$ subject to $g(\pi) = \sum_{k=1}^{K}\pi_k - 1 = 0$. Lagrangian (§11):
$$\mathcal{L}(\pi,\lambda) = \sum_{i,k}\gamma(z_{ik})\log\pi_k - \lambda\!\left(\sum_{k=1}^{K}\pi_k - 1\right).$$
$$\frac{\partial\mathcal{L}}{\partial\pi_j} = \frac{\sum_{i=1}^{n}\gamma(z_{ij})}{\pi_j} - \lambda = \frac{N_j}{\pi_j} - \lambda = 0 \quad\Rightarrow\quad \pi_j = \frac{N_j}{\lambda}.$$
Summing over $j$ and using the constraint: $1 = \sum_j\pi_j = \frac{1}{\lambda}\sum_j N_j$. But $\sum_j N_j = \sum_j\sum_i\gamma(z_{ij}) = \sum_i\underbrace{\sum_j\gamma(z_{ij})}_{=1} = n$. Hence $\lambda = n$ and
$$\boxed{\pi_k = \frac{N_k}{n}}.$$

**8. True or false.**

(i) **True.** $\sum_k\gamma(z_{ik}) = \sum_k \frac{\pi_k N(x_i\mid\mu_k,\Sigma_k)}{\sum_j\pi_j N(x_i\mid\mu_j,\Sigma_j)} = 1$ — numerator sums to denominator (§25.8(i)).

(ii) **False.** EM converges to a *local* optimum / stationary point of the likelihood; which one depends on the initialization. No global guarantee (§25.10).

(iii) **True.** With $\gamma(z_{ik})\in\{0,1\}$, $N_k$ is the count of points in cluster $k$, and the M-step formulas become exactly the per-cluster sample mean (§20.6), sample covariance (§20.7), and fraction (§25.9(i)).

(iv) **False.** The mixture density is a sum — addition commutes — so permuted labels give the identical function (§25.4).

(v) **False.** $K = 1$ is a single Gaussian: one peak, always. Trimodal data needs (at least) three components (§25.2).
