# Solutions — Chapter 21: Inequalities and the central limit theorem

**1.** $X$ = heads in $200$ flips, $X \sim \mathrm{Binomial}(200, 1/10)$ (§15.12). $E[X] = np = 200\cdot\frac{1}{10} = 20$. $X \ge 0$, so Markov (§21.2):
$$P(X \ge 120) \le \frac{E[X]}{120} = \frac{20}{120} = \boxed{\frac{1}{6}}.$$
(The practice assignment's answer: A = $1/6$.)

**2.** $X \sim \mathrm{Binomial}(n,p)$ has $E[X] = np$ (§15.12). Markov (§21.2), with $t = \alpha n > 0$:
$$P(X \ge \alpha n) \le \frac{E[X]}{\alpha n} = \frac{np}{\alpha n} = \boxed{\frac{p}{\alpha}}.$$
The condition $p < \alpha < 1$ makes the bound $< 1$ (non-vacuous) and $\alpha n > E[X]$ (the §21.2 note). For $p = 1/2$, $\alpha = 3/4$:
$$P(X \ge 3n/4) \le \frac{1/2}{3/4} = \frac{2}{3} \approx \boxed{0.667}.$$
(The tutorial's answer: A = $2/3$.)

**3.** First shift to a deviation from the mean, then apply Chebyshev (§21.3):
$$P(X \ge \alpha n) = P(X - np \ge n\alpha - np) \le P(|X - np| \ge n(\alpha - p)) \le \frac{\mathrm{Var}(X)}{[n(\alpha-p)]^2}.$$
The first inequality holds because $\{X - np \ge t\} \subseteq \{|X-np| \ge t\}$ for $t = n(\alpha-p) > 0$. With $\mathrm{Var}(X) = np(1-p)$ (§15.12):
$$P(X \ge \alpha n) \le \frac{np(1-p)}{n^2(\alpha-p)^2} = \boxed{\frac{p(1-p)}{n(\alpha-p)^2}}.$$
For $p = 1/2$, $\alpha = 3/4$, $n = 8$: $\alpha - p = 1/4$,
$$\frac{(1/2)(1/2)}{8\cdot(1/4)^2} = \frac{1/4}{8/16} = \frac{1/4}{1/2} = \boxed{0.5}.$$
(The practice assignment's answer: $0.5$, equivalently $4/n$.) Comparison: Markov (Problem 2) gave $2/3 \approx 0.667$; Chebyshev gives $0.5$ — **Chebyshev is tighter here**, because it exploits the variance, not just the mean.

**4.** (i) $p(x) = 2^{-x}$, $x = 1, 2, \ldots$ is $\mathrm{Geometric}(1/2)$ (first success on trial $x$). From §15.13 (eg 12): $E[X] = 1/p = \boxed{2}$, $\mathrm{Var}(X) = (1-p)/p^2 = (1/2)/(1/4) = \boxed{2}$. Chebyshev (§21.3) with $t = 2$:
$$P(|X - 2| \ge 2) \le \frac{2}{2^2} = \frac{1}{2} \quad\Rightarrow\quad \boxed{P(|X - 2| \le 2) \ge 1 - \frac12 = \frac12}.$$
(ii) Exact: $|X-2| \le 2 \iff 0 \le X \le 4$, and $X \ge 1$ always, so
$$P(|X-2| \le 2) = P(X \le 4) = \sum_{k=1}^{4} 2^{-k} = \frac12 + \frac14 + \frac18 + \frac{1}{16} = \boxed{\frac{15}{16} = 0.9375}.$$
Comparison: Chebyshev guarantees $\ge 0.5$; the truth is $0.9375$. The bound is valid but loose — Chebyshev knows only the mean and variance, and the geometric's long right tail forces it to be pessimistic.

**5.** Chebyshev (§21.3): $P(|X-\mu| \ge t) \le \sigma^2/t^2$. Set $t = k\sigma$ ($k > 0$):
$$P(|X - \mu| \ge k\sigma) \le \frac{\sigma^2}{k^2\sigma^2} = \boxed{\frac{1}{k^2}}.$$
For $k = 3$: $P(|X-\mu| \ge 3\sigma) \le 1/9$, so $\boxed{P(|X-\mu| < 3\sigma) \ge 1 - 1/9 = 8/9 \approx 0.889}$ — for every distribution with finite variance. (For $k = 2$ this recovers eg 3's $1/4$ / $3/4$, the practice assignment's Q8.)

**6.** Chebyshev on $\bar{X}_n$ (the §21.5 calculation): $P(|\bar{X}_n - \mu| \ge 0.1\sigma) \le \dfrac{\sigma^2}{n(0.1\sigma)^2} = \dfrac{100}{n}$. Demand $\le 0.01$:
$$\frac{100}{n} \le 0.01 \;\Rightarrow\; n \ge \frac{100}{0.01} = \boxed{10{,}000}.$$
The answer says Chebyshev's $1/n$ rate is *slow*: cutting the tolerance by $10$ costs $100\times$ the samples, and cutting the failure probability by $10$ costs $10\times$ the samples. The lecture's point (§21.5 note): the bound is honest, but the true concentration (Hoeffding's exponential rate, for bounded variables) needs far fewer samples.

**7.** (i) $Y_{36} = \frac{1}{\sqrt{36}}\sum_{i=1}^{36}(X_i - 10) \xrightarrow{d} N(0, \sigma^2) = \boxed{N(0, 4)}$ (§21.6). Equivalently (§21.6 corollary),
$$\bar{X}_{36} \;\dot\sim\; \boxed{N\!\left(10,\ \frac{4}{36}\right) = N(10,\ 1/9)}, \qquad \text{sd} = \frac{2}{6} = \frac13.$$
(ii) Standardize:
$$P(\bar{X}_{36} \ge 10.5) \approx P\!\left(Z \ge \frac{10.5 - 10}{1/3}\right) = P(Z \ge 1.5) = 1 - \Phi(1.5) \approx 1 - 0.93319 = \boxed{0.0668}.$$

**8.** $X = \sum_{i=1}^{100} X_i$, $X_i \sim \mathrm{Bernoulli}(1/2)$: $E[X] = 50$, $\mathrm{Var}(X) = 25$, sd $= 5$ (§21.8 setup). CLT: $X \;\dot\sim\; N(50, 25)$:
$$P(X \le 42) \approx P\!\left(Z \le \frac{42-50}{5}\right) = \Phi(-1.6) = 1 - \Phi(1.6) \approx 1 - 0.94520 = \boxed{0.0548}.$$
Exact binomial value: $0.0666$. Error: $|0.0666 - 0.0548| = 0.0118$ — about $1.2$ percentage points, i.e. the CLT is off by $\approx 18\%$ relative. Decent for $n = 100$ with no continuity correction; it improves as $n$ grows. (The CLT is an approximation, not a bound — §21.8.)

**9.** $E[X] = 0\cdot P(X=0) + t\cdot P(X=t) = 0\cdot(1-\mu/t) + t\cdot(\mu/t) = \boxed{\mu}$. And $P(X \ge t) = P(X = t) = \boxed{\mu/t} = E[X]/t$ — equality in Markov (§21.2). Conditions: $\mu > 0$, $t > \mu$ ensure $0 < \mu/t < 1$, so the probabilities are valid ($1 - \mu/t > 0$). This is the general form of eg 2's $\{0, 4/5;\ 50, 1/5\}$ example ($\mu = 10$, $t = 50$): no bound using only the mean can beat Markov, because this distribution attains it.

**10.** The $1/n$ scaling in $\bar{X}_n$ *averages*: $\mathrm{Var}(\bar{X}_n) = \sigma^2/n \to 0$, so the distribution collapses onto the single point $\mu$ — the only possible limit of a "distribution" with vanishing variance is a constant, and "convergence to a constant" is convergence in probability (the WLLN). If you fed $\bar{X}_n$ to the CLT's machinery instead, you'd get the degenerate claim "$\bar{X}_n \xrightarrow{d} N(\mu, 0)$" — a normal with zero variance is just the point $\mu$ again; all shape information is lost. The CLT's $1/\sqrt{n}$ scaling is the unique zoom where $\mathrm{Var}(Y_n) = \sigma^2$ stays fixed (§21.6 note): the center still converges (that part is the WLLN), but the *fluctuations* stay visible and converge in distribution to $N(0,\sigma^2)$. In short: $1/n$ asks "where does it settle?" (answer: $\mu$, in probability); $1/\sqrt{n}$ asks "what do the wiggles look like on the way there?" (answer: normal, in distribution).
