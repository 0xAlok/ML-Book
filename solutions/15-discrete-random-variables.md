# Solutions — 15. Discrete random variables and key distributions

**1. (Random variables and ranges)** (a) The number of wickets and the number of dot balls are random variables: each is a function from the over's outcome (the ball-by-ball record) to a real number. "Whether the over was exciting" is not — it is not a numerical function of the outcome (no rule assigns it a real number).

(b) Number of wickets: $\{0, 1, \dots, 10\}$ (at most ten wickets can fall). Number of dot balls: $\{0, 1, \dots, 6\}$ (at most six legal deliveries in the over; wides/no-balls add deliveries but not dot balls).

**2. (PMF validity)** Sum to $1$:
$$1 = \sum_{k=1}^{\infty} \frac{c}{2^k} = c\cdot\frac{1/2}{1-1/2} = c,$$
so $\boxed{c = 1}$ — i.e. $X \sim \mathrm{Geometric}(1/2)$ in the trials-until-success convention (§15.13). Then $P(X \le 2) = f_X(1) + f_X(2) = \tfrac{1}{2} + \tfrac{1}{4} = \boxed{3/4}$.

**3. (Expectation and variance)**
$$E[X] = 0(0.3) + 1(0.5) + 2(0.2) = 0.9.$$
$$E[X^2] = 0^2(0.3) + 1^2(0.5) + 2^2(0.2) = 0.5 + 0.8 = 1.3.$$
$$\mathrm{Var}(X) = E[X^2] - (E[X])^2 = 1.3 - 0.81 = \boxed{0.49}.$$
So $\boxed{E[X] = 0.9}$, $\sigma = 0.7$.

**4. (CDF)** (a) The CDF jumps by $f_X(t)$ at each range value:
$$F_X(x) = \begin{cases}
0 & x < 0,\\
0.3 & 0 \le x < 1,\\
0.8 & 1 \le x < 2,\\
1 & x \ge 2.
\end{cases}$$
(b) $P(0 < X \le 2) = F_X(2) - F_X(0) = 1 - 0.3 = \boxed{0.7}$. Check via the PMF: $f_X(1) + f_X(2) = 0.5 + 0.2 = 0.7$. ✓

**5. (Linearity)** $E[X] = np = 100\cdot\tfrac{1}{4} = 25$ (§15.12). By linearity (§15.8): $E[2X + 3] = 2E[X] + 3 = 2(25) + 3 = \boxed{53}$.

**6. (Variance scaling)** $\mathrm{Var}(X) = (1-p)/p^2 = (1/2)/(1/4) = 2$ (§15.13). By the corollary in §15.9: $\mathrm{Var}(3X - 1) = 3^2\cdot\mathrm{Var}(X) = 9\cdot 2 = \boxed{18}$.

**7. (Binomial)** $X \sim \mathrm{Binomial}(5, 1/3)$:
$$P(X = 2) = \binom{5}{2}\left(\tfrac{1}{3}\right)^2\left(\tfrac{2}{3}\right)^3 = 10\cdot\tfrac{1}{9}\cdot\tfrac{8}{27} = \frac{80}{243} \approx \boxed{0.3292}.$$
(Ten sequences with exactly $2$ heads, each with probability $(1/3)^2(2/3)^3$ — the §15.12 counting argument.)

**8. (Memorylessness)** $P(X > k) = (1-p)^k = 0.7^k$. By memorylessness (§15.13):
$$P(X > 10 \mid X > 4) = \frac{0.7^{10}}{0.7^4} = 0.7^6 = \boxed{0.117649}.$$
The four survived years are forgotten — this equals $P(X > 6)$, the fresh $6$-year survival probability.

**9. (Negative binomial)** $X \sim \mathrm{NegativeBinomial}(3, 1/4)$:
$$P(X = 6) = \binom{5}{2}\left(\tfrac{3}{4}\right)^{3}\left(\tfrac{1}{4}\right)^{3} = 10\cdot\tfrac{27}{64}\cdot\tfrac{1}{64} = \frac{270}{4096} = \boxed{\frac{135}{2048} \approx 0.0659}.$$
(The $6$th trial is the $3$rd success; among the first $5$ trials exactly $2$ are successes: $\binom{5}{2}$ ways.)

**10. (Poisson fit)** (a) $P(X \le 1) = P(X = 0) + P(X = 1) = e^{-\lambda} + \lambda e^{-\lambda} = e^{-3.8673}(1 + 3.8673)$. With $e^{-3.8673} \approx 0.02094$: $P(X \le 1) \approx 0.02094 \times 4.8673 \approx \boxed{0.1018}$.

(b) The observed fraction is $0.022 + 0.078 = 0.100$ — the model ($0.1018$) matches the data to within $0.002$. The Poisson fit is good (see panel (c) of the figure).

**11. (Hypergeometric)** $E[X] = m\cdot r/N = 20\cdot\tfrac{50}{100} = \boxed{10}$.
$$\mathrm{Var}(X) = m\cdot\tfrac{r}{N}\cdot\tfrac{N-r}{N}\cdot\tfrac{N-m}{N-1} = 20\cdot\tfrac{1}{2}\cdot\tfrac{1}{2}\cdot\tfrac{80}{99} = \frac{400}{99} \approx \boxed{4.04}.$$
(Without the finite-population correction $\tfrac{80}{99}$, the variance would be $5$.)

**12. (Functions of a random variable)** $Y = \min(X, 3)$ caps $X$ at $3$:
- $P(Y = 1) = P(X = 1) = 1/2$,
- $P(Y = 2) = P(X = 2) = 1/4$,
- $P(Y = 3) = P(X \ge 3) = 1 - \tfrac{1}{2} - \tfrac{1}{4} = 1/4$.

Check: $\tfrac{1}{2}+\tfrac{1}{4}+\tfrac{1}{4} = 1$. ✓ Then
$$E[Y] = 1\cdot\tfrac{1}{2} + 2\cdot\tfrac{1}{4} + 3\cdot\tfrac{1}{4} = 0.5 + 0.5 + 0.75 = \boxed{1.75}.$$
(Compare $E[X] = 2$: capping the long tail lowers the mean, as it should.)
