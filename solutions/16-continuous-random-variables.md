# Solutions — 16. Continuous random variables

**1. (PDF validity)** (a) $f(x) = 3x^2 \ge 0$ on $(0,1)$; $\int_{-\infty}^{\infty} f(x)\,dx = \int_0^1 3x^2\,dx = [x^3]_0^1 = 1$; piecewise continuous. All three density properties (§16.3) hold, so $f$ is valid.

(b) $P(X = 1/5) = \boxed{0}$ — every single point has probability zero for a continuous random variable (§16.2). For the interval,
$$P(X \in [1/5 - \epsilon, 1/5 + \epsilon]) = \int_{1/5-\epsilon}^{1/5+\epsilon} 3x^2\,dx = \left[x^3\right]_{1/5-\epsilon}^{1/5+\epsilon} = \boxed{(1/5+\epsilon)^3 - (1/5-\epsilon)^3} \approx \frac{6\epsilon}{25}$$
for small $\epsilon$ (using $P \approx f(1/5)\cdot 2\epsilon$). Positive, but shrinking to $0$ with $\epsilon$ — precision costs probability.

**2. (Finding the constant)** Integrate to $1$:
$$1 = k\cdot\frac{1}{4} + 2k\cdot\frac{1}{2} + 3k\cdot\frac{1}{4} = \frac{k}{4} + k + \frac{3k}{4} = 2k,$$
so $\boxed{k = 1/2}$. Then
$$P(X \le 1/2) = \int_0^{1/4}\tfrac{1}{2}\,dx + \int_{1/4}^{1/2} 1\,dx = \tfrac{1}{2}\cdot\tfrac{1}{4} + 1\cdot\tfrac{1}{4} = \frac{1}{8} + \frac{1}{4} = \boxed{\frac{3}{8}}.$$

**3. (CDF from PDF)** (a) For $0 \le x \le 2$: $F_X(x) = \int_0^x (u/2)\,du = x^2/4$. So
$$F_X(x) = \begin{cases}
0 & x < 0,\\
x^2/4 & 0 \le x \le 2,\\
1 & x > 2.
\end{cases}$$
Check: $F_X(2) = 1$ ✓, and $F_X' = x/2 = f_X$ on $(0,2)$ ✓.

(b) $P(0.5 < X \le 1.5) = F_X(1.5) - F_X(0.5) = \dfrac{2.25}{4} - \dfrac{0.25}{4} = \boxed{0.5}$.

**4. (Uniform)** $f_X(x) = 1/20$ on $[-10, 10]$ (§16.7).

$P(-3 \le X \le 2) = \dfrac{2-(-3)}{20} = \boxed{1/4}$ (length ratio).

$P(5 < |X| < 7)$: $|X| \in (5,7)$ iff $X \in (-7,-5)\cup(5,7)$, total length $2+2=4$: $4/20 = \boxed{1/5}$.

$P(X > 7 \mid X > 3) = \dfrac{P(X > 7)}{P(X > 3)} = \dfrac{(10-7)/20}{(10-3)/20} = \boxed{3/7}$.

**5. (Expectation and variance)** $E[X] = \int_0^1 x\cdot 2x\,dx = \int_0^1 2x^2\,dx = \left[\tfrac{2x^3}{3}\right]_0^1 = \boxed{2/3}$.

$E[X^2] = \int_0^1 x^2\cdot 2x\,dx = \int_0^1 2x^3\,dx = \left[\tfrac{x^4}{2}\right]_0^1 = 1/2$.

$\mathrm{Var}(X) = E[X^2] - (E[X])^2 = \tfrac{1}{2} - \tfrac{4}{9} = \tfrac{9-8}{18} = \boxed{1/18}$, $\sigma = 1/\sqrt{18} \approx 0.236$.

**6. (Exponential probabilities)** $F_X(x) = 1 - e^{-2x}$ for $x > 0$ (§16.8).

$P(5 < X < 7) = F_X(7) - F_X(5) = (1 - e^{-14}) - (1 - e^{-10}) = e^{-10} - e^{-14} \approx \boxed{4.5\times 10^{-5}}$.

$P(X > 7 \mid X > 3) = P(X > 7-3) = P(X > 4)$ by memorylessness $= e^{-2\cdot 4} = e^{-8} \approx \boxed{3.35\times 10^{-4}}$.

**7. (Exponential mean)** $E[X] = 1/\lambda = 1/0.5 = \boxed{2\text{ years}}$ (§16.8). $P(X > 2) = e^{-\lambda\cdot 2} = e^{-1} \approx \boxed{0.3679}$ — about a $37\%$ chance the bulb outlives its own mean lifetime (the exponential's right skew at work).

**8. (Memorylessness)** $P(X > 10 \mid X > 4) = P(X > 10-4) = P(X > 6) = e^{-0.3\cdot 6} = e^{-1.8} \approx \boxed{0.1653}$ (§16.8). The $4$ survived years are forgotten — same answer as for a brand-new component asked to survive $6$ years. (Compare §15's Problem 8: the geometric version.)

**9. (Normal)** Standardize with $Z = (X-3)/1 = X-3$.

$P(5 < X < 7) = P(2 < Z < 4) = \Phi(4) - \Phi(2) \approx 0.99997 - 0.97725 = \boxed{0.0227}$.

$P(X > 7 \mid X > 3) = \dfrac{P(X > 7)}{P(X > 3)} = \dfrac{1-\Phi(4)}{1-\Phi(0)} \approx \dfrac{0.0000317}{0.5} = \boxed{6.3\times 10^{-5}}$.
(The normal has no memorylessness — the ratio must be computed directly; contrast Problem 8.)

**10. (Minimum of exponentials)** $P(Z > z) = P(X > z)P(Y > z) = e^{-0.25z}e^{-0.25z} = e^{-0.5z}$ for $z \ge 0$ (§16.8, Note). So $\boxed{Z \sim \mathrm{Exponential}(0.5)}$ per year — the laptop fails twice as fast as either part alone, with $E[Z] = 2$ years.

$P(Z \le 1) = 1 - e^{-0.5} \approx \boxed{0.3935}$ — about a $39\%$ chance the laptop dies within its first year.
