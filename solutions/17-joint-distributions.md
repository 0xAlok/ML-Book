# Solutions — 17. Joint distributions: two random variables

**1. (Joint PMF validity and marginals)** The table:

| $X_1 \setminus X_2$ | $0$ | $1$ |
|---|---|---|
| $0$ | $0.05$ | $0.35$ |
| $1$ | $0.25$ | $0.35$ |

(a) Validity: every entry is in $[0, 1]$ ✓; the entries sum to $0.05 + 0.35 + 0.25 + 0.35 = 1.00$ ✓ (§17.2). Valid joint PMF.

(b) Summing out (§17.3):
$$f_{X_1}(0) = 0.05 + 0.25 = \boxed{0.30}, \qquad f_{X_1}(1) = 0.35 + 0.35 = \boxed{0.70},$$
$$f_{X_2}(0) = 0.05 + 0.35 = \boxed{0.40}, \qquad f_{X_2}(1) = 0.25 + 0.35 = \boxed{0.60}.$$

(c) Independence test (§17.5): $f_{X_1,X_2}(0, 0) = 0.05$ but $f_{X_1}(0)\,f_{X_2}(0) = 0.30 \times 0.40 = 0.12$. Since $0.05 \ne 0.12$, $\boxed{\text{not independent}}$ — one failing cell is enough.

**2. (The 2-digit number)** (a) $f_{X,Y}(2, 3)$: units digit $2$ and number $\equiv 3 \bmod 4$. A number ending in $2$ is even, hence $0$ or $2 \bmod 4$ — never $3$. The count is $0$: $\boxed{f_{X,Y}(2,3) = 0}$.

(b) One-cell justification (§17.5, zero-cell shortcut): $f_{X,Y}(2, 3) = 0$ while $f_X(2) = 1/10 > 0$ and $f_Y(3) = 1/4 > 0$, so $f_{X,Y}(2,3) \ne f_X(2)\,f_Y(3) = 1/40$. $\boxed{\text{Not independent}.}$

**3. (Conditional PMFs)** On eg 5's table, $f_X(1) = 1/4$, $f_Y(0) = 1/2$.

(a) $f_{Y \mid X = 1}(0) = \dfrac{f_{X,Y}(1,0)}{f_X(1)} = \dfrac{1/8}{1/4} = \boxed{\tfrac{1}{2}}$, $f_{Y \mid X = 1}(1) = \dfrac{1/8}{1/4} = \boxed{\tfrac{1}{2}}$. Check: $1/2 + 1/2 = 1$ ✓ — a genuine PMF (§17.4).

(b) $f_{X \mid Y = 0}(2) = \dfrac{f_{X,Y}(2,0)}{f_Y(0)} = \dfrac{1/8}{1/2} = \boxed{\tfrac{1}{4}}$.

**4. (Joint CDF)** (a) With $1/4$ in every cell of eg 1's table:
$$\boxed{F_{X_1,X_2}(0,0) = 1/4}, \quad \boxed{F_{X_1,X_2}(0,1) = 1/2}, \quad \boxed{F_{X_1,X_2}(1,0) = 1/2}, \quad \boxed{F_{X_1,X_2}(1,1) = 1}.$$

(b) Rectangle formula (§17.6) on $(0,1] \times (0,1]$:
$$P(X_1 = 1, X_2 = 1) = F(1,1) - F(0,1) - F(1,0) + F(0,0) = 1 - \tfrac{1}{2} - \tfrac{1}{2} + \tfrac{1}{4} = \boxed{\tfrac{1}{4}},$$
matching the table ✓.

**5. ($E[g(X,Y)]$ and covariance)** (a) By total expectation (§17.7): $E[Y \mid X = t] = t/2$ for $(Y \mid X = t) \sim \mathrm{Binomial}(t, 1/2)$, and $E[X] = 7/2$, so
$$E[Y] = \sum_{t=1}^{6} \frac{t}{2}\cdot\frac{1}{6} = \frac{1}{2}E[X] = \boxed{\frac{7}{4}}.$$

(b) $E[XY] = E_X\big[X\cdot E[Y \mid X]\big] = E[X^2]/2$. For a fair die, $E[X^2] = (1 + 4 + 9 + 16 + 25 + 36)/6 = 91/6$, so $E[XY] = 91/12$. Then
$$\mathrm{Cov}(X, Y) = E[XY] - E[X]E[Y] = \frac{91}{12} - \frac{7}{2}\cdot\frac{7}{4} = \frac{91}{12} - \frac{49}{8} = \frac{182 - 147}{24} = \boxed{\frac{35}{24} \approx 1.46},$$
positive — more die pips mean more tosses, hence more heads, as expected.

**6. (Covariance and correlation)** On eg 5's table (§17.9–17.10):
$$E[X] = 0\cdot\tfrac{3}{8} + 1\cdot\tfrac{1}{4} + 2\cdot\tfrac{3}{8} = 1, \qquad E[Y] = \tfrac{1}{2},$$
$$E[XY] = 1\cdot 1\cdot\tfrac{1}{8} + 2\cdot 1\cdot\tfrac{1}{4} = \tfrac{5}{8},$$
$$\boxed{\mathrm{Cov}(X, Y) = \tfrac{5}{8} - 1\cdot\tfrac{1}{2} = \tfrac{1}{8}}.$$
For $\rho$: $E[X^2] = 0 + 1\cdot\tfrac{1}{4} + 4\cdot\tfrac{3}{8} = 7/4$, so $\mathrm{Var}(X) = 7/4 - 1 = 3/4$; $E[Y^2] = 1/2$, so $\mathrm{Var}(Y) = 1/2 - 1/4 = 1/4$. Hence
$$\boxed{\rho(X, Y) = \frac{1/8}{\sqrt{3/4}\,\sqrt{1/4}} = \frac{1/8}{\sqrt{3}/4} = \frac{1}{2\sqrt{3}} = \frac{\sqrt{3}}{6} \approx 0.29}.$$

**7. (Variance of sums)** $X, Y \stackrel{\text{i.i.d.}}{\sim} \mathrm{Bernoulli}(p)$ are independent, so $\mathrm{Cov}(X, Y) = 0$ (§17.9) and $\mathrm{Var}(X) = \mathrm{Var}(Y) = p(1-p)$ (§15.11). By §17.11:
$$\boxed{\mathrm{Var}(X + Y) = p(1-p) + p(1-p) + 0 = 2p(1-p)},$$
$$\boxed{\mathrm{Var}(X - Y) = p(1-p) + p(1-p) - 0 = 2p(1-p)}.$$
Note the sign of the sum doesn't matter for independent variables — both give $2p(1-p)$.

**8. (Joint PDF)** (a) $1 = \int_0^2\!\!\int_1^3 cxy\,dy\,dx = c\cdot 2\cdot\frac{9-1}{2} = 8c$, so $\boxed{c = 1/8}$.

(b) $P(X < 1,\ Y > 2) = \int_0^1\!\!\int_2^3 \dfrac{xy}{8}\,dy\,dx = \dfrac{1}{8}\cdot\left[\dfrac{x^2}{2}\right]_0^1\!\cdot\left[\dfrac{y^2}{2}\right]_2^3 = \dfrac{1}{8}\cdot\dfrac{1}{2}\cdot\dfrac{9-4}{2} = \boxed{\dfrac{5}{32}}.$

**9. (Marginal and conditional PDFs)** (a) Integrating out (§17.13):
$$f_X(x) = \int_1^3 \frac{xy}{8}\,dy = \frac{x}{8}\cdot\frac{9-1}{2} = \boxed{\frac{x}{2}}, \quad 0 < x < 2,$$
$$f_Y(y) = \int_0^2 \frac{xy}{8}\,dx = \frac{y}{8}\cdot\frac{4}{2} = \boxed{\frac{y}{4}}, \quad 1 < y < 3.$$

(b) $f_{Y \mid X = x}(y) = \dfrac{f_{X,Y}(x,y)}{f_X(x)} = \dfrac{xy/8}{x/2} = \boxed{\dfrac{y}{4}}, \quad 1 < y < 3$ (§17.14).

(c) $\boxed{\text{Yes, independent}}$: $f_{Y \mid X = x}(y) = y/4 = f_Y(y)$ for every $x$, equivalently $f_{X,Y}(x,y) = (x/2)(y/4) = f_X(x)f_Y(y)$, on the rectangular support $(0,2)\times(1,3)$ (§17.13's Note).

**10. (IPL powerplay)** By total expectation (§17.7's Note): $E[X \mid Y = 0] = (6 + 12)/2 = 9$, $E[X \mid Y = 1] = (2 + 8)/2 = 5$, $E[X \mid Y = 2] = (0 + 6)/2 = 3$ (means of discrete uniforms, §15.10). So
$$E[X] = \tfrac{13}{16}\cdot 9 + \tfrac{1}{8}\cdot 5 + \tfrac{1}{16}\cdot 3 = \frac{117 + 10 + 3}{16} = \boxed{\frac{65}{8} = 8.125}.$$
Sanity check: most overs have $0$ wickets ($13/16$) with mean $9$ runs, so $E[X]$ should sit just below $9$ — $8.125$ ✓.
