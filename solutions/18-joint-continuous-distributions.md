# Solutions — 18. Joint continuous distributions

**1. (2-D uniform triangle — the deck's Q1)** $D$ is the right triangle with vertices $(0,0), (2,0), (0,2)$; $|D| = \frac{1}{2}\cdot 2\cdot 2 = 2$, so $f_{X,Y}(x,y) = 1/2$ on $D$ (§18.2).

(a) $A = \{x + y < 1,\ x > 0,\ y > 0\}$ is a triangle with legs $1$: $|A| = 1/2$. By the area ratio,
$$\boxed{P(X + Y < 1) = \frac{1/2}{2} = \frac{1}{4}}.$$

(b) The complement inside $D$: $\{x + 2y < 1,\ x > 0,\ y > 0\}$ is a triangle with $x$-intercept $1$ and $y$-intercept $1/2$, area $\frac{1}{2}\cdot 1\cdot\frac{1}{2} = 1/4$. Hence
$$\boxed{P(X + 2Y > 1) = \frac{2 - 1/4}{2} = \frac{7/4}{2} = \frac{7}{8}}.$$

**2. (Region integration — the deck's Q2)** (a) Non-negative on the unit square ✓, and
$$\int_0^1\!\!\int_0^1 (x + y)\,dy\,dx = \int_0^1\!\left(x + \tfrac{1}{2}\right)dx = \tfrac{1}{2} + \tfrac{1}{2} = \boxed{1} \text{ ✓},$$
so it is a valid joint density (§18.2, eg 2).

(b)
$$P(X < \tfrac{1}{2},\, Y < \tfrac{1}{2}) = \int_0^{1/2}\!\!\int_0^{1/2}(x+y)\,dy\,dx = \int_0^{1/2}\!\left(\tfrac{x}{2}+\tfrac{1}{8}\right)dx = \left[\tfrac{x^2}{4}+\tfrac{x}{8}\right]_0^{1/2} = \tfrac{1}{16}+\tfrac{1}{16} = \boxed{\tfrac{1}{8}}.$$
(The deck prints $3/16$, integrating $x/2$ to $x^2/2$ instead of $x^2/4$ — an arithmetic slip; $1/8$ is correct.)

(c) The region $\{x + y < 1\}\cap[0,1]^2$: $0 < x < 1$, $0 < y < 1 - x$:
$$\begin{aligned}
P(X + Y < 1) &= \int_0^1\!\!\int_0^{\,1-x}(x+y)\,dy\,dx = \int_0^1\!\left(x(1-x)+\tfrac{(1-x)^2}{2}\right)dx \\
&= \int_0^1\!\left(\tfrac{1}{2}-\tfrac{x^2}{2}\right)dx = \tfrac{1}{2}-\tfrac{1}{6} = \boxed{\tfrac{1}{3}}.
\end{aligned}$$

**3. (Marginals — the deck's 3.5 Q3)** (a) Integrating out (§17.13):
$$f_X(x) = \int_0^1 \tfrac{6}{5}(x+y^2)\,dy = \tfrac{6}{5}\left[x + \tfrac{1}{3}\right], \qquad \boxed{f_X(x) = \tfrac{6}{5}\!\left(x + \tfrac{1}{3}\right)},\ 0 < x < 1,$$
$$f_Y(y) = \int_0^1 \tfrac{6}{5}(x+y^2)\,dx = \tfrac{6}{5}\left[\tfrac{1}{2} + y^2\right], \qquad \boxed{f_Y(y) = \tfrac{6}{5}\!\left(\tfrac{1}{2} + y^2\right)},\ 0 < y < 1.$$
Check: $\int_0^1 \frac{6}{5}(x + 1/3)\,dx = \frac{6}{5}(1/2 + 1/3) = 1$ ✓; $\int_0^1 \frac{6}{5}(1/2 + y^2)\,dy = \frac{6}{5}(1/2 + 1/3) = 1$ ✓.

(b) Independence test (§17.5): at $(0, 0)$, $f_{X,Y}(0,0) = 0$ but $f_X(0)\,f_Y(0) = \frac{6}{5}\cdot\frac{1}{3}\cdot\frac{6}{5}\cdot\frac{1}{2} = \frac{6}{25} \ne 0$. $\boxed{\text{Not independent}.}$

**4. (Uniform marginals, dependent joint — the deck's §3.2 Q2)** $|D| = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}$, so $f_{X,Y} = 2$ on $D$ (the deck's §3.1).

(a) For $0 < x < \frac{1}{2}$: $f_X(x) = \int_0^{1/2} 2\,dy = 1$. For $\frac{1}{2} < x < 1$: $f_X(x) = \int_{1/2}^1 2\,dy = 1$. So $\boxed{f_X(x) = 1,\ 0 < x < 1}$: $X \sim \mathrm{Uniform}[0,1]$. By symmetry, $\boxed{f_Y(y) = 1,\ 0 < y < 1}$: $Y \sim \mathrm{Uniform}[0,1]$.

(b) Zero-cell test (§17.5): $f_{X,Y}(1/4,\,3/4) = 0$ (off the diagonal squares), while $f_X(1/4)\,f_Y(3/4) = 1\cdot 1 = 1 \ne 0$. $\boxed{\text{Not independent}}$ — uniform marginals do not imply a uniform (independent) joint.

**5. (Convolution: uniforms)** §18.3, eg 3. With $f_X = f_Y = \mathbf{1}_{(0,1)}$,
$$f_Z(z) = \int_{-\infty}^{\infty} f_X(x)\,f_Y(z-x)\,dx,$$
the integrand is $1$ on $x \in (0,1)\cap(z-1,z)$. For $0 < z < 1$ the overlap is $(0,z)$: $f_Z(z) = z$. For $1 \le z < 2$ the overlap is $(z-1,1)$: $f_Z(z) = 2 - z$. Hence
$$\boxed{f_Z(z) = \begin{cases} z & 0 < z < 1, \\ 2 - z & 1 \le z < 2, \\ 0 & \text{otherwise}, \end{cases}}$$
the triangular distribution — panel (a) of the figure.

**6. (Convolution: exponentials)** §18.3, eg 4. For $z > 0$:
$$f_Z(z) = \int_0^z \lambda e^{-\lambda x}\,\lambda e^{-\lambda(z-x)}\,dx = \lambda^2 e^{-\lambda z}\int_0^z dx = \boxed{\lambda^2 z\,e^{-\lambda z}},\qquad z > 0$$
($0$ for $z \le 0$): the Gamma$(2,\lambda)$ density. Check the mean: $E[Z] = \int_0^\infty \lambda^2 z^2 e^{-\lambda z}\,dz = \frac{1}{\lambda}\cdot 2! = 2/\lambda = E[X]+E[Y]$ ✓ (§17.7).

**7. (Max and min)** §18.4, eg 6. $F_X(z) = z$ on $(0,1)$. $\max$: $F_M(z) = z^2$, so $\boxed{f_{\max}(z) = 2z,\ 0 < z < 1}$. $\min$: $F_m(z) = 1-(1-z)^2 = 2z - z^2$, so $\boxed{f_{\min}(z) = 2(1-z),\ 0 < z < 1}$. (Panel (b) of the figure.)

**8. (Min of exponentials)** (a) §18.4, eg 7:
$$P\big(\min(X,Y) > z\big) = P(X > z)\,P(Y > z) = e^{-\lambda_1 z}\,e^{-\lambda_2 z} = e^{-(\lambda_1+\lambda_2)z},\quad z > 0,$$
so $\boxed{\min(X,Y) \sim \mathrm{Exp}(\lambda_1+\lambda_2)}$.

(b) §18.4, eg 8:
$$\begin{aligned}
P(X > Y) &= \int_0^\infty\!\!\int_0^x \lambda_1 e^{-\lambda_1 x}\,\lambda_2 e^{-\lambda_2 y}\,dy\,dx = \int_0^\infty \lambda_1 e^{-\lambda_1 x}\big(1-e^{-\lambda_2 x}\big)\,dx \\
&= 1 - \frac{\lambda_1}{\lambda_1+\lambda_2} = \boxed{\frac{\lambda_2}{\lambda_1+\lambda_2}}.
\end{aligned}$$

**9. (Jacobian)** §18.5, eg 9. (a) $U = X+Y$, $V = X-Y$ inverts to $x = (u+v)/2$, $y = (u-v)/2$;
$$\left|\frac{\partial(x,y)}{\partial(u,v)}\right| = \left|\det\begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix}\right| = \tfrac{1}{2}.$$
The unit square maps to the diamond $0 < u < 2$, $|v| < \min(u, 2-u)$ (panel (c) of the figure). Hence
$$\boxed{f_{U,V}(u,v) = \tfrac{1}{2}\ \text{on the diamond},\ 0\ \text{elsewhere}.}$$

(b) $f_U(u) = \int_{-\min(u,2-u)}^{\min(u,2-u)} \frac{1}{2}\,dv = \min(u,2-u)$, i.e.
$$\boxed{f_U(u) = \begin{cases} u & 0 < u < 1, \\ 2-u & 1 \le u < 2, \end{cases}}$$
identical to Problem 5's convolution result ✓ — the Jacobian and convolution routes agree.

**10. (MGF of a sum)** §18.6, eg 11. $M_X(s) = e^{\mu_1 s + \sigma_1^2 s^2/2}$, $M_Y(s) = e^{\mu_2 s + \sigma_2^2 s^2/2}$. By independence,
$$M_{X+Y}(s) = M_X(s)\,M_Y(s) = e^{(\mu_1+\mu_2)s + (\sigma_1^2+\sigma_2^2)s^2/2},$$
the MGF of $\mathrm{Normal}(\mu_1+\mu_2,\ \sigma_1^2+\sigma_2^2)$. By MGF uniqueness,
$$\boxed{X + Y \sim \mathrm{Normal}\big(\mu_1+\mu_2,\ \sigma_1^2+\sigma_2^2\big)}.$$
