# Solutions — 20. Estimation: MLE and Bayesian/MAP

**1.** $L(p) = P_\text{Bern}(1)\,P_\text{Bern}(0)\,P_\text{Bern}(1)\,P_\text{Bern}(1) = p\cdot(1-p)\cdot p\cdot p = \boxed{p^3(1-p)}$.
$\log L(p) = 3\log p + \log(1-p)$. Differentiate:
$$\frac{d}{dp}\log L = \frac{3}{p} - \frac{1}{1-p} \stackrel{!}{=} 0 \;\Rightarrow\; 3(1-p) = p \;\Rightarrow\; 3 = 4p \;\Rightarrow\; \boxed{\hat p_{ML} = 3/4}.$$
Second derivative $-\frac{3}{p^2} - \frac{1}{(1-p)^2} < 0$ everywhere, so this is the (unique) maximizer.
Numeric check: $L(0.75) = 0.75^3\cdot0.25 = 0.421875\cdot0.25 = 0.10546875$; $L(0.5) = 0.5^4 = 0.0625 < 0.10546875$ ✓ — the peak is at $3/4$.

**2.** $\hat p_{ML} = \frac{\text{\#ones}}{n} = \frac{10}{10 + n_0} = 0.25$. Solve: $10 = 0.25(10+n_0)$ ⇒ $10 = 2.5 + 0.25n_0$ ⇒ $7.5 = 0.25n_0$ ⇒ $\boxed{n_0 = 30}$ zeros (so $n = 40$ total; check: $10/40 = 0.25$ ✓).

**3.** (i) $L(\theta) = \prod_{i=1}^{4}\theta x_i^{\theta-1} = \theta^4\prod_{i=1}^{4}x_i^{\theta-1}$, so
$$\boxed{\ell(\theta) = 4\log\theta + (\theta-1)\sum_{i=1}^{4}\log x_i}.$$
(ii) $\log x_i = \log e^{-i} = -i$, so $\sum\log x_i = -(1+2+3+4) = -10$:
$$\ell(\theta) = 4\log\theta - 10(\theta-1), \qquad \frac{d\ell}{d\theta} = \frac{4}{\theta} - 10 \stackrel{!}{=} 0 \;\Rightarrow\; \boxed{\hat\theta_{ML} = \frac{4}{10} = 0.4}.$$
Second derivative $-4/\theta^2 < 0$ ⇒ maximum ✓. (Sanity: $\theta = 0.4 < 1$ makes $x^{\theta-1} = x^{-0.6}$ decreasing in $x$... the density piles toward $0$, matching data clustered near $0$: $e^{-4} \approx 0.018$.)

**4.** With $\mu$ known, only $\sigma^2$ is free:
$$R(\sigma^2) = \frac{n}{2}\log\sigma^2 + \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2 + \text{const}.$$
Let $v = \sigma^2$. $\frac{dR}{dv} = \frac{n}{2v} - \frac{1}{2v^2}\sum_{i=1}^n(x_i-\mu)^2$ (using $\frac{d}{dv}\frac{1}{v} = -\frac{1}{v^2}$). Set to $0$:
$$\frac{n}{2v} = \frac{\sum_{i=1}^n(x_i-\mu)^2}{2v^2} \;\Rightarrow\; nv = \sum_{i=1}^n(x_i-\mu)^2 \;\Rightarrow\; \boxed{\hat\sigma^2_{ML} = \frac{1}{n}\sum_{i=1}^n (x_i-\mu)^2}.$$
(Only the known $\mu$, not the estimated $\bar x$, appears inside — this is the tutorial's answer B.)

**5.** $\boxed{\hat a_{ML} = 1.8}$, $\boxed{\hat b_{ML} = 4.0}$ (min and max of the data). For $[1.0, 5.0]$: $L = \left(\frac{1}{5-1}\right)^4 = \frac{1}{256} \approx 0.0039$, versus $L(\hat a,\hat b) = \left(\frac{1}{2.2}\right)^4 \approx 0.0427$ — ten times larger. One line: the likelihood is $(b-a)^{-n}$, strictly decreasing in the width $b-a$, so any strictly wider interval containing the data has strictly smaller likelihood.

**6.** $\hat\mu_{ML} = (4+5+7+8)/4 = \boxed{6}$; squared deviations $4, 1, 1, 4$ sum to $10$, so $\hat\sigma^2_{ML} = 10/4 = \boxed{2.5}$.
$R(\mu) = \frac{1}{2\hat\sigma^2_{ML}}\sum_{i=1}^4(x_i-\mu)^2 + \frac{4}{2}\log\hat\sigma^2_{ML}$ with $\frac{4}{2}\log 2.5 = 2(0.9162907) = 1.8326$:
- $R(6) = \frac{10}{5} + 1.8326 = 2 + 1.8326 = \boxed{3.8326}$,
- $R(5) = \frac{1+0+4+9}{5} + 1.8326 = \frac{14}{5} + 1.8326 = 2.8 + 1.8326 = \boxed{4.6326} > 3.8326$ ✓.
So $\mu = 6$ really gives the smaller negative log-likelihood.

**7.** $\hat{\boldsymbol{\mu}}_{ML} = \left(\frac{0+1+2}{3}, \frac{0+2+1}{3}\right)^T = \boxed{(1,1)^T}$.
Centered observations: $(-1,-1)^T$, $(0,1)^T$, $(1,0)^T$. Outer products:
$$\begin{pmatrix}-1\\-1\end{pmatrix}\!(-1,-1) = \begin{pmatrix}1&1\\1&1\end{pmatrix},\quad \begin{pmatrix}0\\1\end{pmatrix}\!(0,1) = \begin{pmatrix}0&0\\0&1\end{pmatrix},\quad \begin{pmatrix}1\\0\end{pmatrix}\!(1,0) = \begin{pmatrix}1&0\\0&0\end{pmatrix},$$
sum $= \begin{pmatrix}2&1\\1&2\end{pmatrix}$, so $\boxed{\hat{\boldsymbol{\Sigma}}_{ML} = \frac{1}{3}\begin{pmatrix}2&1\\1&2\end{pmatrix} = \begin{pmatrix}2/3&1/3\\1/3&2/3\end{pmatrix}}$.
Symmetric ✓ (equal off-diagonals). PSD by §7.9's $2\times2$ test: $2/3 > 0$ and $\det = \frac{4}{9}-\frac{1}{9} = \frac{1}{3} > 0$ ✓.

**8.** (i) Posterior $= \text{Beta}(3+12,\ 3+8) = \boxed{\text{Beta}(15,11)}$.
(ii) $\hat p_{MAP} = \frac{15-1}{15+11-2} = \frac{14}{24} = \boxed{7/12 \approx 0.5833}$.
(iii) Posterior mean $= \frac{15}{15+11} = \boxed{15/26 \approx 0.5769}$.
(iv) $\hat p_{ML} = 12/20 = \boxed{0.6}$.
One line: the $\text{Beta}(3,3)$ prior (centered at $1/2$) pulled the estimate slightly down from the MLE $0.6$ toward $0.5$ — with only $3+3$ pseudo-observations against $20$ real ones, the pull is small.

**9.** Prior $\text{Beta}(1,1)$ = uniform on $(0,1)$: $f(p) = 1$. Posterior $\propto p^{n_1}(1-p)^{n_0}\cdot 1 = p^{n_1}(1-p)^{n_0}$, i.e. $\text{Beta}(1+n_1,\ 1+n_0)$ (conjugacy, §20.11). Its mode:
$$\hat p_{MAP} = \frac{(1+n_1)-1}{(1+n_1)+(1+n_0)-2} = \frac{n_1}{n_1+n_0} = \frac{n_1}{n} = \boxed{\hat p_{ML}}.\ \blacksquare$$
Equivalently from §20.10's boxed form: with $f(\theta)$ constant, $\arg\max_\theta[\log P(D\mid\theta) + \log f(\theta)] = \arg\max_\theta \log P(D\mid\theta)$.

**10.** $P(y_i \mid x_i) = N(wx_i, \sigma^2)$, so $R(w) = -\sum_{i=1}^3\log\left[\frac{1}{\sigma\sqrt{2\pi}}e^{-(y_i-wx_i)^2/(2\sigma^2)}\right]$
$$= \boxed{R(w) = \frac{1}{2\sigma^2}\sum_{i=1}^3 (y_i - wx_i)^2 + 3\log(\sigma\sqrt{2\pi})}.$$
(The second term is $w$-free.) $\frac{dR}{dw} = -\frac{1}{\sigma^2}\sum_{i=1}^3 x_i(y_i - wx_i) \stackrel{!}{=} 0$ ⇒ $\sum x_iy_i = w\sum x_i^2$ ⇒ $\boxed{\hat w_{ML} = \frac{\sum_{i}x_iy_i}{\sum_{i}x_i^2}}$.
Numerically: $\sum x_iy_i = 1\cdot2 + 2\cdot4 + 3\cdot5 = 25$; $\sum x_i^2 = 1+4+9 = 14$; $\boxed{\hat w_{ML} = 25/14 \approx 1.7857}$.
