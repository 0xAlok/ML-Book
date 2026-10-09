# Solutions — Chapter 43: Optimization for neural nets

## 1. The ravine, by hand

$f(x_1, x_2) = 0.1x_1^2 + 2x_2^2$, start $(-4, 1)$.

**(i)** $\nabla f = (0.2x_1,\ 4x_2)$. One GD step $x \gets x - \eta\nabla f$:
$$x_1 \gets x_1 - \eta(0.2x_1) = (1 - 0.2\eta)x_1, \qquad x_2 \gets x_2 - \eta(4x_2) = (1 - 4\eta)x_2.$$

**(ii)** $\eta = 0.4$: $x_1 = -4(1-0.08) = -3.68$, $x_2 = 1(1-1.6) = -0.60$.
$\eta = 0.6$: $x_1 = -4(1-0.12) = -3.52$, $x_2 = 1(1-2.4) = -1.40$.
Both match the §43.11 traces exactly [verified-NumPy].

**(iii)** The $x_2$ factor $|1-4\eta| > 1$ when $\eta > 0.5$ (or $\eta < 0$) — then $|x_2|$ grows every step: divergence. Fast progress in $x_1$ wants $(1-0.2\eta)$ near $0$, i.e. $\eta \approx 5$; stability in $x_2$ wants $\eta < 0.5$. The two requirements don't overlap — that non-overlap *is* ill-conditioning: no single $\eta$ is both fast-flat and stable-steep.

## 2. Velocity is a leaky average

**(i)** Base: $v_0 = \beta v_{-1} + g_0 = g_0$ (taking $v_{-1} = 0$). Inductive step: assume $v_{t-1} = \sum_{\tau=0}^{t-1}\beta^{t-1-\tau}g_\tau$; then
$$v_t = \beta\sum_{\tau=0}^{t-1}\beta^{t-1-\tau}g_\tau + g_t = \sum_{\tau=0}^{t}\beta^{t-\tau}g_\tau. \quad \blacksquare$$

**(ii)** The weights are a geometric series: $\sum_{i=0}^{t}\beta^i \to \sum_{i=0}^{\infty}\beta^i = 1/(1-\beta)$ for $|\beta| < 1$. For $\beta = 0.9$: $1/0.1 = 10$ (partial sum to $k=60$ is $9.984$ [verified-NumPy]).

**(iii)** $v_t \to g\sum_{i=0}^{\infty}\beta^i = g/(1-\beta)$, so the update tends to $x_{t+1} - x_t \to -\eta g/(1-\beta)$: momentum with a constant gradient behaves like plain GD with effective learning rate $\eta/(1-\beta)$ — $10\eta$ for $\beta = 0.9$.

## 3. Mini-batch noise

**(i)** Sampling $i$ uniformly, $\mathbb{E}[\nabla f_i] = \frac{1}{n}\sum_{i=1}^{n}\nabla f_i = \nabla f$ — unbiased. For $b$ i.i.d. draws, $\mathrm{Var}\big(\frac{1}{b}\sum \nabla f_i\big) = \frac{1}{b^2}\cdot b\,\mathrm{Var}(\nabla f_i) = \mathrm{Var}/b$.

**(ii)** Computational: blocks fit in cache and parallelize on CPUs/GPUs (the deck's vectorization slide). Statistical: averaging divides gradient noise by $b$.

**(iii)** "May converge to sharp minima (worse generalization)" and needs more memory (notebook's batch-size table [recorded]).

## 4. AdaGrad vs RMSProp, one step each

At $(-4, 1)$: $g = (0.2\cdot(-4),\ 4\cdot 1) = (-0.8,\ 4)$.

**(i) AdaGrad, $\eta = 2.0$.** $s_1 = (0.64,\ 16)$. Effective LRs: $2/\sqrt{0.64} = 2/0.8 = 2.5$; $2/\sqrt{16} = 2/4 = 0.5$. Update $= -(2.5\cdot(-0.8),\ 0.5\cdot 4) = (2.0,\ -2.0)$. New $x = (-4+2,\ 1-2) = (-2,\ -1)$.

**(ii) RMSProp, $\eta = 0.1$, $\beta = 0.9$.** $E[g^2]_1 = 0.1\cdot(0.64,\ 16) = (0.064,\ 1.6)$. Effective LRs: $0.1/\sqrt{0.064} \approx 0.3953$; $0.1/\sqrt{1.6} \approx 0.0791$. Update $= -(0.3953\cdot(-0.8),\ 0.0791\cdot 4) \approx (0.3162,\ -0.3162)$. New $x \approx (-3.6838,\ 0.6838)$.

**(iii)** AdaGrad steps the flat direction $5\times$ harder per unit gradient ($2.5$ vs $0.5$); RMSProp does the same ($0.3953$ vs $0.0791$, also $5\times$). That is the preconditioner story: each direction's step is divided by its own typical gradient size, so the $20\times$ scale gap between the directions is cancelled instead of fought.

## 5. Adam's bias correction, with and without

**(i)** $v_1 = 0.1\cdot 0.5 = 0.05$; $s_1 = 0.001\cdot 0.25 = 0.00025$. $\hat v_1 = 0.05/0.1 = 0.5$; $\hat s_1 = 0.00025/0.001 = 0.25$. Corrected step $= \eta\cdot 0.5/\sqrt{0.25} = 1.0\,\eta$.

**(ii)** Uncorrected: $\eta\cdot 0.05/\sqrt{0.00025} = \eta\cdot 0.05/0.015811 \approx 3.162\,\eta$.

**(iii)** Without correction the first steps are inflated by a factor ($3.16\times$ here) that comes from the zero initialization of the moments, not from the data — early training overshoots for no reason.

## 6. Weight decay = L2, until Adam

**(i)** $\nabla J_\lambda = \nabla J + \lambda\theta = g_t + \lambda\theta_t$. Plain SGD:
$$\theta_{t+1} = \theta_t - \eta(g_t + \lambda\theta_t) = (1-\eta\lambda)\theta_t - \eta g_t. \quad \blacksquare$$

**(ii)** Adam divides the *entire* gradient — including the $\lambda\theta_t$ piece — by $\sqrt{\hat s_t} + \epsilon$, which is different for every parameter. So the $\lambda\theta_t$ shrink is no longer the uniform $(1-\eta\lambda)$ factor; heavily-updated parameters get less decay, rarely-updated ones more.

**(iii)** AdamW: $\theta_{t+1} = \theta_t - \eta\big(\hat v_t/(\sqrt{\hat s_t}+\epsilon) + \lambda\theta_t\big) = (1-\eta\lambda)\theta_t - \eta\,\hat v_t/(\sqrt{\hat s_t}+\epsilon)$. The $+\lambda\theta_t$ sits *outside* the adaptive fraction — that term restores the uniform per-step shrink.

## 7. Read the schedules

**(i)** $\eta(500)$: exponential $= 0.2e^{-0.01\cdot 500} = 0.2e^{-5} \approx 0.2(0.006738) \approx 0.0013$. Inverse time $= 0.2/(1+0.01\cdot 500) = 0.2/6 \approx 0.0333$. Cosine $= 0 + \tfrac12(0.2)(1+\cos(\pi\cdot 500/1000)) = 0.1(1+\cos(\pi/2)) = 0.1(1+0) = 0.1$.

**(ii)** Cosine: at $t = T_{\max}$, $\cos(\pi) = -1$, so $\eta = \eta_{\min} + 0 = 0$ exactly — $T_{\max}$ is defined as the horizon where the curve lands on $\eta_{\min}$.

**(iii)** sklearn: $\eta_0/t^{p}$; deck: $\eta_0/(1+kt)$. sklearn's diverges at $t = 0$ (implementations index steps from $t \ge 1$ or equivalent); the deck's equals $\eta_0$ at $t = 0$. For large $t$ with $p = k = 1$: $\eta_0/t$ vs $\eta_0/(kt)$ — the same $1/t$ tail up to the constant; the difference is negligible.

## 8. Init scale through the $\delta$ recursion

**(i)** $g'(z) = \sigma(z)(1-\sigma(z))$, symmetric in $z$, maximal $0.25$ at $z = 0$, decreasing as $|z|$ grows. At $|z| = 3$: $\sigma(3) \approx 0.9526$, so $g'(3) \approx 0.9526\cdot 0.0474 \approx 0.0451$; for $|z| \ge 3$, $g'(z) \le 0.0451$.

**(ii)** $\delta^{[l]} = ((W^{[l+1]})^T\delta^{[l+1]})\odot g'(z^{[l]})$ (§42.4). With $\lVert W\rVert \approx 1$ and $g' \le 0.0451$ at each of the three backward steps from layer 4 to layer 1:
$$\lVert\delta^{[1]}\rVert \lesssim (0.0451)^3\,\lVert\delta^{[4]}\rVert \approx 9.2\times 10^{-5}\,\lVert\delta^{[4]}\rVert.$$
The error signal is crushed to $\sim 10^{-4}$ of itself before reaching the first layer — before training has even started.

**(iii)** Rule: initial weights must be small enough that pre-activations stay where the activation's slope is healthy — $|z| \gtrsim 3$ everywhere in a sigmoid stack means the gradients are dead on arrival. (The course gives no formula for that scale — the criterion, not the constant, is the takeaway; see §43.10's flagged gap.)

## 9. Scaling as preconditioning

**(i)** The $x_2$ direction: its gradients are $\sim 1000\times$ larger, so the loss surface is $1000\times$ steeper along $x_2$ — the same ravine as §43.1(i), built from units.

**(ii)** Roughly: both features end up mean $0$, standard deviation $1$, spanning about $[-3, 3]$ each — the $1000\times$ gap is gone.

**(iii)** AdaGrad would discover the scale gap through its accumulators ($s_t$ growing $10^6\times$ faster for $x_2$) and compensate per-parameter; `StandardScaler` removes the gap before the optimizer runs, so no compensation is needed. Scaling is the preconditioner you apply once to the data instead of every step to the gradients.

## 10. Triage

**(i)** Causes: learning rate too small; poor initialization. Fixes: increase the LR; use proper initialization (notebook's pitfalls table [recorded]).

**(ii)** Cause: Adam's interaction with L2/weight decay — the decay goes through the adaptive scaling and isn't true weight decay. Fix: switch to AdamW.

**(iii)** AdamW, LR $0.001$ (low, to preserve the pretrained features), weight decay $0.001$ (the notebook's fine-tuning row: $0.0001$–$0.001$). AdamW is the recorded industry standard for transformers because its decoupled decay regularizes uniformly; the low LR keeps fine-tuning from destroying what pretraining learned.
