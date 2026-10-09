# Chapter 42 solutions — Backpropagation, worked end-to-end

All numbers below were re-run in numpy [verified-NumPy] unless marked [recorded] (transcribed from the course materials).

## Problem 1 — the deck's scalar graph, re-derived

**(i) Forward pass.** Nodes: $d = a + b$, $\mathcal{L} = d \cdot c$.
$$d = 2 + 3 = 5, \qquad \mathcal{L} = 5 \cdot 6 = 30.$$
Cached: $a=2,\ b=3,\ c=6,\ d=5,\ \mathcal{L}=30$.

**(ii) Backward pass.** Start $\partial\mathcal{L}/\partial\mathcal{L} = 1$ ("by definition," slide 63).
- At the $\times$ node: local gradients $\partial\mathcal{L}/\partial d = c = 6$, $\partial\mathcal{L}/\partial c = d = 5$. So $\partial\mathcal{L}/\partial c = 1 \cdot 5 = 5$, $\partial\mathcal{L}/\partial d = 1 \cdot 6 = 6$.
- At the $+$ node: local gradients $\partial d/\partial a = \partial d/\partial b = 1$. So $\partial\mathcal{L}/\partial a = 6 \cdot 1 = 6$, $\partial\mathcal{L}/\partial b = 6 \cdot 1 = 6$.

**(iii) Direct check.** $\mathcal{L} = (a+b)c$, so $\partial\mathcal{L}/\partial c = a + b = 5$ — matches. (Similarly $\partial\mathcal{L}/\partial a = c = 6$.)

## Problem 2 — the three components

**(i)** $\partial L/\partial a^{[l]}$ = **1. Upstream Gradient** — "How the final loss changes with this layer's output. This is passed back from layer $l+1$." $\partial a^{[l]}/\partial z^{[l]} = g'(z^{[l]})$ = **2. Activation Gradient** — "The derivative of the activation function." $\partial z^{[l]}/\partial W^{[l]}$ = **3. Local Gradient** — "How this layer's pre-activation changes with its weights."

**(ii)** $\delta^{[l]} = \partial L/\partial z^{[l]}$ absorbs (i) and (ii): $\delta^{[l]} = (\partial L/\partial a^{[l]}) \odot g'(z^{[l]})$. Useful because $\delta^{[l]}$ is the *single* quantity each layer needs — it is computed once per layer and then reused for both $\partial L/\partial W^{[l]}$ and $\partial L/\partial b^{[l]}$, and it is what gets passed one step further back.

**(iii)** $\delta^{[l]} = \big((W^{[l+1]})^T \delta^{[l+1]}\big) \odot g'(z^{[l]})$ (the deck's boxed recursion; for the output layer, $\delta^{[L]} = \nabla_{a^{[L]}} L \odot g'(z^{[L]})$).

## Problem 3 — batch-form translation

**(i)** Per-example: $\partial L/\partial W^{[l]} = \delta^{[l]}(a^{[l-1]})^T$ with $\delta^{[l]}$ $S_l \times 1$, $a^{[l-1]}$ $S_{l-1} \times 1$. The batch layout stores examples as rows: $A^{l-1}$ is $n \times S_{l-1}$, $\Delta^l$ is $n \times S_l$. Summing the per-example outer products over the batch:
$$\frac{\partial L}{\partial W^l} = \sum_{i=1}^n \Delta^l_i (A^{l-1}_i)^T = (A^{l-1})^T \Delta^l,$$
an $S_{l-1} \times S_l$ matrix — the shape of $W^l$.

**(ii)** In per-example form the weights left-multiply ($W^{[l]} a^{[l-1]}$), so the outer product is $\delta (a)^T$; in batch form they right-multiply rows ($A^{l-1} W^l$), so the sum-of-outer-products assembles as $(A^{l-1})^T \Delta^l$ — the transpose moves to keep each factor's rows/columns aligned with $W^l$'s $(S_{l-1} \times S_l)$ layout (§41.4's Note).

**(iii)** In §41.10, `dz2` already carries the $1/m$: `dz2 = (out - y)/m`. So `dz2.sum(axis=0)` $= \sum_i \delta_i / m$ = the row-*mean* of the per-example $\delta$'s. It says `sum` but computes a mean, because the $1/m$ was applied one line earlier. (The notebook does it the other way: unscaled $\Delta$, then `/ m` at the `dW`/`db` lines — identical result.)

## Problem 4 — the XOR backward pass, a second example

Weights as in §42.3. Input $(0, 1)$, target $y = 1$.

**(i) Forward.**
$$z_1 = [0(1)+1(1)+0,\ 0(-1)+1(-1)+1] = [1.0,\ 0.0], \quad h_1 = [0.7311,\ 0.5],$$
$$z_3 = 2(0.7311) - 1(0.5) - 1 = -0.0379, \quad \hat y = 0.4905, \quad L = -\log(0.4905) = 0.7123.$$
*Coincidence worth noting:* the forward values are identical to the $(1,0)$ case — with these hand-picked weights, $x_1 + x_2$ always equals $1$ on the two $y=1$ corners. The gradients will differ only through the $x$ factor.

**(ii) Backward.**
$$\frac{\partial L}{\partial z_3} = \hat y - y = -0.5095, \quad \frac{\partial L}{\partial z_1} = -0.2003,\ \frac{\partial L}{\partial z_2} = 0.1274$$
(same deltas as §42.3 — same $\hat y$, same $h_1$). Weight gradients now use $x = (0, 1)$:
$$\frac{\partial L}{\partial W_1} = x^T \delta = \begin{bmatrix} 0 & 0 \\ -0.2003 & 0.1274 \end{bmatrix}, \quad \frac{\partial L}{\partial b_1} = [-0.2003,\ 0.1274],$$
$$\frac{\partial L}{\partial W_2} = [-0.3725,\ -0.2547]^T, \quad \frac{\partial L}{\partial b_2} = -0.5095.$$

**(iii)** Now $x_1 = 0$, so $w_{11}$ and $w_{12}$ (the weights fed by $x_1$) get zero gradient — the mirror image of §42.3's Note, where $x_2 = 0$ froze $w_{21}, w_{22}$.

## Problem 5 — the $(g - y)$ check

**(i)** At $\hat y = 0.4905$, $y = 1$:
$$\frac{\partial L}{\partial \hat y} = -\frac{1}{0.4905} = -2.0387, \qquad \frac{\partial \hat y}{\partial z} = 0.4905(1 - 0.4905) = 0.2499,$$
$$(-2.0387)(0.2499) = -0.5095 = 0.4905 - 1 = \hat y - y \quad \checkmark$$
[verified-NumPy].

**(ii)** `dz2 = (self.out - y).reshape(-1, 1) / m` — the entire sigmoid-derivative-times-loss-derivative two-step collapses to the subtraction. The `/ m` is there because the loss is the *mean* over the batch: each example's $\delta$ is its share $1/m$ of the mean.

**(iii)** The shortcut is unavailable whenever the output activation and loss are *not* a cancelling pair — e.g. a ReLU (or tanh) output with cross-entropy, or sigmoid with squared error: then use the general $\delta^{[L]} = \nabla_{a^{[L]}} L \odot g'(z^{[L]})$ with no cancellation.

## Problem 6 — softmax pairing

**(i)** $z = (1.0, 0.0, -1.0)$: $e^z = (2.7183, 1.0, 0.3679)$, sum $= 4.0862$,
$$\hat y = (0.6652,\ 0.2447,\ 0.0900), \quad L = -\log(0.2447) = 1.4076,$$
$$\frac{\partial L}{\partial z} = \hat y - y = (0.6652,\ -0.7553,\ 0.0900) \quad [verified-NumPy].$$

**(ii)** Two-step for $j = 1$ (the true class): $\partial L/\partial \hat y_k = -y_k/\hat y_k$ is nonzero only at $k = 1$: $-1/0.2447 = -4.0862$. $\partial \hat y_1/\partial z_1 = \hat y_1(1 - \hat y_1) = 0.2447 \times 0.7553 = 0.1848$. Product: $(-4.0862)(0.1848) = -0.7553$ — matches $\hat y_1 - y_1$ $\checkmark$.

**(iii)** What cancels: the softmax Jacobian's $\hat y_i(\delta_{ij} - \hat y_j)$ structure against the log-loss's $1/\hat y_k$ — every term with $k \ne j$ folds into $-\hat y_j \sum_k y_k = -\hat y_j$, leaving $\hat y_j - y_j$.

## Problem 7 — gradient check by hand

**(i)** $z = 0.5(2.0) + 0.1 = 1.1$; $\hat y = \sigma(1.1) = 0.75026$; $L = -\log(0.75026) = 0.28734$.
$$\frac{\partial L}{\partial z} = \hat y - y = -0.24974, \quad \frac{\partial L}{\partial w} = (-0.24974)(2.0) = -0.49948, \quad \frac{\partial L}{\partial b} = -0.24974.$$

**(ii)** $\varepsilon = 10^{-5}$:
$$\text{FD}_w = -0.499479789,\ \text{FD}_b = -0.249739894 \quad \Rightarrow \quad \text{agreement to } \sim 10^{-10} \ \checkmark \ [verified-NumPy].$$

**(iii)** If $\hat y$ hits exactly $0$ or $1$, $\log \hat y$ (or $\log(1-\hat y)$) is undefined/$-\infty$ and the finite-difference evaluation explodes — hence the `np.clip(yhat, 1e-15, 1 - 1e-15)` in every loss implementation (§41.10).

## Problem 8 — credit where it's due

**(i)** $\partial L/\partial w_{21} = (\partial L/\partial z_1)(\partial z_1/\partial w_{21}) = (\partial L/\partial z_1) \cdot x_2$, and $x_2 = 0$ — the chain multiplies by the local gradient, which is the input value itself.

**(ii)** Yes — on input $(0, 1)$ we have $x_2 = 1$, so $\partial L/\partial w_{21} = (\partial L/\partial z_1) \cdot 1 \ne 0$ and $w_{21}$ moves (with the pre-update weights the numbers are Problem 4's: $-0.2003$; after the update the deltas shift slightly but stay nonzero).

**(iii)** A weight only learns from examples where its input is nonzero: each weight's gradient is its delta times its input, so an example teaches exactly the weights on its own active input paths. Sparse/zero inputs mean sparse updates — the same logic behind embedding layers only updating the looked-up rows.

## Problem 9 — vanishing, read backwards

**(i)** $\sigma(3) \approx 0.9526$, $g'(3) = 0.9526 \times 0.0474 \approx 0.0452$.

**(ii)** Each backward step multiplies by $g' \le 0.0452$, so over four steps $\|\delta^{[1]}\| \lesssim (0.0452)^4 \|\delta^{[5]}\| \approx 4.2 \times 10^{-6}\,\|\delta^{[5]}\|$ — the error signal is attenuated by a factor of $\sim 240{,}000$ [verified-NumPy].

**(iii)** This is §41.7's "flat at both ends... gradient near zero for large inputs, effectively stops learning," now with a number on it: saturated sigmoids don't just slow learning, they shrink the gradient *exponentially* in depth. ReLU's $g' = 1$ for $z > 0$ passes the error through unshrunk — hence "start with ReLU."

## Problem 10 — map the notebook

**(i)** Notebook cell-11 `backward`, with $m$ = batch size:
| notebook line | §42.4 formula |
|---|---|
| `dL_dy = self.y_pred - y_true` | per-example $\delta$ for the sigmoid+BCE pairing (§42.5(i)) |
| `dL_dz2 = dL_dy` (+ reshape) | $\Delta^2$ (unscaled; the $/m$ comes later) |
| `dL_dW2 = np.dot(self.h1.T, dL_dz2) / m` | $\partial L/\partial W^2 = (A^1)^T \Delta^2$, mean over batch |
| `dL_db2 = np.mean(dL_dz2, axis=0)` | $\partial L/\partial b^2$ = row-mean of $\Delta^2$ |
| `dL_dh1 = np.dot(dL_dz2, self.W2.T)` | upstream $\partial L/\partial A^1 = \Delta^2 (W^2)^T$ |
| `dL_dz1 = dL_dh1 * self.sigmoid_derivative(self.z1)` | $\Delta^1 = \text{upstream} \odot g'(Z^1)$ |
| `dL_dW1 = np.dot(X.T, dL_dz1) / m` | $\partial L/\partial W^1 = (A^0)^T \Delta^1$, mean over batch |
| `dL_db1 = np.mean(dL_dz1, axis=0)` | $\partial L/\partial b^1$ = row-mean of $\Delta^1$ |

**(ii)** For sigmoid output + BCE, the true $\partial L/\partial z_2 = (\partial L/\partial\hat y)\,\hat y(1-\hat y)$ *would* need the $y(1-y)$ factor — but the product collapses to $\hat y - y$, so the code skips the two-step and writes the answer directly (§42.5(i)).

**(iii)** Notebook: $\Delta$ unscaled, gradients $= (1/m) \sum_i$. §41.10: $\Delta$ pre-scaled by $1/m$ at `dz2`, gradients $= \sum_i$. Since $\sum_i (\delta_i/m) = (1/m)\sum_i \delta_i$, the two are algebraically identical — the $1/m$ commutes with the sum. (Note §41.10's `dh1 = dz2 @ W2.T` then carries the $1/m$ through the whole backward pass consistently.)
