# Chapter 44 solutions

Full worked solutions for the Chapter 44 problem set. (Problems live in `chapters/44-cnns.md`; never inline.) All numpy checks below were re-run here and are marked **[verified-NumPy]**; course numbers transcribed as **[recorded]**.

## Solution 1 — Parameter explosion, quantified

(i) Dense layer: weights $224 \cdot 224 \cdot 3 \cdot 100 = 15{,}052{,}800$, biases $100$. Total $15{,}052{,}900$ [verified-NumPy].

(ii) Conv layer: each filter is $3 \times 3 \times 3 = 27$ weights (as deep as the 3 input channels) plus 1 bias; 64 filters: $64 \cdot (27 + 1) = 1{,}792$ [verified-NumPy].

(iii) §41.14(ii) already warned that $39{,}760$ parameters made a one-hidden-layer MNIST net "prone to overfitting." The dense vision layer is $378\times$ that; the conv layer is $22\times$ *smaller* than the MNIST net. Parameter count alone says the dense version will memorize, not generalize — the conv architecture is the structural answer to §41.14(ii)'s warning.

## Solution 2 — The convolution, by hand

$$I = \begin{bmatrix} 1 & 0 & 2 \\\\ 3 & 1 & 1 \\\\ 0 & 2 & 1 \end{bmatrix}, \qquad K = \begin{bmatrix} 1 & 1 \\\\ 0 & -1 \end{bmatrix}, \qquad \text{stride } 1, \text{ valid} \Rightarrow 2 \times 2 \text{ output}.$$

(i) Entry by entry:
- $y_{0,0}$: patch $\begin{bmatrix} 1 & 0 \\\\ 3 & 1 \end{bmatrix}$ → $1\cdot1 + 0\cdot1 + 3\cdot0 + 1\cdot(-1) = 1 - 1 = 0$.
- $y_{0,1}$: patch $\begin{bmatrix} 0 & 2 \\\\ 1 & 1 \end{bmatrix}$ → $0 + 2 + 0 - 1 = 1$.
- $y_{1,0}$: patch $\begin{bmatrix} 3 & 1 \\\\ 0 & 2 \end{bmatrix}$ → $3 + 1 + 0 - 2 = 2$.
- $y_{1,1}$: patch $\begin{bmatrix} 1 & 1 \\\\ 2 & 1 \end{bmatrix}$ → $1 + 1 + 0 - 1 = 1$.

Output $\begin{bmatrix} 0 & 1 \\\\ 2 & 1 \end{bmatrix}$ [verified-NumPy].

(ii) The center pixel $I_{1,1} = 1$ sits in all four $2 \times 2$ windows, so it contributes to all four outputs (corners contribute to one each, edge-middles to two). That overlap is the mechanism of "local pattern matching": neighbouring output positions share evidence, so a pattern is judged by consensus across nearby patches, not by one isolated pixel.

(iii) $2 \times 2 = 4$ kernel weights plus 1 bias $= 5$ learned parameters. The same 5 numbers are reused at all four positions — weight sharing in one sentence.

## Solution 3 — Name the operation

(i) True (mathematical) convolution: $(I * K)(i,j) = \sum_m\sum_n I(i-m, j-n)\,K(m,n)$. Deep-learning "convolution": $(I * K)(i,j) = \sum_m\sum_n I(i+m, j+n)\,K(m,n)$ — cross-correlation, no flip.

(ii) If $K$ is learned, define $K'(m,n) = K(-m,-n)$ (the flipped kernel). The true convolution with $K$ equals cross-correlation with $K'$; since training searches over all $K$, it searches over all $K'$ too. The flip is absorbed into the learned values.

(iii) The deck's "Does it matter?" — **no**: a flip is a re-indexing of weights the optimizer would have found anyway.

## Solution 4 — Stride and padding

(i) $W_{\text{out}} = \lfloor(7 - 3 + 0)/2\rfloor + 1 = \lfloor 2 \rfloor + 1 = 3$.

(ii) With $P = 1$: $\lfloor(7 - 3 + 2)/2\rfloor + 1 = \lfloor 3 \rfloor + 1 = 4$. Padding bought one extra position per side.

(iii) Conv $3 \times 3$, stride 1, "same" on $32 \times 32$: $32 \times 32$ (same padding preserves size for odd $K$, stride 1). Then $2 \times 2$ max-pool, stride 2: $\lfloor(32-2)/2\rfloor + 1 = 16$ → $16 \times 16$ (channels unchanged by pooling).

## Solution 5 — Trace the MNIST stack

(i) Conv1: $\lfloor(28-3)/1\rfloor + 1 = 26$; Pool1: $\lfloor(26-2)/2\rfloor + 1 = 13$; Conv2: $\lfloor(13-3)/1\rfloor + 1 = 11$; Pool2: $\lfloor(11-2)/2\rfloor + 1 = \lfloor 4.5 \rfloor + 1 = 5$; Conv3: $\lfloor(5-3)/1\rfloor + 1 = 3$. Chain $28 \to 26 \to 13 \to 11 \to 5 \to 3$ [verified-NumPy].

(ii) Because the output size is a *count of window positions*, an integer: $\lfloor 4.5 \rfloor = 4$ full steps plus the starting position $= 5$. The last column of the $11 \times 11$ map is simply never covered by a $2 \times 2$ stride-2 window (it would start at index 10 and need index 11).

(iii) Conv2 with 128 filters on 32 input channels: $128 \cdot (3 \cdot 3 \cdot 32 + 1) = 128 \cdot 289 = 36{,}992$ weights [verified-NumPy]. The flatten size does *not* change: Conv3 still has 128 filters on the $5 \times 5$ map, giving $3 \times 3 \times 128 = 1152$.

## Solution 6 — Pooling vs PCA

(i) $16 \to 4$: 12 numbers discarded. What is lost is *position within each $2 \times 2$ patch* — the output records that the max was $9$, not that it sat at the bottom-left of its patch.

(ii) PCA (§24.5) *learns* a global linear projection keeping the top-variance directions and discards low-variance structure — but it flattens the image, losing the 2D layout entirely. Pooling is a *fixed, local, nonlinear* summary that *keeps* the spatial grid and discards within-patch position. PCA keeps variance directions and throws layout; pooling keeps layout and throws exact positions.

(iii) PCA over pooling: tabular or already-vectorized data with no spatial layout (e.g. compressing 100 correlated features to 10). Pooling over PCA: any image CNN — flattening for PCA would destroy the neighbour structure the conv layers need.

## Solution 7 — Equivariance, then invariance

(i) **Equivariance:** shifting the input shifts the output by the same amount. **Invariance:** shifting the input leaves the output (approximately) unchanged.

(ii) A conv layer is *equivariant* — "if a feature moves in the input, its representation moves correspondingly in the feature maps" [recorded]. Pooling *enhances invariance* — summarizing each patch means the exact position of a feature matters less [recorded].

(iii) Shift the cat 10 px right: every conv feature map shifts ~10 px right too (equivariance — the cat detector still fires, just 10 px over). After pooling, the pooled values barely change: the strongest response in each $2 \times 2$ patch is still the strongest, unless the shift moves a feature *across a pool boundary* — which is why pooling gives *approximate* invariance, not perfect.

## Solution 8 — Freeze vs fine-tune

(i) 200 images → **feature extraction**: freeze the backbone, train only the classifier. The backbone's millions of parameters cannot be re-estimated from 200 images; the classifier's few thousand can.

(ii) Mechanically: `requires_grad=False` tells autograd not to track or accumulate gradients for those parameters, so the optimizer has nothing to update for them. Note the forward pass is unchanged, and the *deltas still flow through* the frozen layers to reach earlier trainable layers if any exist — freezing stops $\partial\mathcal{L}/\partial W$ from being computed, not $\partial\mathcal{L}/\partial x$ from propagating.

(iii) The backbone's filters were learned on ImageNet-normalized inputs; its feature detectors expect that distribution. Feed raw $[0,255]$ or wrongly-scaled pixels and the activations land in a regime the weights never saw — the transferred features silently degrade. The notebook's rule: "the custom data going into the model should be prepared in the same way as the original training data."

## Solution 9 — The transfer-learning head

(i) The head must output 3 scores: a linear map from the feature-vector dimension to 3 (plus 3 biases). The pretrained 1000-class head is discarded and replaced.

(ii) `efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)` builds the pretrained model; then set `requires_grad=False` on every parameter in `model.features` to freeze the base [recorded].

(iii) Same LR everywhere risks **destroying the pretrained features** — large updates to carefully tuned backbone weights (in Chapter 43's terms: divergence/forgetting of a good initialization). The fine-tuning row's fix: unfreeze with a *reduced* learning rate so the backbone moves gently while the new head learns fast.

## Solution 10 — Design a small CNN

(i) One valid design ($3 \times 3$ convs, valid, stride 1; $2 \times 2$ pools, stride 2):
```
64x64x3 -> Conv1 (16 filters) -> 62x62x16 -> Pool -> 31x31x16
-> Conv2 (32 filters) -> 29x29x32 -> Pool -> 14x14x32
-> Conv3 (32 filters) -> 12x12x32 -> Pool -> 6x6x32
-> Flatten 1152 -> FC 10, softmax
```
Chain check: $(64-3)+1 = 62$; $(62-2)/2+1 = 31$; $(31-3)+1 = 29$; $(29-2)/2+1 = 14$; $(14-3)+1 = 12$; $(12-2)/2+1 = 6$; $6 \cdot 6 \cdot 32 = 1152 \le 2000$ ✓ [verified-NumPy].

(ii) Conv1: $16 \cdot (3\cdot3\cdot3 + 1) = 448$. Conv2: $32 \cdot (3\cdot3\cdot16 + 1) = 4{,}640$. Conv3: $32 \cdot (3\cdot3\cdot32 + 1) = 9{,}248$. Total conv parameters $= 14{,}336$ [verified-NumPy] — versus a dense $64\cdot64\cdot3 \to 1152$ layer's $14{,}156{,}928$ weights, same moral as §44.7.

(iii) §41.14(ii)'s guardrail: **dropout** (e.g. $p = 0.5$) before the output dense layer — with only 1152 flattened features and a small dataset, the dense head is the overfitting risk; dropout "modifies the network," complementing the conv layers' parameter efficiency.
