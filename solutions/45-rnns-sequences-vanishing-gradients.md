# Solutions — Chapter 45: RNNs — sequences and vanishing gradients

Full worked solutions. All numbers re-run in numpy [verified-NumPy].

## 1. Factor and cut

(i) Autoregressive factorization:
$$P(x_1,x_2,x_3,x_4) = P(x_1)\,P(x_2\mid x_1)\,P(x_3\mid x_1,x_2)\,P(x_4\mid x_1,x_2,x_3).$$

(ii) First-order Markov: each term conditions only on the immediately previous token:
$$P(x_1,x_2,x_3,x_4) \approx P(x_1)\,P(x_2\mid x_1)\,P(x_3\mid x_2)\,P(x_4\mid x_3).$$

(iii) Bigram: one distribution over $|V|$ next-words for each of $|V|$ histories → $|V|^2 = 10{,}000^2 = 10^8$ entries. Trigram: $|V|^2$ histories → $|V|^3 = 10^{12}$ entries. Ratio $10^{12}/10^8 = 10^4 = |V|$: every extra token of memory multiplies the table by the vocabulary size — the deck's "grows exponentially" made concrete.

## 2. One recurrent step, by hand

(i) $h_1 = \tanh(0.8\cdot1.0 + (-0.3)\cdot0 + 0.1) = \tanh(0.9) = 0.7163$.

(ii) $h_2 = \tanh(0.8\cdot(-1.0) + (-0.3)(0.7163) + 0.1) = \tanh(-0.8 - 0.2149 + 0.1) = \tanh(-0.9149) = -0.7235$.

(iii) The memory term is $w_{hh}\,h_{t-1} = -0.3 \times 0.7163 = -0.2149$ — the only channel by which $x_1$ influences $h_2$. With $w_{hh} = 0$: $h_2 = \tanh(-0.8 + 0.1) = \tanh(-0.7) = -0.6044$; the network becomes memoryless, each step a function of its own input only.

## 3. The sharing payoff, counted

(i) $W_{xh}$: $50 \times 100 = 5{,}000$; $W_{hh}$: $100 \times 100 = 10{,}000$; $b_h$: $100$. Total $15{,}100$.

(ii) One timestep's MLP: $50\cdot100 + 100 = 5{,}100$; thirty independent copies: $30 \times 5{,}100 = 153{,}000$ — about $10\times$ more, and it still can't share a pattern learned at position $5$ with position $25$ (§45.2(iii)).

(iii) The RNN's $15{,}100$ does not move as $T$ grows — same weights, every timestep — the temporal twin of §44.7's $8{,}400\times$: sharing is what makes the count independent of the input size.

## 4. The two-component recursion

(i) In the unrolled graph, $h_t$ has two outgoing edges: one down to this step's output $y_t$ (via $W_{hy}$), and one right to the next hidden state $h_{t+1}$ (via $W_{hh}$). The loss reaches $h_t$ along both paths — the chain rule sums over all paths to a node (§42.2's rule), so $\partial L/\partial h_t$ = (error from $y_t$) + (error from $h_{t+1}$).

(ii) At $t = T$ there is no $h_{T+1}$: the "future loss" term vanishes, leaving $\partial L/\partial h_T = (\partial L/\partial y_T)(\partial y_T/\partial h_T)$.

(iii) Feedforward $\delta^{[l]}$ recursion: the error crosses layers through $(W^{[l+1]})^T$ (§42.4). BPTT: the error crosses time through $W_{hh}^T\,\mathrm{diag}(\phi')$ (§45.7's product) — same chain-rule job, different axis.

## 5. Finish the BPTT eg

(i) $y_t = w_{hy} h_t$ gives $\partial y_t/\partial w_{hy} = h_t$, so by the chain rule $\partial L/\partial w_{hy} = \sum_{t=1}^{2} \frac{\partial L}{\partial y_t}\,h_t$ (one shared weight, two votes — §45.5). Numerically: $(-0.3907)(0.7616) + (-0.8567)(0.9292) = -0.2976 - 0.7960 = -1.0936$.

(ii) $h_t = \tanh(w_{xh} x_t + w_{hh} h_{t-1})$ (biases $0$) gives $\partial h_t/\partial w_{xh} = (1-h_t^2)\,x_t$, so $\partial L/\partial w_{xh} = \sum_t \frac{\partial L}{\partial h_t}(1-h_t^2)x_t$. With $\partial L/\partial h_1 = -0.1657$ (current $-0.1563$, future $-0.0094$), $\partial L/\partial h_2 = -0.3427$: $(-0.1657)(1-0.7616^2)(2) + (-0.3427)(1-0.9292^2)(3) = (-0.1657)(0.4200)(2) + (-0.3427)(0.1366)(3) = -0.1392 - 0.1404 = -0.2796$.

(iii) $\left.\partial L/\partial w_{hh}\right|_{t=1}$ carries the local factor $h_0 = 0$ — no signal passed through the memory edge at $t{=}1$, so nothing to blame (the §42.3 Note logic). $\left.\partial L/\partial w_{xh}\right|_{t=1}$ carries the local factor $x_1 = 2 \ne 0$ — the input edge was live, so it earns its gradient.

## 6. Name the regime

(i) $0.64^{20} \approx 1.33\times10^{-4}$ — about one ten-thousandth survives: **vanishing**.

(ii) $1.5^{20} \approx 3325$ — the gradient compounds to thousands: **exploding**.

(iii) (i) is the deck's vanishing row ("gradients for early time steps become nearly zero... cannot learn long-range dependencies"); (ii) is the exploding row ("gradients become enormous (NaN or infinity)"). §42.8(ii) is the same story along depth: a saturated sigmoid multiplies the error by at most $\approx 0.045$ per layer, so deep stacks strangle the signal the way long sequences do here — the two halves of one phenomenon.

## 7. Clip vs truncate

(i) Clipping fixes **explosion** (caps the update size when $\|\nabla\|$ exceeds the threshold); it does **not** fix vanishing — a near-zero gradient stays near-zero after rescaling — and it does not shorten the $T$-step product.

(ii) Dependencies longer than $\tau = 5$ steps: the backward chain is cut, so no gradient path connects $x_t$ to $y_{t+6}$.

(iii) The hidden state still carries information forward through the full sequence ("maintaining its long-term memory"), so the *model* can in principle use long context — but the *gradient* can't reach back past $\tau$, so the model can never *learn* to use it. Memory without learnability: the compromise.

## 8. The LSTM's gradient path

(i) $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t = (0.9\cdot1.0 + 0.5\cdot0.6,\ 0.2\cdot2.0 + 0.8\cdot(-0.4)) = (0.9 + 0.3,\ 0.4 - 0.32) = (1.2,\ 0.08)$.

(ii) $\partial c_t/\partial c_{t-1} = f_t = (0.9,\ 0.2)$ element-wise — the additive update's derivative is the forget gate, no weight matrix involved.

(iii) Over $n$ steps the first dimension's gradient scales as $0.9^n$ ($0.9^{10} \approx 0.35$ — a third survives ten steps) while the second scales as $0.2^n$ ($0.2^{10} \approx 10^{-7}$ — gone). The gate *value* is the gradient highway: open ($f \approx 1$) preserves, closed ($f \approx 0$) erases — and the network learns which, per dimension.

## 9. The GRU's mix

(i) $H_t = (0.8\cdot1.0 + 0.2\cdot0.0,\ 0.1\cdot(-1.0) + 0.9\cdot2.0) = (0.8,\ -0.1 + 1.8) = (0.8,\ 1.7)$.

(ii) $Z_t \approx 1$: the old hidden state is (almost) copied through — long-term memory preserved. $Z_t \approx 0$: the old state is discarded in favor of the candidate $\tilde H_t$ — fresh start.

(iii) The **forget gate** $f_t$ of the LSTM: both are sigmoids deciding "how much of the past survives," and both enter an *additive* update ($Z_t \odot H_{t-1}$ here, $f_t \odot c_{t-1}$ there) — the deck's "acts similarly to the forget gate in an LSTM."

## 10. Generate and judge

(i) Predicted unigrams $\{A,B\}$: $p_1 = 2/2 = 1.0$. Predicted bigrams $\{AB\}$: $p_2 = 1/1 = 1.0$. Brevity penalty: $\exp(1 - 4/2) = \exp(-1) \approx 0.3679$. $\mathrm{BLEU} = 0.3679 \times (1.0\times1.0)^{1/2} = 0.3679$.

(ii) Without the penalty the score would be $(1.0\times1.0)^{1/2} = 1.0$ — "misleadingly perfect": pure n-gram precision rewards a two-word prefix of the target as if it were the whole answer. The penalty prices length.

(iii) Mismatch: training never shows the decoder its own mistakes (it always gets the true previous token), while inference runs entirely on its own — possibly erroneous — outputs, so errors compound (the §45.2(ii) error-accumulation story, learned). Beam search does not fix it: beam search is a *decoding* strategy (better search over the model's distribution), not a *training* fix; the mismatch is baked in at training time.
