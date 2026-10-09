# Chapter 45: RNNs — sequences and vanishing gradients

Chapter 44 shared weights across *space* — one filter, every pixel. This chapter shares them across *time*: the same cell, every timestep (§44.10's promise). The spine is the GenAI Weeks 7–8 deck ("Introduction to Recurrent Neural Networks," Balaji Srinivasan and Ganapathy Krishnamurthi, 80 slides — sequence modelling, the RNN, BPTT, vanishing/exploding gradients, LSTM, GRU, bidirectional and encoder–decoder architectures, BLEU, beam search) plus its two companion notebooks: *Time_Series_Analysis_using_RNN* (AMZN stock prediction: data prep, a `torch.nn.RNN` model, Adam at $\eta = 0.001$ for 50 epochs) and *sentiment-analysis* (IMDB: `SentimentRNN` and `SentimentLSTM` with embedding, dropout, and a sigmoid head). Every numpy number below was re-run here and is marked **[verified-NumPy]**; deck, notebook, and torch numbers are transcriptions, marked **[recorded]**.

**Notation.** The deck writes the timestep as a function argument, $h(t), x(t)$; this book writes it as a subscript, $h_t, x_t$ — the same objects. Weight names follow the deck: $W_{xh}$ (input→hidden), $W_{hh}$ (hidden→hidden, the recurrent weights), $W_{hy}$ (hidden→output), biases $b_h, b_y$, activation $\phi$. LSTM gates keep the deck's letters: $f_t, i_t, \tilde c_t, c_t, o_t$. The deck's GRU slides switch to row-vector notation ($X_t W_{xr} + H_{t-1} W_{hr}$) while its LSTM slides use column concatenation ($W_f \cdot [h_{t-1}, x_t]$) — a layout change, not a math change; §45.9 flags it.

## 45.1 Sequence data: order matters

**Def (the deck's).** "Sequential data is a type of data where the order of the data points is significant. The sequence itself contains important information. Each data point is dependent on the previous ones (e.g., a sentence)." Three characteristics: **Ordered** ("Data points have a specific position or index in a sequence"), **Variable Length** ("Sequences can have different lengths (e.g., different sentence lengths)"), and **Dependencies** ("Values can depend on other values at different positions in the sequence").

**The modelling target.** "A sequence model aims to estimate the joint probability of an entire sequence of observations" — for $x_1, x_2, \dots, x_T$, the model estimates $P(x_1, x_2, \dots, x_T)$, "to assess how plausible a sequence is, or to generate new, similar sequences" [recorded]. Applied to text this is a **language model**: "When a sequence model is applied to text data, it is called a language model. Its goal is to estimate the probability of a sequence of words or characters."

**The autoregressive factorization.** Predicting $x_t$ from the past, the joint probability factors by the chain rule of probability (the deck's term):
$$\boxed{P(x_1, \dots, x_T) = P(x_1) \prod_{t=2}^{T} P(x_t \mid x_1, \dots, x_{t-1})} \quad \text{[recorded]}.$$
"In words: the probability of a sequence is the product of the conditional probabilities of each element, given the elements that came before it." Models that "predict future values in a sequence based on past values" this way are **autoregressive**. The catch is stated immediately: "The conditioning term $P(x_t \mid x_1, \dots, x_{t-1})$ becomes computationally expensive and statistically difficult to model as the sequence length $t$ grows" — the full history is too much to carry raw.

**Basically, ...** "Order is the data: shuffle a sentence and it dies. The model's job is to score whole sequences — how plausible is this one? The chain rule turns that into 'predict each token from everything before it', which is the autoregressive game. The problem: 'everything before it' gets long, and long is expensive."

## 45.2 Before the RNN: Markov, windows, and why they break

Three simplifications, each buying tractability and each paying for it.

**i) The Markov assumption / n-grams.** Cut the history: "Assumes that the current state $x_t$ only depends on the immediately preceding state $x_{t-1}$," so $P(x_1, \dots, x_T) \approx P(x_1)\prod_{t=2}^{T} P(x_t \mid x_{t-1})$ — the first-order Markov model, "also known as a bigram model in the context of language processing," extendable to "a fixed-length window of previous states (e.g., trigrams, etc.)" [recorded]. The price: "Markov models and n-grams have a fixed memory; they only look at the last $n - 1$ tokens. To remember more, the number of model parameters grows exponentially, making them impractical for capturing long-range dependencies" [recorded]. Count it: a bigram table over a vocabulary of size $|V|$ needs $|V|^2$ entries; a trigram $|V|^3$ — each extra token of memory multiplies the table by $|V|$.

**ii) Linear regression on a fixed window.** "It follows an n-gram Markovian assumption. It uses a fixed window of past data and applies a linear model to predict the next value": $\hat x_t = \sum_{i=1}^{\tau} w_i x_{t-i} + b$ [recorded]. The deck lists three limitations:

1. **Linearity.** "They fail to capture complex, non-linear relationships, such as: the interaction between different past events. Sudden shocks or spikes in the data (e.g., a stock market crash)."
2. **Fixed window.** "By conditioning only on a fixed-length window ($\tau$), the model is completely blind to any information that occurred before that window" — fatal "when long-range dependencies are important."
3. **Error accumulation.** For multi-step prediction the model feeds its own outputs back in: "$\hat x_{t+1}$ will have some error, $\epsilon_1$. This incorrect value... is then fed back into the model to predict $\hat x_{t+2}$. The error in the input leads to a new, often larger, error $\epsilon_2$... causing the predictions to diverge rapidly from the true data."

**iii) MLPs and CNNs.** Chapter 44's machinery doesn't transfer. The deck's three counts: **Fixed-size vectors** — "They require inputs of a fixed length. This forces us to either truncate long sequences or pad short ones, both of which can lead to loss of information." **No parameter sharing across time** — "An MLP learns separate weights for features at each position. It cannot generalize a pattern it learns at position 'i' to position 'j'. A CNN shares parameters spatially, but isn't designed for capturing temporal dependencies of arbitrary length." **No memory of context** — "They process each part of the input independently. They have no mechanism to remember previous elements when processing a new one." The verdict: "We need a model that processes sequences step-by-step, with memory" [recorded].

**Basically, ...** "Three workarounds, three bills. Markov: only look back $n{-}1$ steps — but a longer look-back multiplies the parameter table by the vocabulary size each time. Linear window: same blind spot, plus straight-line-only predictions that drift when fed their own errors. MLPs/CNNs: fixed-length inputs, and an MLP learns position 5's pattern separately from position 50's — no sharing across time, no memory. The spec for the fix: step-by-step, with memory."

## 45.3 The RNN: one cell, reused across time

**The latent hidden state.** Instead of conditioning on the raw past, "an RNN maintains a compact summary of the entire history in a hidden state, $h_t$":
$$\boxed{P(x_t \mid x_1, \dots, x_{t-1}) \approx P(x_t \mid h_{t-1})}, \qquad h_t = f(x_t, h_{t-1}) \quad \text{[recorded]}.$$
"An RNN processes a sequence by iterating through its elements one by one. It maintains a hidden state (or 'memory') that captures information about what it has seen so far. The output at a given time step depends not only on the current input but also on the hidden state from the previous time step." At each step the network "takes a new input $x_t$; updates its hidden state $h_t$; produces an output $y_t$" [recorded].

**The equations (the deck's).** A single recurrent neuron:
$$\boxed{h_t = \tanh(w_{xh}\, x_t + w_{hh}\, h_{t-1} + b_h)}, \qquad \boxed{y_t = w_{hy}\, h_t + b_y} \quad \text{[recorded]}.$$
$w_{xh}$ weights the current input, $w_{hh}$ "the recurrent or 'memory' weight" scales the previous hidden state, $w_{hy}$ reads the output out. A layer vectorizes it (§41.4's move):
$$\boxed{h_t = \tanh(W_{xh}\, x_t + W_{hh}\, h_{t-1} + b_h)}, \qquad \boxed{y_t = W_{hy}\, h_t + b_y} \quad \text{[recorded]}.$$
This is §41.4's two equations with one addition: the $W_{hh} h_{t-1}$ term, the memory of the previous step, added to the pre-activation before the nonlinearity. The deck uses $\tanh$ — §41.7's zero-centered S-bend, "often speeding up convergence" for exactly this inner recurrence.

**The key point (the deck's, §44.10's twin).** "Key Point: The same weight matrix is used at every time step." Drawn folded, the RNN is a neuron with a self-loop; drawn **unfolded over time**, it is a chain of identical cells, $W_{xh}, W_{hh}, W_{hy}$ repeated at $t-1, t, t+1$. Chapter 44's filter is reused at every *pixel*; the RNN's cell is reused at every *timestep* — one set of parameters no matter how long the sequence, which is also what lets it accept variable-length inputs that an MLP cannot.

<!-- Diagram reference (CC-BY-SA): d2l.ai Fig. 9.1 shows the folded cyclic RNN beside its time-unfolded chain with shared parameters across steps — the canonical figure for this section. Source URL: https://d2l.ai/chapter_recurrent-neural-networks/index.html -->

**Note (the sentiment notebook's variant).** Its §1.4 writes the same update with a generic nonlinearity, $h_t = \sigma(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$, $\sigma$ = "a non-linear activation function" [recorded] — the deck's neuron fixes $\sigma = \tanh$. Same cell, activation swapped.

**Basically, ...** "One small network, run in a loop. Each step: mix the new input with a summary of everything seen so far (the hidden state), squash with tanh, read out the prediction. The loop is the memory — and the weights are shared across every step, so a 10-word sentence and a 1000-word sentence use the same parameters. That is Chapter 44's sharing idea, rotated from space to time."

## 45.4 eg: one forward step — next-character prediction [verified-NumPy]

The deck's worked example. Vocabulary $\{\text{h}{:}0, \text{e}{:}1, \text{l}{:}2, \text{o}{:}3\}$ (the §45.2 tokenization story in miniature: text → integer indices → one-hot). Task: input character "e" ($x_t = (0,1,0,0)^T$), memory of having just seen "h" ($h_{t-1} = (0.6, 0.2)^T$, "pre-computed for simplicity"), predict the next character. Ground truth: "l" ($(0,0,1,0)^T$).

Weights:
$$W_{xh} = \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\\\ 0.4 & 0.5 & 0.6 & 0.7 \end{bmatrix},\quad W_{hh} = \begin{bmatrix} 0.1 & 0.5 \\\\ 0.5 & 0.1 \end{bmatrix},\quad W_{hy} = \begin{bmatrix} 2 & -1 \\\\ 1 & 0 \\\\ 1 & 3 \\\\ 0 & -2 \end{bmatrix},\quad b_h = \begin{bmatrix} 0.1 \\\\ 0.1 \end{bmatrix},\ b_y = \begin{bmatrix} 0.1 \\\\ 0.2 \\\\ 0.3 \\\\ 0.4 \end{bmatrix}.$$

Step 1 — the new hidden state. Input contribution: $W_{xh} x_t$ picks out column 1 (the one-hot selects the "e" column): $(0.2, 0.5)^T$. Memory contribution: $W_{hh} h_{t-1} = (0.1\cdot0.6 + 0.5\cdot0.2,\ 0.5\cdot0.6 + 0.1\cdot0.2)^T = (0.16, 0.32)^T$. Add the bias:
$$h_t = \tanh\!\left(\begin{bmatrix} 0.2 \\\\ 0.5 \end{bmatrix} + \begin{bmatrix} 0.16 \\\\ 0.32 \end{bmatrix} + \begin{bmatrix} 0.1 \\\\ 0.1 \end{bmatrix}\right) = \tanh\!\begin{bmatrix} 0.46 \\\\ 0.92 \end{bmatrix} = \begin{bmatrix} 0.4301 \\\\ 0.7259 \end{bmatrix}.$$

Step 2 — output logits. $W_{hy} h_t = (2(0.4301) - 0.7259,\ 0.4301,\ 0.4301 + 3(0.7259),\ -2(0.7259))^T = (0.1343,\ 0.4301,\ 2.6078,\ -1.4518)^T$; add $b_y$:
$$y_t = \begin{bmatrix} 0.2343 \\\\ 0.6301 \\\\ 2.9078 \\\\ -1.0518 \end{bmatrix} \begin{array}{l} \leftarrow \text{'h'} \\\\ \leftarrow \text{'e'} \\\\ \leftarrow \text{'l'} \\\\ \leftarrow \text{'o'} \end{array}.$$
Highest logit $2.9078$ → 'l'. "The model predicts 'l'!" [recorded] — matching the ground truth.

**Note (deck arithmetic slip, corrected here).** The deck's slide writes the fourth pre-bias entry as $-1.01$ and the final logit as $-0.61$; the correct values are $-1.4518$ and $-1.0518$ ($0\cdot0.4301 - 2\cdot0.7259 = -1.4518$, then $+0.4$). The argmax and the conclusion are unaffected. The deck's rounded $h_t = (0.43, 0.72)$ accounts for the small differences in the first three entries ($0.24, 0.63, 2.89$ vs $0.234, 0.630, 2.908$ here).

**Basically, ...** "One-hot 'e' picks out one column of $W_{xh}$ (the 'what does e mean' column); the old hidden state gets mixed through $W_{hh}$ (the 'what did I remember' path); add, tanh, done — new memory $(0.43, 0.73)$. Then $W_{hy}$ scores each vocabulary character against that memory, and 'l' wins with $2.91$. The one-hot trick makes the input matrix multiply into a column lookup — cheap and readable."

## 45.5 BPTT: the same backprop, unrolled

Unfold the RNN over $T$ steps and it is a $T$-layer feedforward network whose layers happen to share weights. So training is Chapter 42's machinery with a new name: "BPTT is simply the application of the standard backpropagation algorithm to the 'unrolled' computational graph of an RNN. It is not a new algorithm, but a name for the process" [recorded]. The loss at the end "depends on computations from every single time step" — $h_t$ depends on $h_{t-1}$, which depends on $h_{t-2}$, "and so on" — so the error must flow back through the whole chain.

**The heart of BPTT (the deck's recursion).** The gradient of the total loss $L = \sum_{t=1}^{T} L_t$ w.r.t. a hidden state has *two* components:
$$\boxed{\frac{\partial L}{\partial h_t} = \underbrace{\frac{\partial L}{\partial y_t}\frac{\partial y_t}{\partial h_t}}_{\text{current loss}} + \underbrace{\frac{\partial L}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial h_t}}_{\text{future loss}}} \quad \text{[recorded]}.$$
"This recursive formula is the heart of BPTT": the error at $h_t$ is the error from *this* step's output plus the error arriving back from *the future* through the recurrent connection. Start at $t = T$ (no future term) and walk backwards.

**The parameter gradients (the deck's).** With $\partial L/\partial h_t$ in hand, "we can find the gradients for the weights and biases at that time step. The final gradient for a parameter is the sum of its gradients from all time steps" [recorded]:
$$\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^{T} \frac{\partial L}{\partial y_t} h_t^T, \qquad \frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \odot \text{(local terms)}\, x_t^T, \qquad \frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \odot \text{(local terms)}\, h_{t-1}^T.$$
The local terms are the $\tanh$ derivative $\phi'(z) = 1 - \tanh^2(z)$ at each step — §42.4's three-component product (upstream × activation × local), summed over time because the weights are shared across it. Sharing buys parameter efficiency (§45.3); the sum is its bookkeeping cost.

**Note.** The two-term recursion is §42.4's $\delta$ recursion seen from the side: in a feedforward net the error flows *down through layers*; here it flows *back through time* — the same chain rule, and §42.6's gradient-checking advice applies to it unchanged.

**Basically, ...** "Unroll the loop and it's just a deep net with tied weights — backprop as usual, hence 'backprop through time'. Each hidden state's blame has two sources: the output it produced *now*, and the future states it helped build *later*. Add the two, step back one timestep, repeat. Parameter gradients are the per-step gradients *summed* — one weight, many timesteps, so every step gets a vote."

## 45.6 eg: BPTT by hand on a 1-neuron net [verified-NumPy]

The deck's setup, with its arithmetic corrected (see the Note). One neuron, sequence length $2$, squared-error loss $L_t = (y_t - \text{target}_t)^2$, biases $0$:
$$x_1 = 2,\ x_2 = 3,\ h_0 = 0,\quad w_{xh} = 0.5,\ w_{hh} = 0.2,\ w_{hy} = 0.4,\quad \text{targets } 0.5,\ 0.8,\ \phi = \tanh.$$

**Forward.** $t{=}1$: $h_1 = \tanh(0.5\cdot2 + 0.2\cdot0) = \tanh(1) = 0.7616$; $y_1 = 0.4\cdot0.7616 = 0.3046$; $L_1 = (0.3046-0.5)^2 = 0.0382$. $t{=}2$: $h_2 = \tanh(0.5\cdot3 + 0.2\cdot0.7616) = \tanh(1.6523) = 0.9292$; $y_2 = 0.4\cdot0.9292 = 0.3717$; $L_2 = (0.3717-0.8)^2 = 0.1835$. Total $L = 0.2216$.

**Backward, from the end.**
$t{=}2$: $\frac{\partial L}{\partial y_2} = 2(0.3717 - 0.8) = -0.8567$; $\frac{\partial L}{\partial h_2} = -0.8567 \cdot 0.4 = -0.3427$. The $w_{hh}$ gradient at this step (local: $\phi'(z_2)\,h_1 = (1-h_2^2)\,h_1$):
$$\left.\frac{\partial L}{\partial w_{hh}}\right|_{t=2} = (-0.3427)\,(1-0.9292^2)\,(0.7616) = -0.0357.$$
$t{=}1$ — the crucial step, both components:
$$\frac{\partial L}{\partial h_1} = \underbrace{2(0.3046-0.5)\cdot0.4}_{\text{current } = -0.1563} + \underbrace{(-0.3427)\,(1-0.9292^2)\,(0.2)}_{\text{future } = -0.0094} = -0.1657.$$
Then $\left.\frac{\partial L}{\partial w_{hh}}\right|_{t=1} = (-0.1657)\,(1-0.7616^2)\,(0) = 0$ — $h_0 = 0$ kills it, the same silent-input logic as §42.3's Note ($x_2 = 0$ there). Final:
$$\boxed{\frac{\partial L}{\partial w_{hh}} = 0 + (-0.0357) = -0.0357}.$$
The other two, same recipe: $\partial L/\partial w_{hy} = \sum_t \frac{\partial L}{\partial y_t} h_t = (-0.3907)(0.7616) + (-0.8567)(0.9292) = -1.0936$, and $\partial L/\partial w_{xh} = \sum_t \frac{\partial L}{\partial h_t}(1-h_t^2)x_t = (-0.1657)(0.4200)(2) + (-0.3427)(0.1366)(3) = -0.2796$ [verified-NumPy].

**Note (deck arithmetic slip, corrected here).** The deck writes $y_2 = 0.4\cdot0.93 = 0.744$ — but $0.4\cdot0.93 = 0.372$; its $0.744$ equals $0.8\cdot0.93$, as if the target $0.8$ crept in as the weight. Everything downstream in the deck ($L_2 = 0.003$, $\partial L/\partial y_2 = -0.112$, $\partial L/\partial w_{hh} = -0.0046$) follows consistently from the slipped value, so only the one line needed fixing; the corrected pass above re-derives it all. The structural points survive: the two-component recursion, and the $h_0 = 0$ zero gradient.

**Basically, ...** "Forward: two tanh squashes, two squared errors, $L = 0.222$. Backward: at the last step the blame is just 'prediction minus target, doubled' times the output weight; at the first step it's that *plus* the future's blame passed back through $w_{hh}$ and the tanh slope — current loss plus future loss, exactly §45.5's recursion. The $t{=}1$ memory gradient is zero because $h_0 = 0$: nothing flowed through, nothing to blame."

## 45.7 The price of sharing: vanishing and exploding gradients

**The product.** To reach the early weights, the error crosses every timestep. The deck writes the chain for the hidden-state Jacobian over $T$ steps:
$$\boxed{\frac{\partial h_T}{\partial h_0} = \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=1}^{T} W_{hh}^T\,\mathrm{diag}(\phi'(\dots))} \quad \text{[recorded]}.$$
Each step multiplies by the *same* $W_{hh}$ and by the $\tanh$ slope $\phi'(z) = 1 - \tanh^2(z) \le 1$ — "This long product is the source of major numerical instability" [recorded]. This is §42.8(ii)'s other half: there the error crossed *layers* and each sigmoid multiplied it by at most $0.25$; here it crosses *time* and each step multiplies by $W_{hh}^T\,\mathrm{diag}(\phi')$.

**The twin problems (the deck's table).**

i) **Gradient explosion.** "If the recurrent weight matrix $W_{hh}$ has large values, the long product of matrices can grow exponentially." Result: "Gradients become enormous (NaN or infinity)" → "Model weights are updated by huge amounts" [recorded]. Note: the deck's Week-5 companion (§43.12's troubleshooting table) lists "Loss exploding → Decrease LR, gradient clipping" — the same fix, now with its derivation.

ii) **Gradient vanishing.** "If the recurrent weight matrix has small values, the long product of matrices can shrink exponentially towards zero." Result: "Gradients for early time steps become nearly zero" → "The model cannot learn long-range dependencies" → "The model forgets what happened early in the sequence" [recorded].

The $\tanh$ makes (ii) worse even at moderate weights: $\phi'(z) \le 1$ always, and a saturated neuron ($\tanh(z) \approx \pm 1$) has $\phi'(z) \approx 0$ — §41.7's tanh row ("same vanishing-gradient flattening — 'difficult to use in very deep networks'") is the same sentence, now about depth *in time*. A neuron that saturates stops its own past from being learnable.

**eg (the shrink and the blow-up, counted — [verified-NumPy]).** Suppose each step multiplies the error by roughly $0.64$ (e.g. $\|W_{hh}\| \approx 0.8$, $\phi' \approx 0.8$): after 20 steps the surviving fraction is $0.64^{20} \approx 1.3\times10^{-4}$ — four orders of magnitude gone. With $1.5$ per step instead: $1.5^{20} \approx 3325$ — the gradient more than triples per step and compounds to thousands. The boundary between the two regimes is a knife's edge, and the same $W_{hh}$ must serve every timestep — that is why the deck calls the product itself, not any one bad weight, "the source."

**The three fixes (the deck's).**

i) **Gradient clipping** (for explosion). "If the norm of the gradient exceeds a certain threshold, it is scaled down" [recorded]. No formula in the deck; the torch call lives in §43.12 (`torch.nn.utils.clip_grad_norm_`), which lists clipping as "essential for RNNs/Transformers" — the derivation above is why.

ii) **Truncated BPTT** (for cost and instability). "Instead of backpropagating through the entire sequence, we detach the gradient history after a fixed number of steps ($\tau$). The model still propagates its hidden state forward through the whole sequence, maintaining its long-term memory. However, during the backward pass, the gradient calculation is cut off after $\tau$ steps" [recorded]. "Trade-off: It limits the model's ability to learn dependencies that span longer than $\tau$ time steps."

iii) **More complex RNN architectures** (for vanishing). "Solution: More complex RNN architectures" — the gates of §45.8, which keep a gradient path that does not multiply by $W_{hh}$ every step.

**Basically, ...** "Backprop through 20 timesteps multiplies the error by the same weights 20 times. Slightly-small weights: the signal dies ($0.64^{20}$ is basically zero) and the net forgets the past. Slightly-big: it explodes ($1.5^{20}$ is 3325) and you get NaNs. Clipping caps the explosions, truncation stops the backward chain early (but blinds the net beyond $\tau$ steps), and the real fix for vanishing is architectural — gates."

## 45.8 LSTM: gates and the additive cell

**What it is.** "LSTM networks are a type of recurrent neural network (RNN) designed to solve the vanishing gradient problem. They are particularly effective at learning long-term dependencies in sequential data" [recorded]. Three differences from the simple RNN: **dual states** — "a short-term state $h_t$ and a long-term state (or cell state) $c_t$"; a **gated mechanism** — "special 'gates' to regulate the flow of information into and out of the cell state"; and **gradient stability** — "The gates and additive interactions within the cell state help combat both vanishing and exploding gradients, making training more stable" [recorded].

**The gates (the deck's equations, [recorded]).** Each gate is a sigmoid over the concatenated $[h_{t-1}, x_t]$ — a number in $(0,1)$ per cell dimension, i.e. a soft valve:

- **Forget gate** — "Decides what information from the old cell state to throw away": $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$. "$1$ means 'keep this information completely' and $0$ means 'forget it completely.'"
- **Input gate** — "Decides what new information from the current input to store in the cell state": $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$, paired with a **candidate** $\tilde c_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$ — "$\tanh$... to create a new vector of potential information, scaled between -1 and 1." The two "are multiplied element-wise to form the new information that will be added to the cell state."
- **Cell update** — "This is the core of the LSTM's memory mechanism":
$$\boxed{c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t} \quad \text{[recorded]}.$$
"The first term, $f_t \odot c_{t-1}$, represents the old memory with some parts forgotten. The second term, $i_t \odot \tilde c_t$, represents the new information being added."
- **Output gate** — "Decides what part of the cell state to output as the new hidden state": $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$, $h_t = o_t \odot \tanh(c_t)$. "The hidden state $h_t$ is a filtered version of the cell state."

**Why the gates fix vanishing.** "This additive interaction is crucial for preserving gradients. Instead of a series of multiplicative steps, the addition allows gradients to flow directly back, preventing them from vanishing" [recorded]. In symbols: $\partial c_t / \partial c_{t-1} = f_t$ — the error on the cell state passes back through a *single* gate value, not through $W_{hh}^T\,\mathrm{diag}(\phi')$. If the forget gate stays near $1$ (keep everything), the gradient crosses timesteps almost intact — the §45.7 product becomes a sum of $f_t$'s instead of a product of $W_{hh}$'s.

**eg (the gradient path, counted — [verified-NumPy]).** Over 20 timesteps with a vanilla step-factor of $0.64$, §45.7's eg left $1.3\times10^{-4}$. With an LSTM whose forget gates sit at $f_t = 0.95$ (mostly keeping), the cell-state gradient factor is $0.95^{20} \approx 0.36$ — more than a third survives. The gates *learn* where to sit: $f_t \approx 1$ on stretches worth remembering, $f_t \approx 0$ where the past should be dropped. The network learns its own truncation window.

**Basically, ...** "An LSTM keeps two memories: a long-term cell state and a short-term hidden state. Three sigmoid valves guard the cell — forget (what to erase), input (what new stuff to write, via a tanh candidate), output (what to show the world). The cell update is *addition*, not a matrix multiply, so the gradient flows back through a chain of forget-gate values instead of a chain of weight matrices — and a forget gate near 1 means the past survives. Basically: a memory with an eraser, a pen, and a lid, all learned."

## 45.9 GRU: two gates, no separate cell

**What it is.** "The Gated Recurrent Unit (GRU), introduced in 2014, is a popular alternative to the LSTM. It aims to solve the same vanishing gradient problem but with a simpler architecture and fewer parameters" [recorded]. "Two Gates instead of Three" and "No Separate Cell State: GRU merges the cell state and hidden state into a single hidden state vector, $h_t$." The payoff: "Due to its simpler design, a GRU often trains faster than an LSTM and can perform just as well, or sometimes even better, on less complex datasets" [recorded].

**The equations (the deck's, [recorded]).** Step 1, the two gates:

- **Reset gate** — "determines how the previous hidden state $H_{t-1}$ is used to compute the new candidate hidden state": $R_t = \sigma(X_t W_{xr} + H_{t-1} W_{hr} + b_r)$. "When entries in $R_t$ are close to 1, the gate allows the previous hidden state to pass through... close to 0, the gate effectively 'resets' or ignores parts of the previous hidden state."
- **Update gate** — "controls how much of the previous hidden state $H_{t-1}$ is directly carried over to the final hidden state $H_t$": $Z_t = \sigma(X_t W_{xz} + H_{t-1} W_{hz} + b_z)$. "It acts similarly to the forget gate in an LSTM. When entries in $Z_t$ are close to 1, the corresponding dimension in $H_{t-1}$ is almost entirely copied to $H_t$, preserving long-term memory."

Step 2, the states. **Candidate** (a vanilla RNN step, but with the reset gate throttling the past): $\tilde H_t = \tanh(X_t W_{xh} + (R_t \odot H_{t-1}) W_{hh} + b_h)$. **Final state** — "a linear interpolation between the previous hidden state $H_{t-1}$ and the new candidate hidden state $\tilde H_t$":
$$\boxed{H_t = Z_t \odot H_{t-1} + (1 - Z_t) \odot \tilde H_t} \quad \text{[recorded]}.$$
"$Z_t \approx 1$: retain old information; $Z_t \approx 0$: fully replace it with new information." The same additive trick as the LSTM's cell update (§45.8): $\partial H_t / \partial H_{t-1}$ carries a $Z_t$ term, so gradients survive when the update gate stays open.

**Note (deck notation switch).** The GRU slides write row-vector products ($X_t W_{xr}$) while the LSTM slides write $W_f \cdot [h_{t-1}, x_t]$ — the GRU's $H_t$ is the LSTM section's $h_t$, transposed layout, same role. The LSTM-vs-GRU comparison that matters is architectural: three gates + separate cell (LSTM) vs two gates + merged state (GRU).

**LSTM vs GRU, one line each.** LSTM: more expressive (separate long/short states, three valves) at the cost of more parameters. GRU: fewer parameters, faster training, "just as well, or sometimes even better, on less complex datasets" — and §43.12 files both under "RNNs / LSTMs → RMSProp / Adam." When in doubt, the course's notebooks reach for the LSTM (*sentiment-analysis* builds `SentimentLSTM`; the time-series notebook's class is a plain `nn.RNN`) — the GRU appears in the deck's theory, not in either notebook's code.

**Basically, ...** "GRU = LSTM on a diet. Two valves instead of three (reset: how much past goes into the *candidate*; update: how much past survives into the *final* state), one state instead of two. The final state is a learned mix — update gate near 1 keeps the old memory, near 0 takes the new candidate. Same vanishing-gradient fix (the mix is additive), fewer knobs, faster training."

## 45.10 Reading both ways, speaking at length

**Bidirectional RNNs.** "A standard RNN is unidirectional. The hidden state at time $t$ ($H_t$) only captures information from past inputs" — "a major drawback for tasks where future context is crucial," e.g. "The man who ___ movies is a critic," where "we need the word 'movies' to predict 'reviews'" [recorded]. The fix: "two independent RNNs: one that processes the sequence from left-to-right (a forward layer) and another that processes it from right-to-left (a backward layer). For any time step $t$, the final hidden state is formed by concatenating the hidden states from both the forward and backward layers":
$$\boxed{H_t = [\overrightarrow{h}_t,\ \overleftarrow{h}_t]}, \qquad O_t = H_t W_{hq} + b_q \quad \text{[recorded]}.$$
The deck's walkthrough ("I am ___ happy"): the forward pass reaches the blank with only "I am"; the backward pass with only "happy"; the concatenated $H_3$ "contains a rich representation of the full sentence context" [recorded]. Cost: two passes, and the backward pass must wait for the whole sequence — fine for scoring a sentence, impossible for live generation.

**Encoder–decoder (seq2seq).** "Standard RNNs are only designed for tasks where the input and output lengths are the same. How do we handle tasks like machine translation or summarization, where the lengths can be completely different?" [recorded]. Two RNNs with different jobs: the **encoder** "reads the entire input sequence step-by-step" and "compress[es] the information of the whole sequence into a single fixed-size vector, often called the **context vector**"; the **decoder** is "initialized with the encoder's context vector" and "generates the output sequence one token at a time, using the context and the previously generated token" [recorded]. Trained jointly with cross-entropy "at each timestep of the decoder's output," comparing "the predicted probability distribution over the vocabulary with the one-hot encoded ground-truth token"; with padding, "the loss function uses a mask to ignore these padding tokens during calculation, ensuring they do not contribute to the gradient updates" [recorded].

**Teacher forcing.** "During training, instead of feeding the decoder's own prediction from the previous step as input for the current step, we use the actual ground-truth token from the target sequence." Benefit: "stabilizes training and helps the model learn the sequence structure more efficiently, as it always receives correct input." But "at inference time... the model must use its own previously generated tokens as input, since the ground truth is not available" [recorded] — a train/test mismatch the chapter states and leaves open (Chapter 46's attention does not remove it either, but it does remove the context-vector bottleneck below).

**The context-vector bottleneck.** One fixed-size vector must hold a 500-word article for a 50-word summary ("Text Summarization... Machine Translation... Question Answering" are the deck's applications [recorded]). Everything the decoder knows about the input passes through that single vector — compress a long sequence into it and something is lost. This bottleneck is exactly what §44.10(ii)'s attention shortcut is for: "attention lets any position talk to any other directly, instead of waiting for pooling to bring them together" — here, instead of squeezing through one vector.

**Basically, ...** "Forward-only RNNs can't see the future — so run a second one backwards and concatenate: now every position knows both sides. Different input/output lengths (translation, summaries) need two RNNs: an encoder squeezes the input into one context vector, a decoder unrolls the output from it. Training cheats by feeding the decoder the true previous word (teacher forcing); at test time it's on its own. The weak link: one vector carrying a whole paragraph — remember this when attention arrives."

## 45.11 Judging sequences: perplexity, BLEU, beam search

**Perplexity.** "Perplexity (PPL) is the standard metric for evaluating language models. It measures how 'surprised' or 'perplexed' a model is by a sequence of text" — "Lower Perplexity is Better!":
$$\boxed{\mathrm{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_1, \dots, x_{t-1})\right)} \quad \text{[recorded]}.$$
It is "the exponentiation of the average negative log-likelihood" — the negative log-likelihood of §35.6's cross-entropy vocabulary, exponentiated. Intuition: "A perplexity of $k$ means that, on average, the model is as uncertain about the next word as if it had to choose uniformly from $k$ different words" [recorded]. Sanity check [verified-NumPy]: a model assigning probability $0.5$ to each of four true next-words has average NLL $-\ln 0.5 = 0.693$, so $\mathrm{PPL} = e^{0.693} = 2.0$ — as confused as a fair coin.

**BLEU.** "BLEU (Bilingual Evaluation Understudy) is a standard metric for evaluating the quality of a predicted sequence by comparing it to a target (ground truth) sequence," based on "the precision of matching n-grams... between the two sequences" [recorded]:
$$\boxed{\mathrm{BLEU} = \exp\!\left(\min\!\left(0, 1 - \frac{\mathrm{len}_{\text{label}}}{\mathrm{len}_{\text{pred}}}\right)\right) \cdot \left(\prod_{n=1}^{k} p_n\right)^{1/k}} \quad \text{[recorded]},$$
$p_n$ = matched n-grams / predicted n-grams. The $\min(0, 1 - \cdot)$ term is the **brevity penalty**: "If the predicted sequence is identical to the target, the BLEU score is 1."

**eg (the deck's BLEU example — [verified-NumPy]).** Target "A B C D E F", predicted "A B", $k = 2$: unigrams $\{A, B\}$ both match → $p_1 = 2/2 = 1.0$; bigram $\{AB\}$ matches → $p_2 = 1/1 = 1.0$. "Without a penalty, the score would be $(1.0\times1.0)^{1/2} = 1.0$, which is misleadingly perfect." Brevity penalty: $\exp(1 - 6/2) = \exp(-2) \approx 0.1353$. Final: $\mathrm{BLEU} = 0.1353 \times 1.0 = 0.1353$ — "correctly lowers the score from a perfect 1.0 to 0.135, reflecting that while the precision is high, the prediction is far too short" [recorded].

**Beam search.** Generating means finding "the sequence with the highest conditional probability." **Greedy search** — "at each step, we choose the single most likely token" — is "fast and computationally cheap" but "not optimal. The sequence of most-likely tokens is not necessarily the most-likely sequence overall." **Exhaustive search** "guarantees finding the optimal sequence" but is "computationally infeasible." **Beam search** is "a practical compromise... controlled by a hyperparameter called the beam size, $k$": at each step keep the top-$k$ tokens, expand each into $|V|$ continuations, keep the best $k$ of the $k \times |V|$ candidates, repeat "until an end-of-sequence token is generated or a maximum length is reached" [recorded]. Final scoring normalizes by length to avoid short-sequence bias: $\mathrm{Score} = \frac{1}{L^\alpha}\sum_{t'=1}^{L} \log P(y_{t'} \mid y_{<t'}, c)$, $\alpha \approx 0.75$ [recorded]. "Greedy search is a special case of beam search arising when the beam size is set to 1" [recorded]. (The deck's beam-search illustration is credited to d2l.ai.)

**Basically, ...** "Perplexity: how surprised is the model? Lower is better; 10 means 'as confused as 10 choices'. BLEU: for generated sequences, count matching n-grams (precision) times a brevity penalty — a two-word prediction can't score 1.0 against a six-word target. Beam search: greedy picks the best word each step (fast, wrong); exhaustive tries everything (right, impossible); beam keeps the $k$ best partial sentences and prunes — greedy is just beam with $k = 1$."

## 45.12 The PyTorch face

Both notebooks' torch code is transcribed as [recorded] (torch is not installed here; nothing re-run).

**The time-series notebook's RNN** (`Time_Series_Analysis_using_RNN.ipynb`, AMZN daily closes, 95/5 split, lookback window reshaped to `(samples, timesteps, features)`):
```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])   # last timestep's hidden state -> prediction
        return out
```
[recorded]. Training: `loss_function = nn.MSELoss()`, `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`, 50 epochs [recorded] — §43.12's defaults in the wild. The scaling discipline of §43.9 rides along: `MinMaxScaler(feature_range=(-1, 1))` before training, predictions inverse-transformed after [recorded].

**The sentiment notebook's pair** (IMDB, binary classification). `SentimentRNN`: `nn.Embedding(vocab_size, embedding_dim)` → `nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)` → dropout → `nn.Linear(hidden_dim, output_size)` → sigmoid [recorded]. `SentimentLSTM`: identical scaffolding with `nn.LSTM(...)` in place of `nn.RNN`, and `forward` takes `hidden` as a tuple `(h, c)` — the dual states of §45.8 surfacing in the API [recorded]. Dropout $0.5$ is §41.14(ii)'s guardrail, unchanged.

**§43.12's RNN rows, now grounded.** "RNNs / LSTMs → RMSProp / Adam — Handles non-stationary objectives" and "Clip gradients for RNNs/transformers (`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`)" [recorded] — the optimizer choice and the §45.7(i) fix, as the course's own cheat sheet.

**Basically, ...** "In torch, an RNN is `nn.RNN(input, hidden, layers, batch_first=True)` plus a linear head on the last timestep; an LSTM is a drop-in swap whose hidden state comes as a `(h, c)` pair. Shape your data as (batch, time, features) — §36.4's axis semantics — scale it (§43.9), train with Adam at the usual $0.001$, and clip the gradients (§45.7). The deck's math and the notebooks' code are the same object."

## 45.13 Where this goes next

i) **Attention (Chapter 46).** §45.10's context-vector bottleneck is the problem attention was built for: §44.10(ii)'s "shortcut" — any decoder position talking to any encoder position directly, no single-vector squeeze. The deck's W9–10 material takes it from here.

ii) **The sharing moral, complete.** Chapter 44: share across space → translation robustness, parameter counts collapse (§44.7). Chapter 45: share across time → variable-length sequences, one cell — and the bill arrives as §45.7's product-of-$T$-matrices, §42.8's vanishing story told along a new axis. Clipping (§45.7(i), §43.12's torch call) and gates (§§45.8–45.9) are the two accepted payments.

iii) **Generative models (Chapter 47).** The decoder of §45.10 *is* a generator: "assess how plausible a sequence is, or... generate new, similar sequences" (§45.1). Beam search (§45.11) is the decoding algorithm the generative chapters inherit.

iv) **The debugging playbook rides along.** §40.5(i): "shapes first" — for an RNN that means the `(batch, timesteps, features)` reshape (§45.12) before anything trains, then scale (§43.9's sermon — the time-series notebook scales to $(-1,1)$), then the §45.7 watch: NaNs mean the exploding regime, a flat loss on long sequences means the vanishing one.

## Problem set

1. **Factor and cut.** For $x_1, x_2, x_3, x_4$: (i) write the full autoregressive factorization. (ii) Write the first-order Markov approximation. (iii) A bigram model over $|V| = 10{,}000$ words needs how many conditional-probability entries? A trigram? In one line, what does the ratio say about "the number of model parameters grows exponentially"?
2. **One recurrent step, by hand.** 1-neuron RNN, $\phi = \tanh$, $w_{xh} = 0.8$, $w_{hh} = -0.3$, $b_h = 0.1$, $x_1 = 1.0$, $h_0 = 0$. (i) Compute $h_1$ to 4 s.f. (ii) Now $x_2 = -1.0$: compute $h_2$ to 4 s.f. (iii) In one line: which term in the $h_2$ computation is "the memory," and what would $h_2$ be if $w_{hh} = 0$?
3. **The sharing payoff, counted.** An RNN layer with input dimension $d = 50$ and hidden size $h = 100$ (include biases). (i) Count its parameters ($W_{xh}, W_{hh}, b_h$). (ii) An MLP applied separately at each of $T = 30$ timesteps with its own weights would need how many? (iii) In one line, connect this to §44.7's $8{,}400\times$ and say what the RNN count does as $T$ grows.
4. **The two-component recursion.** (i) In words: why does $\partial L/\partial h_t$ have a "current loss" term and a "future loss" term — which part of the unrolled graph does each come from? (ii) At $t = T$, which term vanishes and why? (iii) §42.4's $\delta^{[l]}$ recursion pushes error through *layers*; BPTT pushes it through *time*: in one line each, name the matrix that carries it in each case.
5. **Finish the BPTT eg.** Using §45.6's forward values: (i) derive $\partial L/\partial w_{hy} = \sum_t \frac{\partial L}{\partial y_t} h_t$ from $y_t = w_{hy} h_t$ and evaluate it. (ii) Derive $\partial L/\partial w_{xh} = \sum_t \frac{\partial L}{\partial h_t}(1-h_t^2)x_t$ and evaluate it. (iii) In one line: why does the $t{=}1$ term of $\partial L/\partial w_{hh}$ vanish while the $t{=}1$ term of $\partial L/\partial w_{xh}$ does not?
6. **Name the regime.** Each backward step multiplies the error by $\approx 0.64$. (i) After 20 steps, what fraction survives? (ii) Same question for a per-step factor of $1.5$. (iii) In two lines, map (i)/(ii) to the deck's vanishing/exploding rows, and connect (i) to §42.8(ii)'s sigmoid $\le 0.25$ story.
7. **Clip vs truncate.** (i) In one line each: what does gradient clipping fix, and what does it *not* fix? (ii) Truncated BPTT with $\tau = 5$: in one line, what dependency length can the model no longer learn? (iii) In two lines: why does the deck still call truncated BPTT a "compromise" even though the hidden state keeps flowing forward?
8. **The LSTM's gradient path.** $f_t = (0.9, 0.2)$, $i_t = (0.5, 0.8)$, $\tilde c_t = (0.6, -0.4)$, $c_{t-1} = (1.0, 2.0)$. (i) Compute $c_t$. (ii) What is $\partial c_t/\partial c_{t-1}$ (element-wise)? (iii) In two lines: explain why the first dimension's gradient survives far better than the second's over many steps, using your answer to (ii).
9. **The GRU's mix.** $Z_t = (0.8, 0.1)$, $H_{t-1} = (1.0, -1.0)$, $\tilde H_t = (0.0, 2.0)$. (i) Compute $H_t = Z_t \odot H_{t-1} + (1-Z_t)\odot\tilde H_t$. (ii) In one line each: what does $Z_t \approx 1$ do, and $Z_t \approx 0$? (iii) In one line: which LSTM gate does $Z_t$ "act similarly to" (deck's words), and why?
10. **Generate and judge.** (i) Target "A B C D", predicted "A B": compute BLEU with $k = 2$ (brevity penalty included). (ii) In two lines: why is the brevity penalty needed — what would the score be without it? (iii) Teacher forcing feeds the decoder ground-truth tokens during training but its own predictions at inference: in one line, state the resulting mismatch; in one line, say whether beam search (§45.11) fixes it.

---

*Sources: GenAI Weeks 7–8 "Introduction to RNN" deck = Notes `Week7,8-Introduction to RNN.pdf` (Balaji Srinivasan, Ganapathy Krishnamurthi), 80 slides — sequential-data characteristics, sequence-model target $P(x_1,\dots,x_T)$, autoregressive factorization, Markov/n-gram (fixed memory, exponential parameters), linear window regression ($\hat x_t = \sum w_i x_{t-i} + b$; three limitations incl. error accumulation $\epsilon_1, \epsilon_2$), MLP/CNN limitations ("No Parameter Sharing Across Time", "We need a model that processes sequences step-by-step, with memory"), tokenization/vocabulary (the/un k example), random sampling ($\tau$, shift-by-one targets), perplexity (formula, "$k$ words" intuition), RNN hidden state ($P(x_t|x_{<t}) \approx P(x_t|h_{t-1})$, $h_t = f(x_t,h_{t-1})$), "same weight matrix at every time step", neuron/layer equations, next-character worked example (corrected per §45.4 Note), BPTT ("not a new algorithm", two-component recursion "the heart of BPTT", parameter sums, 1-neuron worked example corrected per §45.6 Note), instability product $\prod W_{hh}^T\mathrm{diag}(\phi')$, twin-problems table (clipping / "more complex RNN architectures"), truncated BPTT ($\tau$, trade-off), LSTM (dual states, four gate equations, additive cell update, gradient-flow interpretation), GRU (2014, two gates, no separate cell, all equations, "trains faster... just as well"), bidirectional RNNs ($H_t = [\overrightarrow{h}_t,\overleftarrow{h}_t]$, "I am ___ happy" walkthrough), encoder–decoder (context vector, teacher forcing, masking, cross-entropy per timestep), BLEU (formula + worked example), beam search (steps, $k$, length normalization, greedy = $k{=}1$; illustration credited to d2l.ai); `Time_Series_Analysis_using_RNN.ipynb` — AMZN pipeline, `prepare_dataframe_for_lstm`, MinMaxScaler$(-1,1)$, `(samples, timesteps, features)` reshape, `RNN` class (`nn.RNN(..., batch_first=True)` + `nn.Linear`), Adam $\eta{=}0.001$, `nn.MSELoss`, 50 epochs (all torch [recorded]); `sentiment-analysis.ipynb` — IMDB pipeline, $h_t = \sigma(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$ framework, `SentimentRNN`/`SentimentLSTM` (embedding → recurrent → dropout $0.5$ → fc → sigmoid; LSTM `hidden` as tuple) (all torch [recorded]); deck figure credits — BPTT illustration (DOI 10.13140/RG.2.2.34411.08483), LSTM architecture (DOI 10.1109/ACCESS.2021.3125733), GRU architecture (DOI 10.1007/s11042-023-15571-y); d2l.ai RNN chapter index — live-checked, unrolled-RNN figure (Fig. 9.1, CC-BY-SA) cited as HTML-comment diagram reference only; book chapters 35 (§35.6), 36 (§36.4), 40 (§40.5(i)), 41 (§§41.4, 41.7, 41.14(ii)), 42 (§§42.3–42.4, 42.6, 42.8), 43 (§§43.9, 43.12), 44 (§44.7, §44.10).*
