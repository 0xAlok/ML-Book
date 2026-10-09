# Chapter 46: Attention and the transformer

Chapter 45 ended with the bottleneck: one fixed-size context vector must hold a whole paragraph (§45.10). Chapter 44 ended with a promise: "attention lets any position talk to any other directly, instead of waiting for pooling to bring them together" (§44.10(ii)). This chapter cashes both. The spine is the GenAI Weeks 9–10 deck ("Introduction to Attention and Transformer Architecture," Balaji Srinivasan and Ganapathy Krishnamurthi, 99 slides — word embeddings, attention pooling, scaled dot-product attention, self vs cross attention, masked softmax, multi-head attention, positional encoding, residuals and layer norm, the full encoder–decoder, ViT, and the modern LLM arc), cross-checked against the notes PDF of the same deck (identical content, 20 pages), plus three companion notebooks: *machine_translation* (Bahdanau-attention seq2seq, German→English on Multi30k), *HuggingFace Inference* (sentiment, NER, translation pipelines), and Week 12's *Transformers* (a nanoGPT built from scratch, trained on a 19-token toy corpus). Every numpy number below was re-run here and is marked **[verified-NumPy]**; deck, notebook, and torch numbers are transcriptions, marked **[recorded]**.

**Notation.** A query, key, value pair is $q, k, v \in \mathbb{R}^d$; matrices $Q \in \mathbb{R}^{n \times d}$ (one row per query), $K \in \mathbb{R}^{m \times d}$, $V \in \mathbb{R}^{m \times v}$ stack them. $\alpha(q, k_i)$ is the attention weight of key $k_i$ for query $q$. The deck writes the scale factor as $\sqrt{d}$; this book writes $\sqrt{d_k}$ — same number ($d_k$ = dimension of the query/key vectors). Multi-head notation keeps the deck's letters: head $i$'s projected matrices are $Q_i, K_i, V_i$ and the final projection is $W^{(o)}$. Position $p$'s encoding is $PE(p)$.

## 46.1 Two promises this chapter cashes

**i) The context-vector squeeze.** §45.10's encoder compresses the entire input into one fixed-size vector; the decoder sees nothing else. For short sentences this works; past "typically 30–40 tokens" performance "degrades dramatically" (deck) [recorded]. Attention deletes the squeeze: at every generation step, the decoder computes a *fresh* context — a weighted mix of *all* encoder states, with the weights chosen per step. The sentence is never compressed at all.

**ii) The shortcut.** §44.10(ii) promised that "attention lets any position talk to any other directly, instead of waiting for pooling to bring them together." Self-attention does exactly that inside one sequence: token 1 and token 50 interact in a single matrix multiply, with no recurrence chain (§45.7's $T$-step product) and no pooling pyramid (§44.6) in between. The deck's table calls these "direct interactions between any two tokens," against the RNN's $O(T)$ sequential depth and the CNN's slow buildup to global context [recorded].

**Basically, ...** "Two old limits, one fix. RNNs route everything through a single memory vector and a long relay chain; CNNs shrink the picture layer by layer to connect far-apart things. Attention lets every position look at every other position in one step — each output token picks what it needs from the whole input, afresh, every time."

## 46.2 Embeddings: turning words into vectors

Before attention can mix words, words must become vectors. **Def (the deck's).** "A word embedding is a way of representing a word as a list of numbers, called a vector." Each vocabulary entry gets one: $\text{King} \to [0.12, -0.45, 0.88, \dots, -0.15]$ [recorded]. The numbers are coordinates in a "meaning space": words with similar meanings land near each other ("cat" nearer "dog" than "car"), and *directions* capture relations — the deck's famous relation:
$$\overrightarrow{\text{King}} - \overrightarrow{\text{Man}} + \overrightarrow{\text{Woman}} \approx \overrightarrow{\text{Queen}} \quad \text{[recorded]}.$$

Two regimes:

**i) Static embeddings.** Word2Vec, GloVe, FastText: one fixed vector per word, learned from co-occurrence statistics over billions of words ("a word is known by the company it keeps"). Weakness: "bank" gets the same vector in "river bank" and "money bank" [recorded].

**ii) Contextual embeddings.** BERT reads the sentence both ways; GPT reads left-to-right. The vector for a word changes with the sentence it sits in. These are what a transformer layer produces — the embedding of "it" in "the animal didn't cross the street because it was too tired" already knows which "it" it is (§46.9).

**How they're learned.** Nobody hand-writes them. Either learn from a big corpus, or, inside a model: **1)** "Initialize Randomly: Assign a vector of random numbers to every word in your vocabulary," **2)** "Learn Jointly: Train a neural network on a specific task... As the model learns, it simultaneously adjusts the word embedding vectors along with its other weights" [recorded].

**Basically, ...** "Embeddings are word coordinates. Similar words sit near each other; arrows between them capture relationships. The old ones give each word one fixed address; transformer ones compute the address fresh from the surrounding sentence."

## 46.3 Attention, in one formula: queries, keys, values

**Def (the deck's).** "Attention can be described as a process of mapping a Query and a set of Key-Value pairs to an output" — "the core of modern self-attention." The deck's cookbook analogy: the Query is the dish you want to cook; each recipe has a **Key** (its title, "this is what I am") and a **Value** (its contents, "this is what I have to offer"). You **1)** compare the query to every key to score relevance, **2)** normalize the scores into focus percentages, **3)** blend the values by those percentages [recorded].

**The technical implementation.** "Each word in a sentence is disentangled into three vectors": **Query (Q)**: "the current word asking, 'Who should I pay attention to?'" **Key (K)**: "for each word in the sentence saying, 'This is what I am.'" **Value (V)**: "for each word in the sentence saying, 'This is what I have to offer.'" These are *learned linear projections* of the embeddings [recorded].

**The core operation: attention pooling.** Given a query $q$ and a database $D = \{(k_1, v_1), \dots, (k_m, v_m)\}$, the output is a weighted sum of values:
$$\boxed{\text{Attention}(q, D) = \sum_{i=1}^{m} \alpha(q, k_i)\, v_i} \quad \text{[recorded]}.$$
The scalars $\alpha(q, k_i)$ are the **attention weights** — how much weight each value earns. The mechanism "pays attention" to the $v_i$ whose $\alpha$ is significant.

**What the weights can be.** Three regimes (deck) [recorded]: **i) Convex combination** (the standard setting): $\alpha \ge 0$ and $\sum_i \alpha = 1$ — the output is a genuine average of values. **ii) Hard attention**: one weight is $1$, the rest $0$ — a database lookup; non-differentiable, so rarely used in deep models. **iii) Average pooling**: all weights equal $1/m$ — every value counted equally.

**Computing weights: two steps.** **1)** Choose any scoring function $a(q, k)$ measuring query–key relevance. **2)** Apply softmax to turn scores into a convex combination:
$$\boxed{\alpha(q, k_i) = \frac{\exp(a(q, k_i))}{\sum_{j=1}^{m} \exp(a(q, k_j))}} \quad \text{[recorded]}.$$
"Softmax" = exp each score, divide by the sum: non-negative, sums to 1, and differentiable — "its gradient never vanishes," which is why deep learning uses it.

**Basically, ...** "Attention is a soft database query. You hand over a question (query); each item advertises itself (key) and holds content (value). Relevance scores get squashed into percentages by softmax; your answer is the weighted blend of contents."

## 46.4 From kernels to the dot product

Any scoring function works. The deck starts from the **Gaussian kernel** (one of three similarity functions it lists, with boxcar and Epanechnikov):
$$a(q, k) = \exp\!\left(-\tfrac{1}{2}\|q - k\|^2\right) \quad \text{[recorded]}.$$
Now expand the exponent:
$$-\tfrac{1}{2}\|q - k\|^2 = q^\top k - \tfrac{1}{2}\|k\|^2 - \tfrac{1}{2}\|q\|^2 \quad \text{(deck, equation 1)} \quad \text{[recorded]}.$$
Two terms simplify away (deck's argument) [recorded]: **1)** the $q$ term $-\tfrac{1}{2}\|q\|^2$ is identical for all $(q, k_i)$ pairs — under softmax normalization "this term cancels out entirely"; **2)** the key term $-\tfrac{1}{2}\|k_i\|^2$ "can be dropped if the keys have constant norms" — "often the case, such as when keys are generated by layer normalization" (§46.11). What's left: the plain **dot product** $q^\top k_i$.

**Basically, ...** "Dot-product attention is Gaussian similarity with the boring parts cancelled: the query-only term dies in softmax, the key-only term dies when keys have equal length, and the pure 'how alike are they' part is just the dot product."

## 46.5 Why the $\sqrt{d_k}$: scaled dot-product attention

Dot products grow with dimension — and big scores kill softmax. The deck's variance argument [recorded]: if $q, k \in \mathbb{R}^d$ have independent entries with zero mean and unit variance, their dot product has zero mean but variance **$d$** (each of the $d$ terms $q_i k_i$ has variance $E[q_i^2]E[k_i^2] = 1$). Unscaled, the scores spread by $\sqrt{d}$; softmax saturates to a near-one-hot vector, its gradients die, and training stalls. The fix (Vaswani et al., 2017):
$$\boxed{a(q, k_i) = \frac{q^\top k_i}{\sqrt{d_k}}} \quad \text{[recorded]}.$$
Dividing by $\sqrt{d_k}$ keeps the score variance at $1$ regardless of vector length — softmax stays in its sensitive middle range. Note how this rhymes with §43.9's scaling sermon: networks break when numbers drift off a sane scale; here the architecture fixes the scale instead of the optimizer.

**The final form (matrix notation).** Queries $Q \in \mathbb{R}^{n \times d}$, keys $K \in \mathbb{R}^{m \times d}$, values $V \in \mathbb{R}^{m \times v}$:
$$\boxed{\text{Output} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \in \mathbb{R}^{n \times v}} \quad \text{[recorded]}.$$
Each of the $n$ queries gets one row: its attention weights over the $m$ keys, times the values.

**Basically, ...** "Long vectors make dot products huge, huge scores make softmax lazy (everything looks certain), and lazy softmax means dead gradients. Divide by $\sqrt{d}$ and the scores stay a manageable size no matter how wide the vectors get."

## 46.6 The matrix formula, by hand

**eg (own example — [verified-NumPy]).** Two tokens, $d = 3$: $x_1 = (1, 0, 1)$, $x_2 = (0, 1, 1)$. Take $W_q = W_k = I_3$ (queries and keys are the raw embeddings) and $W_v = \begin{bmatrix} 1 & 0 \\\\ 0 & 1 \\\\ 1 & 1 \end{bmatrix}$.

i) Scores $QK^\top$: $x_1^\top x_1 = 2$, $x_1^\top x_2 = 1$, $x_2^\top x_1 = 1$, $x_2^\top x_2 = 2$. Each token resembles itself most — exactly as it should.

ii) Scale by $\sqrt{3} \approx 1.7321$: rows become $[1.1547,\ 0.5774]$ and $[0.5774,\ 1.1547]$.

iii) Softmax: $e^{1.1547} \approx 3.1736$, $e^{0.5774} \approx 1.7814$, sum $4.9550$ → weights $[0.6405,\ 0.3595]$; the rows are symmetric.

iv) Values: $v_1 = x_1 W_v = (2, 1)$, $v_2 = (1, 2)$. Output row 1: $0.6405(2,1) + 0.3595(1,2) = (1.6405,\ 1.3595)$; row 2: $(1.3595,\ 1.6405)$.

Each output is a convex mix of both values — mostly its own token, about a third of the other. That is the whole mechanism on paper; numpy agrees to four decimals.

## 46.7 Self-attention, cross-attention, and the Bahdanau original

**i) Self-attention.** "Relates different positions of a single sequence... to compute a representation of that same sequence" [recorded]. $Q, K, V$ all come from the same input. The question: "For each token in this sequence, how relevant are the other tokens in the same sequence?" — the core of encoders like BERT.

**ii) Cross-attention.** "Relates the positions of two different sequences" [recorded]. Queries come from one sequence (the decoder), keys and values from another (the encoder). The question: "While generating a word in the target sentence, which words from the source sentence are most relevant?" — the mechanism that replaced §45.10's context vector in translation and summarization.

**The original: Bahdanau attention.** Attention predates the transformer — it was bolted onto §45.10's RNN encoder–decoder first (the notebook's class is literally named `Attention`, "Bahdanau attention mechanism"). The deck's formula for that generation: $\alpha_{ij} = \text{softmax}(\text{score}(s_{j-1}, h_i))$, $c_j = \sum_i \alpha_{ij} h_i$ [recorded]. The *machine_translation* notebook builds exactly this (German→English on Multi30k, bidirectional-GRU encoder, Bahdanau decoder): its **energy** $= \tanh(W[h; s])$, **attention** $= \text{softmax}(V \cdot \text{energy})$, **context** $= \sum(\text{attention} \cdot \text{encoder\_outputs})$ [recorded — torch]. Trained 20 epochs with teacher forcing $0.6$: test loss $3.445$, perplexity $31.359$, BLEU $0.3203$ (BLEU-1 $0.6327$ down to BLEU-4 $0.1672$) [recorded]. The transformer (next sections) keeps the attention idea and deletes the RNN underneath.

**Basically, ...** "Self-attention: one sequence reading itself. Cross-attention: the decoder reading the encoder, position by position. The first one ran on top of RNNs (Bahdanau); the transformer asked what happens if you keep the attention and throw away the RNN — that is the rest of this chapter."

## 46.8 The masked softmax

**The problem.** Batched sequences are padded to equal length ("Hello world \<blank\> \<blank\>"), and padding tokens "have no meaning and should not contribute to the attention output" [recorded].

**The solution: masking.** "Limit the attention summation to only the valid tokens." Implementation cheat: "Instead of using conditional logic (which is slow on GPUs), we take the scores for the padded positions and set them to a very large negative number (e.g., $-10^6$)." Then $\exp(-10^6) \approx 0$ — padded tokens get exactly zero weight after softmax [recorded].

**eg (own — [verified-NumPy]).** Scores $[2.0,\ 1.0,\ -10^6]$: $e^2 \approx 7.389$, $e^1 \approx 2.718$, $e^{-10^6} = 0$. Weights: $[7.389,\ 2.718,\ 0] / 10.107 = [0.7311,\ 0.2689,\ 0]$. The pad position is mathematically erased.

**Note.** Masking reappears in the decoder (§46.12) with a different target: instead of pads, it hides *future* tokens (the causal mask) — same $- \infty$ trick, different job.

**Basically, ...** "Padding tokens are filler. Adding $-\infty$ to their score before softmax is a cheap way to multiply them by zero: $\exp(-\text{huge}) = 0$."

## 46.9 Multi-head attention

**The problem.** "In the sentence 'The animal didn't cross the street because it was too tired', what does 'it' refer to? The animal or the street? A single attention mechanism might get confused" [recorded]. Sentences carry many relationship types at once.

**The solution: multiple heads.** "Multi-Head Attention runs all these experts at once" — the deck's committee-of-experts analogy: Expert 1 focuses on syntax (subject–verb agreement), Expert 2 on semantics (similar meanings), Expert 3 on pronouns (linking "it" to the noun) [recorded]. Formally: project $Q, K, V$ into $h$ lower-dimensional subspaces, run attention independently in each, then recombine.

For head $i \in \{1, \dots, h\}$, with learnable projection matrices $W^{(q)}_i, W^{(k)}_i, W^{(v)}_i$:
$$Q_i = Q W^{(q)}_i, \quad K_i = K W^{(k)}_i, \quad V_i = V W^{(v)}_i, \quad \text{head}_i = \text{Attention}(Q_i, K_i, V_i) \quad \text{[recorded]}.$$
Then concatenate along the feature dimension and mix once more:
$$\boxed{\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^{(o)}} \quad \text{[recorded]}.$$
"The final projection layer allows the model to learn how to best combine the information learned from the different attention heads."

**The parameter count.** With $d_{\text{model}} = 512$, $h = 8$ heads, $d_k = d_v = 64$: each head's three projections are $512 \times 64 = 32{,}768$ each ($98{,}304$ per head, $786{,}432$ total), and $W^{(o)}$ is $512 \times 512 = 262{,}144$. Total $1{,}048{,}576 = 4 \cdot 512^2$ [verified-NumPy] — *exactly* what one full-dimensional single head ($3 \cdot 512^2 + 512^2$) costs. Splitting into heads buys expressiveness for free.

**Basically, ...** "One attention head can only learn one kind of relationship. Eight heads are eight specialists — grammar, meaning, pronouns — whose opinions get mixed by a final vote. And it costs exactly the same as one big generalist."

## 46.10 Order is gone — positional encoding

**The problem.** Self-attention "is permutation-invariant. It sees a sentence as a 'bag of words' with no order" [recorded]: "The dog chased the cat" and "The cat chased the dog" are identical to it. Since order *is* meaning, the model needs position injected back.

**The solution: add a position vector to each embedding.** "Before feeding the words into the Transformer, we add a special vector called a Positional Encoding to each word's embedding" — "like giving each word a unique timestamp or sequence number" [recorded]. The deck's version (Vaswani et al.): sinusoidal encodings with per-dimension frequencies. For position $p$ and dimension index $i$ (embedding dimension $d$):
$$\boxed{PE(p, 2i) = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad PE(p, 2i+1) = \cos\!\left(\frac{p}{10000^{2i/d}}\right)} \quad \text{[recorded]}.$$

**Why sines.** For any fixed offset $\delta$, $PE(p+\delta)$ is a *linear transformation* of $PE(p)$ — "a constant rotation, which is easy for the model to learn" — from the sum-of-angles identities $\sin(\alpha+\beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$ and $\cos(\alpha+\beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$ [recorded]. So relative positions cost the model a linear map, not memorization. Low dimensions (small $i$) oscillate fast with position; high dimensions change slowly.

**eg (own — [verified-NumPy]).** $d = 4$: $PE(0) = [\sin 0, \cos 0, \sin 0, \cos 0] = [0,\ 1,\ 0,\ 1]$. For $p = 1$: $[\sin 1,\ \cos 1,\ \sin 0.01,\ \cos 0.01] = [0.8415,\ 0.5403,\ 0.0100,\ 1.0000]$. Check the rotation on the first pair with $\delta = 1$: $\begin{bmatrix}\cos 1 & \sin 1 \\\\ -\sin 1 & \cos 1\end{bmatrix}\begin{bmatrix}0\\\\1\end{bmatrix} = [\sin 1,\ \cos 1] = [0.8415,\ 0.5403]$ — matches $PE(1)$'s first two entries exactly.

**Note.** The nanoGPT notebook (§46.15) instead *learns* positional embeddings (`nn.Embedding(context_length, n_embeddings)`) — a learned lookup table rather than a fixed formula. Both work; the sinusoidal version extrapolates to longer sequences, the learned version is one less thing to design.

**Basically, ...** "Attention reads a bag of words, but language is a sentence. So we stamp each word with a position code made of slow and fast sine waves — and because shifting a sine is just a rotation, 'five words ahead' is a linear operation the model can actually learn."

## 46.11 Training deep stacks: residuals and layer norm

A transformer is "a deep stack of Attention Layers" — dozens or hundreds of sublayers. "Two more components are crucial" (deck): residual connections and layer normalization [recorded].

**i) Residual connections.** Problem: "the original input information can get distorted or lost after passing through many transformation layers." Solution: "An 'Information Express Lane'" — add the sublayer's input to its output. "This 'skip connection' allows the original signal to bypass the transformation, ensuring it's never lost" [recorded]. Same family as U-Net's skip connections (§44.9) — and the same gradient argument: the identity path gives every layer a direct gradient route, the architectural answer to §45.7's vanishing product.

**ii) Layer normalization.** Problem: "As data passes through layers, the numbers (values in the vectors) can become very large or very small, making the training process unstable." Solution: rescale each vector to mean $0$, variance $1$ — "independently for each example in the batch, using only the features of that single example" (the difference from batch norm) [recorded]. The math, for a vector $x$ with $H$ hidden units:
$$\mu = \tfrac{1}{H}\sum_{i=1}^H x_i, \qquad \sigma^2 = \tfrac{1}{H}\sum_{i=1}^H (x_i - \mu)^2, \qquad \hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y_i = \gamma_i \hat x_i + \beta_i \quad \text{[recorded]}.$$
The learnable gain $\gamma$ and bias $\beta$ "restore the expressive power of the network" — plain normalization alone would flatten every layer's output to the same shape. Why it matters: it "makes the residual connections ('Add') effective. Without normalization, repeatedly adding vectors would cause their magnitudes to grow layer by layer."

**eg (own — [verified-NumPy]).** $x = [1, 2, 3, 4]$: $\mu = 2.5$, $\sigma^2 = 1.25$, $\sigma \approx 1.1180$ → $\hat x = [-1.3416,\ -0.4472,\ 0.4472,\ 1.3416]$.

**Basically, ...** "Residuals are a skip lane: every sublayer adds its correction to the original signal instead of replacing it, so gradients always have a direct path home. Layer norm keeps the numbers on a sane scale (mean 0, variance 1) after every addition, so stacking a hundred layers doesn't blow up."

## 46.12 The transformer, end to end

<!-- Diagram: the original transformer architecture, Figure 1 of "Attention Is All You Need" (Vaswani et al., 2017), https://arxiv.org/abs/1706.03762 — the deck cites this paper's figure; encoder stack on the left, decoder stack on the right, with positional encodings, multi-head attention, masked attention, cross-attention, and the feed-forward blocks. -->

**The big picture.** "The Transformer follows a classic encoder-decoder structure": the **encoder** "reads the entire input sequence... builds a rich, context-aware representation of each token" (a stack of identical layers); the **decoder** "generates the output sequence one token at a time," attending to the encoder's output [recorded]. "Unlike RNNs, the Transformer processes all input tokens at once, relying entirely on self-attention instead of recurrence" — §46.1(ii)'s promise in architecture form.

**Inside an encoder layer.** Two sublayers, each wrapped as residual + layer norm ("Add & Norm") [recorded]: **1)** multi-head self-attention — "allows each token to look at all other tokens in the input sequence to understand its context"; **2)** a positionwise feed-forward network — "a simple, fully connected neural network applied independently to each token's representation" (expand–activate–project, §46.15's nanoGPT uses $4\times$ expansion with ReLU).

**Inside a decoder layer.** The same, plus one extra sublayer between them [recorded]: **1)** **masked** multi-head self-attention — "prevents the decoder from looking at future tokens in the output sequence it is trying to predict" (the causal mask: score entries above the diagonal are set to $-\infty$ before the softmax, §46.8's trick); **2)** **encoder–decoder cross-attention** — "the queries (Q) come from the decoder, while the keys (K) and values (V) come from the encoder's final output"; **3)** the feed-forward network.

**The full data flow.** **1)** Input embeddings: tokens → vectors. **2)** Positional encoding: add §46.10's vectors. **3)** Encoder stack: $N$ identical layers refine the representations. **4)** Decoder stack: previously generated tokens (shifted right) plus positional encoding → $N$ layers: masked self-attention, then cross-attention on the encoder output. **5)** Final linear layer + softmax: "a probability distribution over the entire vocabulary for the next token" [recorded].

**Complexity.** The deck's RNN-vs-transformer table: RNNs cost $O(T)$ sequential steps; the transformer does it in $O(1)$ parallel depth — matrix ops, no relay [recorded]. In FLOPs, attention costs $O(T^2 d)$ against the RNN's $O(T d^2)$; at $T = 1000, d = 512$: $5.12 \times 10^8$ vs $2.62 \times 10^8$ [verified-NumPy]. So the transformer does roughly *twice* the arithmetic — and wins anyway, because the arithmetic is parallel and the path length is constant. Scaling with hardware beats scaling with patience.

**Basically, ...** "Encoder: read everything, understand each token in full context. Decoder: write one token at a time, never peeking ahead (causal mask), always consulting the encoder (cross-attention). The whole thing is attention plus two-layer MLPs — no recurrence anywhere — so training runs in parallel."

## 46.13 Three flavors: BERT, GPT, and the full stack

The original transformer is encoder–decoder (T5, BART — "Translation, Summarization" [recorded]), but "many tasks do not require this full structure":

**i) Encoder-only (BERT).** "Goal: Understand the input. Sees: The full context (bidirectional)." Pre-trained with **masked language modeling**: 15% of tokens are replaced with `[MASK]` ("the cat sat on the [MASK]") and the model must recover them from *both* sides — plus **next sentence prediction**: "whether sentence B is the actual sentence that follows A" [recorded]. Best for NLU: classification, question answering, NER.

**ii) Decoder-only (GPT).** "Goal: Generate an output. Sees: Only the past (unidirectional)" via the causal mask. Pre-trained on one task: **standard language modeling** — given "The cat sat on the", predict "mat" [recorded]. Everything becomes text generation: "Translation becomes: 'English: The cat sits. French: ___'. Classification becomes: 'Review: Great movie! Sentiment: ___'" [recorded]. This is the architecture "the modern LLMs" use — simpler (one stack), scales better, generation is what users want.

**BERT vs GPT (deck's table, condensed)** [recorded]:

| | BERT (encoder) | GPT (decoder) |
|---|---|---|
| Visibility | Bidirectional — all tokens see each other | Causal — tokens see only the past |
| Pre-training | Masked language modeling | Next-token prediction |
| Best for | Understanding (NLU) | Generation (NLG) |

**The scale arc (deck's table)** [recorded]: original transformer — 2017, 65M params, 6 layers, 4.5M sentences; GPT-2 — 2019, 1.5B, 48 layers; GPT-3 — 2020, 175B, 96 layers, 300B tokens; GPT-4 — 2023, $\approx 1$T (est.), 3–10T tokens; LLaMA 3 — 2024, 8B/70B, 126 layers, 15T tokens. Context windows went 512 → 1M+ tokens. The empirical discovery behind it: **scaling laws** — performance "improves predictably with more parameters, training data, compute (GPU hours)" — and **emergent abilities** that "suddenly" appear at scale: basic arithmetic, code generation, few-shot learning, chain-of-thought reasoning [recorded].

**The paradigm shift.** Old: train from scratch per task. New: **1)** pre-train once on massive unlabeled text ("predict next token" — the model "learns language, facts, reasoning"); **2)** fine-tune cheaply on task data ("thousands, not billions" of examples), optionally with instruction tuning and RLHF — "Instruction Fine-tuning (SFT): Train on (instruction, response) pairs... Reinforcement Learning from Human Feedback (RLHF): Humans rank model outputs and train reward model optimized using RL (PPO)" [recorded]. Two more ingredients the deck names: **tokenization** ("unbelievable" → subword tokens ["un", "believ", "able"] via BPE/WordPiece/SentencePiece — "handles rare words, multiple languages, code") and **massive training data** ("trillions of tokens," "careful filtering and curation") [recorded].

**Basically, ...** "BERT reads both ways and fills blanks — good at understanding. GPT reads left-to-right and predicts the next word — good at writing; it turned out to be good at everything else too, just by framing tasks as text. Train one giant model once, adapt it cheaply for everything — that is the whole LLM economy."

## 46.14 The Vision Transformer

**The challenge.** Transformers eat 1D token sequences; images are 2D pixel grids. "Applying self-attention directly to every pixel in an image is computationally infeasible. For a modest $224 \times 224$ image, the number of pixel-to-pixel interactions would be $(224^2)^2 \approx 2.5 \times 10^9$" [recorded].

**The solution: patches as tokens.** "The core idea of the Vision Transformer (ViT) is to break an image down into a series of smaller, fixed-size patches and treat these patches as the tokens in a sequence." Three steps [recorded]: **1)** *Patching*: split $224 \times 224$ into $16 \times 16$ patches. **2)** *Flattening*: each patch unrolls to a vector ($16 \times 16 \times 3 = 768$ numbers). **3)** *Linear projection*: an MLP maps it to the model dimension $d$.

<!-- Diagram: ViT architecture, Figure 1 of "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020), https://arxiv.org/abs/2010.11929 — the deck cites this paper's figure; image → patches → linear projection → positional embeddings → transformer encoder. -->

**The convolutional shortcut.** The whole patchify-flatten-project pipeline "can be efficiently implemented as a single 2D convolution layer": kernel size = patch size ($16 \times 16$), stride = patch size (16, non-overlapping), output channels = embedding dim (768) [recorded]. Chapter 44's convolution as a patch extractor.

**eg (the deck's — [verified-NumPy]).** $H \times W = 224 \times 224$, $P \times P = 16 \times 16$, $D = 768$: $N = (224/16) \times (224/16) = 14 \times 14 = 196$ patches, each a 768-dim vector — sequence shape $\mathbb{R}^{196 \times 768}$. Prepend the learnable `[CLS]` token (below): 197 tokens. The conv shortcut uses $768 \times (16 \times 16 \times 3) + 768 = 590{,}592$ parameters.

**The `[CLS]` token.** "Inspired by the BERT model in NLP," a learnable token prepended to the patch sequence acts as "a global information aggregator": after the encoder stack, "the final, output vector corresponding to the '[CLS]' token's position is considered the representation of the entire image" and goes to "a simple MLP head for the final classification" [recorded]. (ViT uses *learnable* positional embeddings $e_i = z_i + P_{\text{pos}, i}$, not sinusoidal ones [recorded].)

**ViT vs CNNs (deck's table, condensed)** [recorded]: CNNs carry strong inductive bias — locality, translation equivariance — so they are "data-efficient and powerful on small to medium-sized datasets." ViTs make "almost no prior assumptions... aside from the initial patching" and learn global patch-to-patch relationships with self-attention — but are "data hungry" (pre-training on JFT-300M / ImageNet-21k), and "when pre-trained on enormous datasets... often outperform state-of-the-art CNNs when fine-tuned on downstream tasks." This is §44.10(ii)'s shortcut in vision: any patch talks to any other patch in one step, instead of waiting for pooling (§44.6's pyramid) to bring them together.

**Basically, ...** "Chop the image into 196 puzzle pieces, flatten each into a vector, and feed the pile to a transformer like it's a sentence. Self-attention lets the eye in the corner talk directly to the ear in the opposite corner — no pooling pyramid needed. The price: no built-in 'things are local' bias, so you need mountains of data to learn it."

## 46.15 The notebooks: a mini-GPT and pretrained models in one line

**i) *Transformers.ipynb*: nanoGPT from scratch** [recorded — torch code]. A decoder-only model on a toy corpus ("lemon tastes sour / apple tastes sweet / ...", a 19-token word-level vocab (17 words plus UNKNOWN/PADDING specials), `context_length = 12`): embedding dim 20, 2 layers, 2 heads, dropout $0.2$, AdamW at $10^{-4}$ (§43.12's transformer default, carried over whole), 10{,}000 iterations, batch 64. Architecture mirrors §§46.9–46.12: a `Head` does $K = XW_k$, $Q = XW_q$, $V = XW_v$, $\text{weights} = QK^\top/\sqrt{d_k}$, causal-mask with $-\infty$, softmax, output $= \text{weights}\,V$; `MultiheadAttention` concatenates heads and projects; `TransformerBlock` is **pre-norm** — LayerNorm → attention → residual, then LayerNorm → $4\times$ feed-forward (ReLU) → residual (note: the deck's §46.12 diagrams show post-norm "Add & Norm"; this implementation normalizes *before* each sublayer, the modern stable choice); the full model stacks blocks, applies a final LayerNorm, projects to the vocabulary, and trains with cross-entropy. Generation is autoregressive with top-$k$ sampling. Results after training: prompt "i like sour so i like" → **lemon at 99.46%**; "i like juicy so i like" → apple 61.55% / orange 37.68%; "i like spicy so i like" → orange 64.83% / chili 33.29% [recorded]. Even at 20 dimensions and 2 heads, the model picks up the taste↔fruit pattern — §46.13's "emergent" phenomenon in miniature.

**ii) *HuggingFace Inference.ipynb*: the ecosystem** [recorded]. Once trained, transformers are infrastructure: `transformers.pipeline` gives one-line access — sentiment analysis (default pipeline, plus a Twitter-RoBERTa variant), NER (BERT-Large fine-tuned on CoNLL-03), machine translation (T5-base English→French, a dedicated English→German model), and Gemini API calls for text generation, translation, and summarization. This is §46.13's pre-train + fine-tune paradigm as a developer tool: the giant model trains once; you call it.

**Basically, ...** "Twenty dimensions and two attention heads are enough to learn 'spicy goes with chili' from 430 characters — scale that thought by ten orders of magnitude and you get GPT. And once someone has done the training, HuggingFace lets you borrow the result in one line of code."

## 46.16 Where this goes next

i) **Generative models (Chapter 47).** The decoder of §46.12 *is* a generator: it scores sequences (the §45.1 modelling target) and samples new ones. Chapter 47 adds GANs and diffusion — two other ways to generate, built on the §44.10(iii) conv stacks and this chapter's decoder.

ii) **The optimizer rides along.** Chapter 43's §43.12 names AdamW the transformer standard; the mini-GPT notebook obeys. No new optimization machinery was needed in this chapter — attention is just another differentiable module (§42.4's machinery handles it).

iii) **The sharing moral, updated.** Chapter 44 shared across space, Chapter 45 across time — both bought efficiency and paid with path length. Attention pays the opposite way: direct pairwise interactions (§46.1(ii)), $O(T^2)$ compute, in exchange for hardware parallelism. Pick your poison per §40.5(i)'s shapes-first discipline: if $T$ is huge, that quadratic term is the first thing to check.

## Problem set

1. **One query, three keys.** Values $v_1 = (2, 0)$, $v_2 = (0, 4)$, $v_3 = (1, 1)$ with attention weights $\alpha = (0.6, 0.3, 0.1)$. (i) Compute $\text{Attention}(q, D)$. (ii) Show the result is a convex combination (weights $\ge 0$, sum $= 1$). (iii) Recompute under hard attention, assuming key 2 wins.
2. **Why scaling matters.** (i) Scores $[100, 50]$: compute softmax to 4 decimals and state what the weight vector approximates. (ii) Scaled scores $[10, 5]$: compute softmax. (iii) In two lines, connect (i) to the $\sqrt{d_k}$ division of §46.5 — what regime do unscaled dot products in high dimensions land in?
3. **The Gaussian-to-dot-product derivation.** Starting from $a(q, k_i) = -\tfrac{1}{2}\|q - k_i\|^2$ (no exponential): (i) expand into the three terms; (ii) name which term softmax normalization erases and why; (iii) state the assumption that erases the key-norm term, and name the chapter-46 component that usually enforces it.
4. **Multi-head parameter count.** $d_{\text{model}} = 512$, $h = 8$, $d_k = d_v = 64$. (i) Count one head's $W^{(q)}, W^{(k)}, W^{(v)}$ parameters. (ii) Add all eight heads and $W^{(o)}$ ($512 \times 512$). (iii) Compare with a single full-dimensional attention ($d_k = 512$): what is the ratio, and what does it say about the "cost" of splitting into heads?
5. **The causal mask.** (i) Write the $4 \times 4$ lower-triangular mask as 1s/0s. (ii) Which positions can token index 2 (0-based) attend to? (iii) In one line, say why the decoder needs this but the encoder does not.
6. **Positional encoding by hand.** $d = 4$. (i) Write $PE(0)$. (ii) Compute $PE(1)$ numerically. (iii) Show the first pair $(PE(0,0), PE(0,1))$ rotates into $(PE(1,0), PE(1,1))$ for $\delta = 1$ using the sum-of-angles identity, and state what this means for relative positions.
7. **Masked softmax.** Scores $[2.0,\ 1.0,\ -10^6]$. (i) Compute the three weights. (ii) What does the third weight equal exactly after masking, and why? (iii) Name the two different jobs masking does in a transformer (§46.8 and §46.12).
8. **Patch math.** $224 \times 224 \times 3$ image, $16 \times 16$ patches, $D = 768$. (i) How many patches? (ii) How long is each flattened patch vector? (iii) How many parameters does the conv shortcut use (with bias)? (iv) What is the encoder's input sequence length including `[CLS]`?
9. **Cross-attention shapes.** Decoder sequence length $T_d = 8$, encoder length $T_e = 30$, model dim $512$. (i) What are the shapes of the cross-attention weight matrix? (ii) Show the parameter count of the decoder's cross-attention layer is independent of $T_d$ and $T_e$; name the book's term for this property (Chapter 44/45).
10. **The quadratic bill.** (i) $T = 1000$, $d = 512$: compare attention's $O(T^2 d)$ with an RNN's $O(T d^2)$ numerically. (ii) If attention does roughly twice the FLOPs here, why does the deck still say transformers "scale well with data + hardware" while RNNs "slow down as sequences grow"? (iii) As $T$ grows at fixed $d$, which term dominates — and what does that imply for very long contexts?

---

*Sources: GenAI Weeks 9–10 "Introduction to Attention and Transformer Architecture" deck (Balaji Srinivasan, Ganapathy Krishnamurthi), 99 slides — embeddings (meaning space, static vs contextual, king−man+woman≈queen, learn-from-scratch recipe), attention pooling ($\sum_i \alpha(q,k_i)v_i$), Q/K/V cookbook analogy and projections, Gaussian/boxcar/Epanechnikov kernels, Gaussian→dot-product derivation, $\sqrt{d}$ variance argument, softmax weight formula, matrix form $\text{softmax}(QK^\top/\sqrt{d})V$, self vs cross attention, masked softmax ($-10^6$ trick), RNN-vs-transformer table, multi-head (committee analogy, $Q_i = QW^{(q)}_i$ formulas, concat + $W^{(o)}$), permutation invariance, sinusoidal PE + relative-position rotation, residual connections, layer-norm math ($\mu, \sigma^2, \hat x, \gamma, \beta$), encoder/decoder layer diagrams, full data flow, BERT (MLM 15%, NSP) vs GPT (causal LM) table, scaling table (65M→1T est.), pre-train + fine-tune, tokenization, emergent abilities, SFT/RLHF, ViT (patch pipeline, conv shortcut, [CLS] token, 196-patch worked example, ViT-vs-CNN table); notes `Week 9-10.pdf` — same deck content, used as cross-check; `machine_translation.ipynb` — Multi30k pipeline, Bahdanau `Attention` class (energy/attention/context formulas), GRU encoder–decoder, teacher forcing $0.6$, 20 epochs, test loss $3.445$ / PPL $31.359$ / BLEU $0.3203$ (all torch [recorded]); `HuggingFace Inference.ipynb` — pipeline API: sentiment (Twitter-RoBERTa), NER (dbmdz/bert-large-cased-finetuned-conll03-english), MT (google-t5/t5-base en→fr, specific en→de), Gemini API examples (all [recorded]); Week 12 `Transformers.ipynb` — nanoGPT: 19-token vocab, context 12, $d=20$, 2 layers, 2 heads, dropout $0.2$, AdamW $10^{-4}$, 10k iters; `Head`/`MultiheadAttention`/`TransformerBlock` (pre-norm, $4\times$ FFN) classes, causal mask, generation samples (all torch [recorded]); diagram references — Vaswani et al. 2017 Fig. 1 (https://arxiv.org/abs/1706.03762), Dosovitskiy et al. 2020 Fig. 1 (https://arxiv.org/abs/2010.11929); book chapters 42 (§42.4), 43 (§43.9, §43.12), 44 (§44.6, §44.8, §44.10), 45 (§45.1, §45.7, §45.10, §45.11, §45.13).*
