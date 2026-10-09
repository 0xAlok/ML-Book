# Solutions — Chapter 22. What is ML: data, models, tasks; supervised vs unsupervised

## Problem 1

Machine learning = "the study of computer algorithms that improve automatically through experience and by the use of data" (Wikipedia definition, MLF Week 1 lecture 1).

| Level | Who builds the tool? |
|---|---|
| Manual labour | No tool — the human directly converts input to output. |
| Programming | The human builds the tool (software) that converts input to output. |
| Machine learning | The human gives only a broad blueprint to a *tool design* (the learning algorithm); the tool design uses data to construct the tool (the model). |

## Problem 2

(a) **Password verification → programming.** Manual labour fails on scale/speed/cost (billions of logins); but a program comparing the input against the password on file is trivial. Since programming works, ML is never considered.

(b) **Face detection → machine learning.** Manual labour fails on scale (a human drawing boxes on every photo). Programming fails because the rule "which pixel patterns are a face" cannot be expressed in code — humans know faces (a baby learns from examples) but cannot convey the idea to a pixel-seeing computer. ML works: lots of face/non-face images exist, and the algorithm builds the tool from them.

(c) **Weather prediction → machine learning.** Manual labour fails — humans don't know the rule mapping today's radar map to tomorrow's rain (physics implements it; nobody wrote it down). Programming fails for the same reason: you can't code a rule you don't know. ML works because the rule exists and there is immense historical weather data.

## Problem 3

(i) House 2 as a vector: $\boxed{(2,\ 7,\ 2.1,\ 3.2)} \in \mathbb{R}^4$.

(ii) In words: a two-bedroom house with $700$ sq.ft of area, $2.1$ km from the metro, priced at $32$ lakhs ($3.2$ tens of lakhs).

(iii) The computer never reads the metadata — it only manipulates numbers. What it needs is *consistency*: coordinate 2 must mean "area in 100 sq.ft" for house 2 *and* every other house, otherwise arithmetic across rows is meaningless. The metadata (the sentence) is for the human, to make the data interpretable.

## Problem 4

(i) **Predictive** (regression): given a new house's area and distance, it outputs a predicted price — a real-valued prediction about the future/unseen.

(ii) **Probabilistic**: it doesn't predict a value for a new input; it *scores* configurations — how likely is this tweet under Chopra's tweet distribution.

(iii) Uses: the regression model prices houses you've never seen (eg 5: $700$ sq.ft at $3$ km → $5$ lakhs). The probabilistic model powers the Chopra-tweet generator (§22.13): score candidate tweets, keep/generate the high-scoring ones.

## Problem 5

Predictions of $k(x) = 1.5x_1 + 1$: $2.5,\ 4,\ 5.5,\ 10,\ 11.5$. Against $(2.1, 3.9, 6.2, 11.5, 13.9)$:
$$L(k) = \tfrac15\big[(2.5-2.1)^2 + (4-3.9)^2 + (5.5-6.2)^2 + (10-11.5)^2 + (11.5-13.9)^2\big]$$
$$= \tfrac15\big[0.16 + 0.01 + 0.49 + 2.25 + 5.76\big] = \tfrac{8.67}{5} = \boxed{1.734}.$$
Ranking: $L(f) = 0.064 < L(k) = 1.734 < L(g) = 5.264$. The algorithm still prefers $f(x) = 2x_1$; $k$ is better than $g$ but worse than $f$.

## Problem 6

$m(\mathbf{x}) = \mathrm{sign}(x_2 - 1)$, true labels $(+1,+1,+1,-1,-1,-1)$:
$$\begin{array}{c|c|c|c}
 & x_2-1 & m & \text{truth} \\ \hline
(0,0) & -1 & -1 & +1 \;\; \text{wrong} \\
(1,0) & -1 & -1 & +1 \;\; \text{wrong} \\
(0,1) & 0 & +1 & +1 \;\; \text{right} \\
(4,4) & 3 & +1 & -1 \;\; \text{wrong} \\
(3,4) & 3 & +1 & -1 \;\; \text{wrong} \\
(4,3) & 2 & +1 & -1 \;\; \text{wrong}
\end{array}$$
Five mistakes out of six: $\boxed{L(m) = 5/6}$. Ranking: $f\ (0) < g\ (1/6) < m\ (5/6)$. ($m$ draws a horizontal cut at $x_2 = 1$; it keeps only $(0,1)$ on the right side.)

## Problem 7

Zero training loss means the model is perfect *on points it has already seen* — and this $f$ achieves it by pure memorization: it hard-codes $+1$ exactly at the three positive training points and defaults to $-1$ everywhere else. It learned no principle (e.g. "positives live near the origin"), so on a *new* point like $(0.1, 0.1)$ — which any reasonable rule calls $+1$ — it predicts $-1$.

To expose it, evaluate on **test data**: fresh points drawn the same way but never shown during training. The memorizer will fail there while a genuine separator (like eg 9's $f$) succeeds. Training loss measures fit; test loss measures learning.

## Problem 8

Encodings $f(\mathbf{x}) = x_1 + x_2$: $1.8,\ 4.2,\ 6.2,\ 7.8$. Reconstructions $g(u) = (u/2, u/2)$: $(0.9,0.9),\ (2.1,2.1),\ (3.1,3.1),\ (3.9,3.9)$ — identical to the good pair's reconstructions in eg 11.
$$L = \tfrac14\big[(0.1^2+0.1^2)\times 4\big] = \tfrac{0.08}{4} = \boxed{0.02},$$
exactly tying eg 11's good pair, and far below the bad pair's $15.2$.

What you notice: the loss only sees the *round trip* $g \circ f$. Here $g(f(\mathbf{x})) = \tilde g(\tilde f(\mathbf{x})) = \big(\tfrac{x_1+x_2}{2}, \tfrac{x_1+x_2}{2}\big)$ — the factor of 2 moved from the encoder into the decoder, but the composition (and hence the loss) is unchanged. Different encoder–decoder pairs can be equivalent.

## Problem 9

$P_2$: scores $\frac15, \frac15$ on $\{1.0, 2.0\}$:
$$L(P_2) = -\log\tfrac15 - \log\tfrac15 = 2\ln 5 \approx \boxed{3.219}.$$
$P_4$: scores $\frac12, \frac12$:
$$L(P_4) = -\log\tfrac12 - \log\tfrac12 = 2\ln 2 \approx \boxed{1.386}.$$
$P_4$ wins. Same reasoning as eg 12: both integrate to 1, but $P_4$ concentrates its unit mass on $[0,2]$ where the data actually sits, instead of spreading it thinly over $[0,5]$. Negative log-likelihood rewards putting probability mass on observed data.

## Problem 10

(a) **Regression** — predictive model; squared loss $L(f) = \frac1n\sum (f(\mathbf{x}^i)-y^i)^2$. Labels are real numbers (temperatures).

(b) **Classification** — predictive model; misclassification fraction $L(f) = \frac1n\sum \mathbf{1}(f(\mathbf{x}^i) \ne y^i)$. Labels are discrete (spam/not-spam).

(c) **Dimensionality reduction** — compression (neither predictive nor probabilistic in the lecture's taxonomy); reconstruction loss $L(f,g) = \frac1n\sum \|g(f(\mathbf{x}^i))-\mathbf{x}^i\|^2$. No labels at all.

(d) **Density estimation** — probabilistic model $P$; negative log-likelihood $L(P) = -\sum \log P(\mathbf{x}^i)$. No labels; the goal is scoring configurations, not predicting values.
