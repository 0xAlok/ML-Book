# Solutions — Chapter 46: Attention and the transformer

Full worked solutions. All numbers re-run in numpy [verified-NumPy].

## 1. One query, three keys

(i) $\text{Attention} = 0.6(2,0) + 0.3(0,4) + 0.1(1,1) = (1.2 + 0 + 0.1,\ 0 + 1.2 + 0.1) = (1.3,\ 1.3)$.

(ii) Every $\alpha_i \ge 0$ and $0.6 + 0.3 + 0.1 = 1.0$ — a convex combination, so the output lies inside the convex hull of $\{v_1, v_2, v_3\}$ (§46.3).

(iii) Hard attention: key 2 takes all the weight, $\alpha = (0, 1, 0)$ → output $= v_2 = (0, 4)$. The $v_1$ and $v_3$ information is discarded entirely — the §46.3 point about differentiability.

## 2. Why scaling matters

(i) $\text{softmax}([100, 50])$: $e^{100}/(e^{100} + e^{50}) = 1/(1 + e^{-50}) \approx 1.0000$, the second weight $\approx 1.93 \times 10^{-22} \approx 0.0000$. The vector approximates a one-hot $[1, 0]$.

(ii) $\text{softmax}([10, 5])$: $e^{10} \approx 22{,}026.47$, $e^5 \approx 148.41$, sum $22{,}174.88$ → $[0.9933,\ 0.0067]$. Still confident, but the second score earns a real (if small) weight.

(iii) Unscaled dot products in dimension $d$ spread with standard deviation $\sqrt{d}$ — for $d = 512$ that is $\approx 22.6$, landing squarely in the $[100, 50]$-style regime where softmax collapses to one-hot and its gradients vanish (§46.5). The $\sqrt{d_k}$ division rescales the spread to $\approx 1$, keeping softmax in its sensitive middle range where small score differences still move the weights.

## 3. The Gaussian-to-dot-product derivation

(i) $-\tfrac{1}{2}\|q - k_i\|^2 = q^\top k_i - \tfrac{1}{2}\|k_i\|^2 - \tfrac{1}{2}\|q\|^2$.

(ii) The $-\tfrac{1}{2}\|q\|^2$ term is identical for every $i$, so $\exp(a_i) = \exp(\text{const}) \cdot \exp(a_i - \text{const})$ and the constant factors out of both numerator and denominator of the softmax and cancels (§46.4).

(iii) Keys must have constant norms (so $-\tfrac{1}{2}\|k_i\|^2$ is also $i$-independent and drops out the same way). Layer normalization (§46.11) is the component that usually enforces this — another reason it sits before attention in deep stacks.

## 4. Multi-head parameter count

(i) One head: $W^{(q)}, W^{(k)}, W^{(v)}$ are each $512 \times 64 = 32{,}768$ → $3 \times 32{,}768 = 98{,}304$ parameters.

(ii) Eight heads: $8 \times 98{,}304 = 786{,}432$; plus $W^{(o)}$ ($512 \times 512 = 262{,}144$): total $1{,}048{,}576$.

(iii) Single full-dimensional attention ($d_k = d_v = 512$): three $512 \times 512$ projections plus one $512 \times 512$ output = $4 \times 262{,}144 = 1{,}048{,}576$. Ratio $1{:}1$ — splitting into eight heads costs exactly nothing extra in parameters (§46.9); the "committee of experts" is free.

## 5. The causal mask

(i) Lower-triangular ones:
$$\begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 1 & 1 & 0 & 0 \\\\ 1 & 1 & 1 & 0 \\\\ 1 & 1 & 1 & 1 \end{bmatrix}.$$

(ii) Token 2 attends to positions $0, 1, 2$ — current and past, never future (the $- \infty$ trick of §46.8 erases the zeros).

(iii) The decoder *predicts* future tokens; seeing them would let it cheat during training and break the train/test match of §45.10's teacher-forcing discussion. The encoder reads the whole input to *understand* it — there is no future to leak, so no mask.

## 6. Positional encoding by hand

(i) $PE(0) = [\sin 0,\ \cos 0,\ \sin 0,\ \cos 0] = [0,\ 1,\ 0,\ 1]$.

(ii) $PE(1) = [\sin 1,\ \cos 1,\ \sin 0.01,\ \cos 0.01] = [0.8415,\ 0.5403,\ 0.0100,\ 1.0000]$.

(iii) Sum-of-angles with $\alpha = 0$, $\beta = 1$: $\sin(0+1) = \sin 0\cos 1 + \cos 0\sin 1 = \sin 1$; $\cos(0+1) = \cos 0\cos 1 - \sin 0\sin 1 = \cos 1$. So the first pair of $PE(0)$ rotates by the fixed $2\times2$ matrix $\begin{bmatrix}\cos 1 & \sin 1 \\\\ -\sin 1 & \cos 1\end{bmatrix}$ into the first pair of $PE(1)$ — $[0.8415,\ 0.5403]$ matches (ii) exactly. Meaning: for a fixed offset $\delta$, position $p+\delta$ is a constant linear function of position $p$, so the model learns *relative* positions with a linear map instead of memorizing absolute ones (§46.10).

## 7. Masked softmax

(i) $e^2 \approx 7.389$, $e^1 \approx 2.718$, $e^{-10^6} = 0$ (underflow) → $[0.7311,\ 0.2689,\ 0]$.

(ii) Exactly $0$: $\exp(-10^6)$ underflows to zero in floating point, so the padded position contributes nothing to the weighted sum — "mathematically erased" (§46.8 eg).

(iii) Job one (§46.8): erase *padding* tokens in batched inputs. Job two (§46.12): erase *future* tokens in the decoder's causal mask. Same $- \infty$ mechanism, different targets.

## 8. Patch math

(i) $N = (224/16)^2 = 14^2 = 196$ patches.

(ii) Each flattens to $16 \times 16 \times 3 = 768$ numbers.

(iii) Conv shortcut: $768$ filters $\times (16 \times 16 \times 3)$ weights $+ 768$ biases $= 768 \times 768 + 768 = 590{,}592$.

(iv) $196 + 1$ (`[CLS]`) $= 197$ tokens.

## 9. Cross-attention shapes

(i) Queries come from the decoder ($8$ positions), keys/values from the encoder ($30$ positions): the weight matrix is $8 \times 30$ — each decoder position gets a distribution over all $30$ encoder positions.

(ii) The cross-attention projections are $512 \times d_k$, $512 \times d_k$, $512 \times d_v$ and $W^{(o)}$ — sized by dimensions only, never by $T_d$ or $T_e$. Same $4 \cdot 512^2$ parameters for a 3-word caption or a 30-word paragraph: the book's weight-sharing property (§44.10(i), the §45.2(iii) theme), here across positions rather than pixels or timesteps.

## 10. The quadratic bill

(i) Attention: $T^2 d = 1000^2 \times 512 = 5.12 \times 10^8$. RNN: $T d^2 = 1000 \times 512^2 = 2.62 \times 10^8$. Attention does $\approx 1.95 \times$ the FLOPs — roughly twice.

(ii) The deck's table (§46.12): the RNN's cost is $O(T)$ *sequential depth* — step 500 cannot start before step 499 finishes, so adding hardware does not help. Attention's $O(T^2 d)$ is one big matrix multiply: $O(1)$ sequential depth, fully parallel, exactly what GPUs/TPUs are built for. Transformers scale with *hardware*, RNNs scale with *patience*.

(iii) At fixed $d$, the $T^2$ term dominates as $T$ grows — attention becomes quadratic in context length. That is the ceiling the deck's "512 tokens → 1M+ tokens" arc is really about (§46.13): long contexts are an active engineering fight, not a solved problem.
