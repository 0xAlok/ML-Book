# Solutions — Chapter 29: KNN and decision trees

## Problem 1 — 3-NN vote, by hand ($q_2 = (4,2)$)

(i) Squared Euclidean distances (from the §29.5 table):
- $d^2(q_2, A_1) = (4-0)^2 + (2-0)^2 = 16 + 4 = 20$
- $d^2(q_2, A_2) = (4-1)^2 + (2-4)^2 = 9 + 4 = 13$
- $d^2(q_2, A_3) = (4-3)^2 + (2-1)^2 = 1 + 1 = 2$
- $d^2(q_2, B_1) = (4-5)^2 + (2-2)^2 = 1 + 0 = 1$
- $d^2(q_2, B_2) = (4-4)^2 + (2-5)^2 = 0 + 9 = 9$
- $d^2(q_2, B_3) = (4-6)^2 + (2-3)^2 = 4 + 1 = 5$
- $d^2(q_2, B_4) = (4-7)^2 + (2-2)^2 = 9 + 0 = 9$

(ii) Sorted: $B_1 (1) < A_3 (2) < B_3 (5) < B_2 (9) = B_4 (9) < A_2 (13) < A_1 (20)$. The $3$ nearest: $\boxed{B_1, A_3, B_3}$.

(iii) Votes: $B, A, B$ → $2$ for $B$, $1$ for $A$ → $\boxed{\text{predict } B}$.

(iv) $k = 5$: neighbours $B_1, A_3, B_3, B_2, B_4$ (the $9$-tie is between two $B$'s, so the set is unaffected) → votes $B,A,B,B,B$ → $4$ vs $1$ → $\boxed{\text{predict } B}$.

## Problem 2 — The metric can flip the answer

(i) Euclidean (squared): $d^2(q,a) = 25 + 0 = 25$; $d^2(q,b) = 9 + 9 = 18$. Nearest is $\boxed{b}$ → predict $\boxed{-}$.

(ii) Manhattan: $d_1(q,a) = 5 + 0 = 5$; $d_1(q,b) = 3 + 3 = 6$. Nearest is $\boxed{a}$ → predict $\boxed{+}$.

(iii) Because "nearest" — and hence the vote — depends on how distance is measured; the metric is a modelling choice with observable consequences, i.e. a hyperparameter (§29.2(ii), §29.3(i)).

## Problem 3 — KNN regression

(i) $\hat y = \tfrac13(2.0 + 3.5 + 6.5) = \tfrac{12.0}{3} = \boxed{4.0}$.

(ii) Classification labels are discrete — they can only be counted (voted); regression outputs are numbers, so they can be averaged — the mean is the central value of the neighbourhood's outputs (§29.4(i)).

## Problem 4 — The effect of $k$

(i) From §29.5's sorted $d^2$ for $q = (2,2)$: $A_3 (2), A_2 (5), A_1 (8), B_1 (9), B_2 (13), B_3 (17), B_4 (25)$.
- $k=1$: $\{A_3\}$ → $\boxed{A}$.
- $k=3$: $\{A_3, A_2, A_1\}$ → $3$ $A$ → $\boxed{A}$.
- $k=5$: $\{A_3, A_2, A_1, B_1, B_2\}$ → $3$ $A$, $2$ $B$ → $\boxed{A}$.

(ii) $k = 7$: the neighbourhood is all $7$ points → $3$ $A$ vs $4$ $B$ → $\boxed{B}$. This is the lecture's endpoint: as $k \to n$, the vote is the dataset's majority class ($B$), so every query — $q$ included — is labelled $B$ (§29.6(iii)).

(iii) At $k = 1$ the boundary is jagged: each training point owns an intricate Voronoi-like cell, and the border weaves between nearby points. At $k = 5$ the border relaxes into one smooth curve separating the blue cluster from the reds — the local wiggles are voted out (§29.6(i)–(ii); see the §29.5 figure).

## Problem 5 — Verify the lecture's Wind number

$IG(S, \text{Wind}) = H(S) - \sum_v \tfrac{|S_v|}{|S|}H(S_v)$:
$$IG = 0.940 - \frac{8}{14}(0.811) - \frac{6}{14}(1.0) = 0.940 - 0.4634 - 0.4286 = 0.0480 \approx \boxed{0.048}.$$
Matches the lecture's $0.048$ (full-precision recomputation in the review log gives $0.0481$).

## Problem 6 — Finish the §29.13 comparison

(i) Red branch ($3$ $+$, $2$ $-$):
$$H = -\tfrac35\log_2\tfrac35 - \tfrac25\log_2\tfrac25 \approx 0.442 + 0.529 = \boxed{0.971}.$$
Blue branch ($1$ $+$, $4$ $-$):
$$H = -\tfrac15\log_2\tfrac15 - \tfrac45\log_2\tfrac45 \approx 0.464 + 0.258 = \boxed{0.722}.$$

(ii) Weighted child entropy: $\tfrac{5}{10}(0.971) + \tfrac{5}{10}(0.722) = 0.8465$.
$$IG(S, \text{colour}) = 0.971 - 0.8465 = \boxed{0.125} < 0.420 = IG(S, \text{size}).$$

(iii) ID3 splits on the largest information gain → root attribute is $\boxed{\text{size}}$.

## Problem 7 — Why Gini beats misclassification for growing

Parent: $80$ points, $40$ $+$, $40$ $-$.

(i) Misclassification error of a child $= 1 - \max(p_+, p_-) = \tfrac{\min(n_+, n_-)}{n_{\text{child}}}$.
- Split A: left $\tfrac{10}{40} = 0.25$, right $\tfrac{10}{40} = 0.25$; weighted: $\tfrac{40}{80}(0.25) + \tfrac{40}{80}(0.25) = \boxed{0.25}$.
- Split B: left $\tfrac{20}{60} = \tfrac13$, right $\tfrac{0}{20} = 0$; weighted: $\tfrac{60}{80}\cdot\tfrac13 + \tfrac{20}{80}(0) = 0.25 + 0 = \boxed{0.25}$.
Identical — misclassification cannot distinguish the splits.

(ii) Gini of a child $= 2p_+(1-p_+)$ (binary case).
- Split A: each child $2\cdot\tfrac34\cdot\tfrac14 = 0.375$; weighted $\boxed{0.375}$.
- Split B: left $2\cdot\tfrac13\cdot\tfrac23 = \tfrac49 \approx 0.444$, right $0$; weighted: $\tfrac{60}{80}\cdot\tfrac49 = \boxed{\tfrac13 \approx 0.333}$.

(iii) Gini prefers $\boxed{\text{split B}}$ ($0.333 < 0.375$). It is the better split because its right child is perfectly pure ($20$ $+$, $0$ $-$) — the split has isolated a clean region, while split A leaves both children equally mixed.

(iv) This is the lecture's point verbatim: Gini (and cross-entropy) are "more sensitive to the node probabilities" — they register the purity improvement in B's right child, whereas the misclassification rate only cares about majority flips and scores both splits the same (§29.16(i)).

## Problem 8 — Cost-complexity arithmetic

$C(T) = \sum_i Q_i(T) + \alpha|T|$.

(i) $\alpha = 1$: $C(T_0) = 10 + 1\cdot 4 = \boxed{14}$; $C(T) = 14 + 1\cdot 2 = \boxed{16}$. $\boxed{T_0 \text{ wins}}$ (keep the larger tree).

(ii) $\alpha = 3$: $C(T_0) = 10 + 3\cdot 4 = \boxed{22}$; $C(T) = 14 + 3\cdot 2 = \boxed{20}$. $\boxed{T \text{ wins}}$ (prune).

(iii) General rule (the lecture's): $\boxed{\text{larger } \alpha \Rightarrow \text{smaller trees}}$ — $\alpha$ prices each leaf, so raising it makes complex trees too expensive to keep (§29.19).
