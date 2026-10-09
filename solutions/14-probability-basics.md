# Solutions — 14. Probability basics: experiments, sample spaces, Bayes

**1.** Restaurant hiring. Write $D$ = David, $M$ = Megha (Delhi); $R$ = Rajesh, $V$ = Veronica (Mumbai). Ordered pairs are (waiter, cashier).

(a) $S = \{(D,M), (D,R), (D,V), (M,D), (M,R), (M,V), (R,D), (R,M), (R,V), (V,D), (V,M), (V,R)\}$ — $4 \cdot 3 = 12$ outcomes, equally likely.
(b) $A =$ cashier $\in \{D, M\}$: $\{(D,M), (M,D), (R,D), (R,M), (V,D), (V,M)\}$ — $6$ outcomes. $B =$ exactly one of the two positions is Delhi: waiter Delhi & cashier Mumbai $\{(D,R), (D,V), (M,R), (M,V)\}$ plus waiter Mumbai & cashier Delhi $\{(R,D), (R,M), (V,D), (V,M)\}$ — $8$ outcomes. $C =$ neither is Delhi: $\{(R,V), (V,R)\}$ — $2$ outcomes.
(c) $P(A) = 6/12 = 1/2$; $P(B) = 8/12 = 2/3$; $P(C) = 2/12 = 1/6$.

**2.** Mixed-up hats, 5 persons.

(a) $A^c =$ "at least one person gets their own hat" $= D$. $B^c =$ "at least one person does not get their own hat" $= C$.
(b) Yes, disjoint. $B$ is the single identity ordering (everyone keeps their own hat), which is not a derangement, so $A \cap B = \emptyset$.
(c) Since $A \cap B = \emptyset$, $A \subseteq B^c$, so $A \cap B^c = A$.

**3.** IPL over.

(a) $A \cup B$: "the over had no 4s or no 6s" (at least one of the two holds). $A \cap B$: "the over had no 4s and no 6s".
(b) No. Under $A \cap B$, every delivery scores at most $3$ runs, so the over total is at most $6 \cdot 3 = 18 < 20$; $C$ requires exactly $20$. Hence $A \cap B \cap C = \emptyset$.
(c) $(A \cup B)^c = A^c \cap B^c$: "the over had at least one 4 *and* at least one 6". $(A \cap B)^c = A^c \cup B^c$: "the over had at least one 4 *or* at least one 6".

**4.** Let $F =$ "catch $> 400$", $E =$ "catch $> 500$". Then $E \subseteq F$, and "between 400 and 500 Kg" is $F \setminus E$. By the subset property (§14.7(iii)): $P(F \setminus E) = P(F) - P(E) = 0.35 - 0.10 = \boxed{0.25}$ ($25\%$).

**5.** Let $R =$ rain, $H =$ temp above $30^\circ$. Inclusion–exclusion: $P(R \cup H) = 0.6 + 0.7 - 0.4 = 0.9$. "No rain and temp at most $30^\circ$" $= (R \cup H)^c$, so by the complement property: $P = 1 - 0.9 = \boxed{0.10}$.

**6.** Given $P(A \cap B) = P(A)P(B)$. Then
$$P(A \cap B^c) = P(A \setminus B) = P(A) - P(A \cap B) = P(A) - P(A)P(B) = P(A)(1 - P(B)) = P(A)P(B^c),$$
using §14.7(iv) for $P(A \setminus B) = P(A) - P(A \cap B)$ and §14.7(ii) for $P(B^c) = 1 - P(B)$. So $A$ and $B^c$ are independent. Applying the same argument to the independent pair $(A, B^c)$: $A^c$ and $(B^c)^c = B$ are independent; applying once more to $(A^c, B)$: $A^c$ and $B^c$ are independent. ∎

**7.** Two urns, $P(\text{urn 1}) = P(\text{urn 2}) = 1/2$.

(a) Law of total probability: $P(\text{red}) = P(\text{red} \mid \text{urn 1})P(\text{urn 1}) + P(\text{red} \mid \text{urn 2})P(\text{urn 2}) = \tfrac{7}{13}\cdot\tfrac{1}{2} + \tfrac{5}{13}\cdot\tfrac{1}{2} = \tfrac{12}{26} = \boxed{6/13}$.
(b) Bayes: $P(\text{urn 1} \mid \text{red}) = \dfrac{P(\text{urn 1})P(\text{red} \mid \text{urn 1})}{P(\text{red})} = \dfrac{\tfrac{1}{2}\cdot\tfrac{7}{13}}{\tfrac{6}{13}} = \boxed{7/12}$.
(c) $P(\text{blue} \mid \text{urn 1}) = 6/13$, $P(\text{blue} \mid \text{urn 2}) = 8/13$. $P(\text{blue}) = \tfrac{6}{13}\cdot\tfrac{1}{2} + \tfrac{8}{13}\cdot\tfrac{1}{2} = \tfrac{14}{26} = 7/13$. Bayes: $P(\text{urn 2} \mid \text{blue}) = \dfrac{\tfrac{1}{2}\cdot\tfrac{8}{13}}{\tfrac{7}{13}} = \boxed{4/7}$. (Sanity: $P(\text{urn 1} \mid \text{blue}) = 3/7$, and $3/7 + 4/7 = 1$. ✓)

**8.** Let $U =$ unemployment rises, $R =$ rates rise. Law of total probability: $P(U) = P(U \mid R)P(R) + P(U \mid R^c)P(R^c) = 0.6 \cdot 0.4 + 0.3 \cdot 0.6 = 0.24 + 0.18 = \boxed{0.42}$.

**9.** Partition by coin type: $P(\text{double-headed}) = 2/5$, $P(\text{double-tailed}) = 1/5$, $P(\text{normal}) = 2/5$. $P(\text{head}) = 1\cdot\tfrac{2}{5} + 0\cdot\tfrac{1}{5} + \tfrac{1}{2}\cdot\tfrac{2}{5} = \tfrac{2}{5} + \tfrac{1}{5} = \boxed{3/5}$.

**10.** Let $K =$ knows the answer, $C =$ answered correctly. $P(K) = 3/4$, $P(K^c) = 1/4$, $P(C \mid K) = 1$, $P(C \mid K^c) = 1/4$. Denominator: $P(C) = 1\cdot\tfrac{3}{4} + \tfrac{1}{4}\cdot\tfrac{1}{4} = \tfrac{3}{4} + \tfrac{1}{16} = \tfrac{13}{16}$. Bayes: $P(K \mid C) = \dfrac{(3/4)\cdot 1}{13/16} = \dfrac{3}{4}\cdot\dfrac{16}{13} = \boxed{12/13 \approx 0.923}$.

**11.** Let $D =$ "die showed 5", $F =$ "5 heads obtained". $P(D) = 1/6$, $P(F \mid D) = \binom{5}{5}(1/2)^5 = 1/32$. Partition over die faces $k = 1,\dots,6$: $P(F \mid k) = \binom{k}{5}(1/2)^k$ for $k \ge 5$, else $0$. So $P(F) = \tfrac{1}{6}\left[\tbinom{5}{5}(1/2)^5 + \tbinom{6}{5}(1/2)^6\right] = \tfrac{1}{6}\left[\tfrac{1}{32} + \tfrac{6}{64}\right] = \tfrac{1}{6}\cdot\tfrac{4}{32} = \tfrac{1}{48}$. Bayes: $P(D \mid F) = \dfrac{(1/32)(1/6)}{1/48} = \dfrac{1/192}{1/48} = \boxed{1/4}$.

**12.** $S = \{HH, HT, TH, TT\}$, uniform, so each singleton has $P = 1/4$.

(a) $P(A) = P(\{HH, TT\}) = 1/2$, $P(B) = P(\{HH, HT\}) = 1/2$, $P(C) = P(\{HH, TH\}) = 1/2$. $A \cap B = \{HH\}$, $P = 1/4 = \tfrac{1}{2}\cdot\tfrac{1}{2}$ ✓; $A \cap C = \{HH\}$, $P = 1/4$ ✓; $B \cap C = \{HH\}$, $P = 1/4$ ✓. All three pairs independent.
(b) $A \cap B \cap C = \{HH\}$, $P = 1/4 \ne 1/8 = P(A)P(B)P(C)$. The triple-product condition fails, so $A, B, C$ are pairwise but not mutually independent.

**13.** Let $q_1 = 0.4$ (Player 1 scores), $q_2 = 0.7$ (Player 2 scores); misses $0.6, 0.3$.

(a) "Wins before the 3rd round" = wins in round 1 or round 2. Round 1: $P = 0.4$. Round 2: Player 1 misses, Player 2 misses, Player 1 scores: $0.6 \cdot 0.3 \cdot 0.4 = 0.072$. Total: $0.4 + 0.072 = \boxed{0.472}$.
(b) Player 1 wins in round $r$ iff both miss the first $r-1$ rounds and he scores in round $r$: $P_r = (0.6 \cdot 0.3)^{r-1} \cdot 0.4 = 0.4 \cdot 0.18^{\,r-1}$ (rounds are disjoint events). Sum the geometric series: $\sum_{r=1}^{\infty} 0.4 \cdot 0.18^{r-1} = \dfrac{0.4}{1 - 0.18} = \dfrac{0.4}{0.82} = \boxed{\dfrac{20}{41} \approx 0.4878}$.
