# Solutions — Chapter 1: Sets, functions, and mathematical preliminaries

## Problem 1

$A = \{x \in \mathbb{N} : x < 5\}$.

Since this book takes $\mathbb{N} = \{0, 1, 2, \dots\}$:

Step 1: List naturals below 5: $0, 1, 2, 3, 4$.

Step 2: $A = \{0, 1, 2, 3, 4\}$, so $|A| = 5$.

Note: If you learned $\mathbb{N}$ starting at 1 elsewhere, you would get $\{1, 2, 3, 4\}$ — the convention matters, which is why the book states it up front.

## Problem 2

i) $\{4, 16\} \subseteq \{16, 4, 32\}$.

Check each element of the left set: $4$ is in the right set, $16$ is in the right set. True.

ii) $\{4, 16\} \subset \{16, 4\}$.

The left set is a subset of the right set (same check as i), but the sets are equal — order does not matter, so $\{16, 4\} = \{4, 16\}$. A proper subset must be strictly smaller. False.

iii) $\{4, 16, 32\} \subset \{16, 4, 32\}$.

Again equal sets (order irrelevant). False — for the same reason as ii).

## Problem 3

$U = [0, 10]$, $A = [2, 5]$, $B = [4, 7]$.

Step 1: $A \cap B$. Need $2 \le x \le 5$ and $4 \le x \le 7$: both hold exactly on $[4, 5]$.

Step 2: $(A \cap B)^c = [4, 5]^c$ inside $[0, 10]$: everything except $[4,5]$, i.e. $[0, 4) \cup (5, 10]$.

Step 3: $A^c = [0, 2) \cup (5, 10]$ (remove $[2,5]$ from $[0,10]$).

Step 4: $B^c = [0, 4) \cup (7, 10]$ (remove $[4,7]$ from $[0,10]$).

Step 5: $A^c \cup B^c = ([0, 2) \cup (5, 10]) \cup ([0, 4) \cup (7, 10])$.

Collect: below 4, $[0,4)$ is covered by $B^c$; above 5, $(5,10]$ is covered by $A^c$. Check the boundary points: $4 \in A$ and $4 \in B$, so $4$ is in neither complement; $5 \in A$ and $5 \in B$, so $5$ is in neither complement. Hence $A^c \cup B^c = [0, 4) \cup (5, 10]$.

Step 6: This equals $(A \cap B)^c$ from Step 2. De Morgan's second law verified.

## Problem 4

$A \times B = \{(x, y) : x \in A,\ y \in B\}$.

List: $A \times B = \{(1, x), (1, y), (2, x), (2, y)\}$.

Count: 4 elements (every one of the 2 choices from $A$ pairs with every one of the 2 choices from $B$).

## Problem 5

i) $R_1 = \{(1,2), (2,3), (3,3)\}$: the first elements $1, 2, 3$ each occur exactly once, so each input has exactly one output — $R_1$ is a function.

$R_2 = \{(1,2), (1,3), (2,2)\}$: the input $1$ is paired with two different outputs ($2$ and $3$), violating "exactly one output per input" — $R_2$ is not a function.

ii) For $R_1$: domain $= \{1, 2, 3\}$ (all first elements); range $= \{2, 3\}$ (the second elements that occur).

## Problem 6

i) $f: \mathbb{R} \to \mathbb{R}$, $f(x) = x^2 + 1$.

Injective? $f(1) = 2 = f(-1)$ with $1 \ne -1$. No.

Surjective? $f(x) \ge 1$ always, so e.g. $0 \in \mathbb{R}$ is never hit. No.

Neither.

ii) $f: \mathbb{R} \to \mathbb{R}$, $f(x) = 5x - 7$.

Injective: $f(x_1) = f(x_2) \Rightarrow 5x_1 - 7 = 5x_2 - 7 \Rightarrow 5x_1 = 5x_2 \Rightarrow x_1 = x_2$. Yes.

Surjective: for any $y \in \mathbb{R}$, $x = \frac{y + 7}{5}$ satisfies $f(x) = 5\cdot\frac{y+7}{5} - 7 = y$. Yes.

Bijective.

iii) $f: \mathbb{N} \to \mathbb{N}$, $f(n) = n + 1$.

Injective: $n_1 + 1 = n_2 + 1 \Rightarrow n_1 = n_2$. Yes.

Surjective? $0 \in \mathbb{N}$: is there $n \in \mathbb{N}$ with $n + 1 = 0$? That needs $n = -1 \notin \mathbb{N}$. No.

Injective but not surjective.

Basically, ... shifting every natural number up by one leaves a hole at 0 — nothing maps to it.

## Problem 7

i) $f(x) = \dfrac{1}{x - 2}$.

The denominator cannot be zero: $x - 2 \ne 0$. Domain $= \mathbb{R} \setminus \{2\}$.

ii) $g(x) = \sqrt{x + 1}$.

Need $x + 1 \ge 0$, i.e. $x \ge -1$. Domain $= [-1, \infty)$.

iii) $h(x) = \dfrac{1}{\sqrt{x - 1}}$.

Two requirements: $x - 1 \ge 0$ (square root) and $\sqrt{x - 1} \ne 0$ (denominator). Together: $x - 1 > 0$. Domain $= (1, \infty)$.

## Problem 8

$f(x) = x + 1$, $g(x) = 2x$.

$(f \circ g)(x) = f(g(x)) = f(2x) = 2x + 1$.

$(g \circ f)(x) = g(f(x)) = g(x + 1) = 2(x + 1) = 2x + 2$.

$(f \circ f)(x) = f(f(x)) = f(x + 1) = (x + 1) + 1 = x + 2$.

No two are equal: $2x + 1$, $2x + 2$, and $x + 2$ are all different functions. (Domains are all $\mathbb{R}$.)

## Problem 9

$f(x) = \sqrt{x}$, $g(x) = x - 5$.

Step 1: $(f \circ g)(x) = f(g(x)) = \sqrt{x - 5}$.

Step 2: Domain rule i): $x \in \text{domain}(g) = \mathbb{R}$ — no restriction.

Step 3: Domain rule ii): $g(x) \in \text{domain}(f)$, i.e. $x - 5 \ge 0$, so $x \ge 5$.

Step 4: Domain of $f \circ g$ is $[5, \infty)$.

Note: Even though $g$ accepts every real, the composition rejects $x < 5$ because $g$'s output would not fit $f$'s input slot.

## Problem 10

$f: \mathbb{R} \to \mathbb{R}$, $f(x) = \dfrac{x + 1}{3}$.

Step 1: Set $y = \dfrac{x + 1}{3}$ and solve for $x$: $3y = x + 1$, so $x = 3y - 1$. Hence $f^{-1}(x) = 3x - 1$.

Step 2: $f^{-1}(f(x)) = 3\left(\dfrac{x + 1}{3}\right) - 1 = (x + 1) - 1 = x$. ✓

Step 3: $f(f^{-1}(x)) = \dfrac{(3x - 1) + 1}{3} = \dfrac{3x}{3} = x$. ✓

## Problem 11

$R = \{(1,1), (2,2), (3,3), (1,3), (3,1)\}$ on $S = \{1, 2, 3\}$.

Reflexive? $(1,1), (2,2), (3,3)$ all present. Yes.

Symmetric? The non-diagonal pairs are $(1,3)$ and $(3,1)$ — each other's reverse is present. Yes.

Transitive? Check chains through distinct elements: $(1,3), (3,1) \in R \Rightarrow$ need $(1,1) \in R$ — present. $(3,1), (1,3) \in R \Rightarrow$ need $(3,3) \in R$ — present. Chains involving only diagonals are automatic. Yes.

All three hold, so $R$ is an equivalence relation.

## Problem 12

i) $\forall x \in \mathbb{Z},\ x \in \mathbb{Q}$ — equivalently, $\forall x\ (x \in \mathbb{Z} \Rightarrow x \in \mathbb{Q})$.

ii) False. For every real $x$, $x^2 \ge 0$, so $x^2 = -1$ has no real solution.
