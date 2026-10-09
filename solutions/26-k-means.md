# Solutions — Chapter 26: K-means clustering

## Problem 1 — Objective arithmetic

Data $\{0, 1, 9, 10\}$, $K = 2$.

**(i)** $z = (1,1,2,2)$. Cluster 1: $\{0,1\}$, $\mu_1 = 0.5$; contributions $(0-0.5)^2 + (1-0.5)^2 = 0.25 + 0.25 = 0.5$. Cluster 2: $\{9,10\}$, $\mu_2 = 9.5$; contributions $0.25 + 0.25 = 0.5$.
$$\boxed{F = 1}.$$

**(ii)** $z = (1,2,2,2)$. Cluster 1: $\{0\}$, $\mu_1 = 0$; contribution $0$. Cluster 2: $\{1,9,10\}$, $\mu_2 = \tfrac{20}{3}$; contributions
$$\left(1-\tfrac{20}{3}\right)^2 + \left(9-\tfrac{20}{3}\right)^2 + \left(10-\tfrac{20}{3}\right)^2 = \tfrac{289}{9} + \tfrac{49}{9} + \tfrac{100}{9} = \tfrac{438}{9} = \tfrac{146}{3}.$$
$$\boxed{F = \tfrac{146}{3} \approx 48.67}.$$

**(iii)** $z = (1,1,2,2)$ is better. "Better" here means *smaller $F$*: the points sit tighter around their centroids (two natural pairs), whereas $z = (1,2,2,2)$ lumps $\{1,9,10\}$ into one stretched cluster. $F$ is the score; smaller wins.

## Problem 2 — One Lloyd step

Data $\{1, 2, 9, 10\}$, $K = 2$, $z^0 = (1,1,1,2)$.

**(i)** $\mu_1^0 = \tfrac{1+2+9}{3} = \boxed{4}$, $\mu_2^0 = \boxed{10}$.

**(ii)** Squared distances to $(4, 10)$:
- $x=1$: $9$ vs $81$ → 1;
- $x=2$: $4$ vs $64$ → 1;
- $x=9$: $25$ vs $1$ → 2;
- $x=10$: $36$ vs $0$ → 2.
$$\boxed{z^1 = (1,1,2,2)}.$$

**(iii)** $F^0$: cluster 1 $= \{1,2,9\}$ about $\mu_1^0 = 4$: $9+4+25 = 38$; cluster 2 $= \{10\}$: $0$. $\boxed{F^0 = 38}$. Recompute means for $z^1$: $\mu_1^1 = 1.5$, $\mu_2^1 = 9.5$. $F^1$: cluster 1: $0.25+0.25 = 0.5$; cluster 2: $0.25+0.25 = 0.5$. $\boxed{F^1 = 1 < 38 = F^0}$ ✓ — the unhappy point $x = 9$ defected and $F$ collapsed.

## Problem 3 — Monotonicity, proved

**(i)** Fix the assignment; for cluster $k$ minimize $g(\mu) = \sum_{i:z_i=k}\|x_i-\mu\|^2$. $\nabla_\mu g = -2\sum_{i:z_i=k}(x_i-\mu) = 0$ gives $\mu\sum_{i:z_i=k}1 = \sum_{i:z_i=k}x_i$, i.e.
$$\boxed{\mu_k = \frac{\sum_{i:z_i=k}x_i}{\#\{i:z_i=k\}}},$$
the centroid. (Hessian $2m_k I \succ 0$: it is the minimizer.)

**(ii)** The compute-means step replaces each $\mu_k$ by the minimizer of that cluster's contribution to $F$ while the assignment is frozen — each cluster's sum can only drop, so $F$ cannot increase.

**(iii)** The reassignment step freezes the centers; each point's new label is $\arg\min_k\|x_i-\mu_k^t\|^2$, the minimizer of that point's term in $F$. Every term can only drop, so $F$ cannot increase.

**(iv)** Both steps are non-increasing, so $\boxed{F^{t+1} \le F^t}$. Strictness: the mean step is strictly decreasing unless the centers were already the centroids; the reassignment step is strictly decreasing iff at least one point moves (a point moves only to a *strictly* closer center). So $F^{t+1} < F^t$ iff at least one point changes cluster in iteration $t$.

## Problem 4 — Bisector geometry

**(i)** $\|x-\mu_1\|^2 = \|x-\mu_2\|^2$: $x_1^2+x_2^2 = (x_1-4)^2+x_2^2$, i.e. $0 = -8x_1+16$:
$$\boxed{x_1 = 2}.$$

**(ii)** $(1,1)$: to $\mu_1$: $1+1 = 2$; to $\mu_2$: $9+1 = 10$. Closer to $\mu_1$ → $\boxed{\text{cluster } 1}$ (also: $1 < 2$, left of the boundary).

**(iii)** $\|x-\mu_1\|^2 = \|x-\mu_2\|^2 \iff -2x^T\mu_1 + \|\mu_1\|^2 = -2x^T\mu_2 + \|\mu_2\|^2 \iff 2x^T(\mu_2-\mu_1) = \|\mu_2\|^2-\|\mu_1\|^2$. This is a hyperplane with normal vector $\mu_2-\mu_1$ — hence perpendicular to the segment $\mu_1\mu_2$. The midpoint $m = \tfrac{\mu_1+\mu_2}{2}$ satisfies it: $2m^T(\mu_2-\mu_1) = (\mu_1+\mu_2)^T(\mu_2-\mu_1) = \|\mu_2\|^2-\|\mu_1\|^2$ ✓. So the boundary is the perpendicular bisector. ∎

## Problem 5 — k-means++ scoring

**(i)** $S(x) = \|x - 0\|^2 = x^2$: $\boxed{S(0) = 0,\ S(5) = 25,\ S(6) = 36}$.

**(ii)** Total $= 61$:
$$\boxed{P(\mu_2^0 = 0) = 0,\quad P(\mu_2^0 = 5) = \tfrac{25}{61} \approx 0.410,\quad P(\mu_2^0 = 6) = \tfrac{36}{61} \approx 0.590}.$$

**(iii)** $x = 6$ is most likely (highest score → highest probability). No — the choice is *probabilistic*, not a max: $6$ is picked with probability $\tfrac{36}{61}$, not $1$.

## Problem 6 — Choosing $K$

**(i)** With $K = n$, put each $x_i$ in its own cluster; then $\mu_{z_i} = x_i$ and every term $\|x_i-\mu_{z_i}\|^2 = 0$. $\boxed{F = 0}$ for any dataset.

**(ii)** $\boxed{K = 1}$ gives the largest optimal $F$: one centroid must serve all $100$ points, so every point's distance is measured from the single global mean; adding centers can only let points sit closer to their own center (practice Q7's argument).

**(iii)** Because $F$ alone is minimized by the trivial $K = n$ ($F = 0$, useless) — without a penalty on $K$ there is nothing stopping the degenerate solution.

## Problem 7 — True or false

(i) **False.** Different initializations can converge to different local minima of $F$ (practice Q9(b)).
(ii) **True.** $F^{t+1} \le F^t$ every iteration — neither Lloyd step can increase $F$ (Problem 3; practice Q2/Q6).
(iii) **False.** Every reassignment strictly reduces $F$, so a visited partition — with its $F$ value — can never recur (practice Q6(a)).
(iv) **False.** k-means minimizes squared distances about means; both are outlier-sensitive (practice Q10).
(v) **True.** Each cluster region is an intersection of $K-1$ half-spaces (§26.6); half-spaces are convex (§12.3) and intersections preserve convexity (§12.4).

## Problem 8 — Hard vs soft

**(i)** EM's M-step (§25.9): $\mu_k = \frac{1}{N_k}\sum_{i=1}^{n}\gamma(z_{ik})x_i$ with $N_k = \sum_i\gamma(z_{ik})$. If $\gamma(z_{ik}) \in \{0,1\}$, the sum keeps exactly the points with $\gamma(z_{ik}) = 1$ — i.e. the points *assigned* to $k$ — and $N_k$ is their headcount:
$$\boxed{\mu_k = \frac{\sum_{i:z_i=k}x_i}{\#\{i:z_i=k\}}},$$
k-means' centroid formula (§26.3). $\pi_k = N_k/n$ becomes the *fraction of data points in cluster $k$* — the empirical cluster proportion.

**(ii)** k-means' reassignment stamps each point with a single hard label (nearest centroid); EM's E-step instead distributes each point *softly* across components via Bayes' rule.
