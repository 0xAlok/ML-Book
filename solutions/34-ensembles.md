# Solutions — Chapter 34: Ensembles — bagging, boosting, AdaBoost

*Full worked solutions to the Chapter 34 problem set. Every number recomputed independently (hand algebra cross-checked in numpy; see the review log).*

## Problem 1 — Vote arithmetic

Five independent classifiers, each with error rate $e = 0.2$. Let $S$ = number wrong; $S \sim \mathrm{Bin}(5, 0.2)$.

**(i)** The majority is wrong iff $S \ge 3$:
$$P(\text{vote wrong}) = \sum_{k=3}^{5}\tbinom{5}{k}(0.2)^k(0.8)^{5-k}$$
$$= \tbinom{5}{3}(0.2)^3(0.8)^2 + \tbinom{5}{4}(0.2)^4(0.8) + (0.2)^5$$
$$= 10(0.008)(0.64) + 5(0.0016)(0.8) + 0.00032 = 0.0512 + 0.0064 + 0.00032 = \boxed{0.05792}.$$

**(ii)** Yes: $0.05792 < 0.2$ — the vote is far better than each voter (same moral as §34.2's eg 1).

**(iii)** **Independence** fails for real bagged trees: they train on resamples of the *same* data, so they are correlated, and the true gain is smaller than the binomial ideal (§34.3).

## Problem 2 — OOB counting

**(i)** Each point is OOB for about one-third of the trees: $\approx B/3 = \boxed{200}$ trees.

**(ii)** Each tree trains on about $n(1 - 1/e) \approx 150 \times 0.632 \approx \boxed{95}$ distinct points ("around two-thirds," §34.4).

**(iii)** Point $i$'s OOB prediction uses only trees that never trained on it — so the OOB error measures performance on genuinely unseen data, like a test set.

## Problem 3 — AdaBoost by hand (Q5-style)

$e = 30/100 = 0.3$. With $s = e^{\alpha} = \sqrt{(1-e)/e} = \sqrt{7/3}$:

**(i)** $\boxed{\alpha = \tfrac12\ln(7/3) \approx 0.4236}$.

**(ii)** Unnormalized: misclassified $\frac{1}{100}s = \frac{\sqrt{7/3}}{100} \approx \boxed{0.015275}$; correctly classified $\frac{1}{100}/s = \frac{\sqrt{3/7}}{100} \approx \boxed{0.0065465}$.

**(iii)** Normalizer $Z = \frac{30s + 70/s}{100}$. Final misclassified weight:
$$\frac{s}{30s + 70/s} = \frac{s^2}{30s^2 + 70} = \frac{7/3}{30\cdot 7/3 + 70} = \frac{7/3}{140} = \boxed{\frac{1}{60} \approx 0.016667}.$$
(This is the MLT practice assignment Q5's answer (c), recomputed: $\sqrt{7/3}\,/\,(30\sqrt{7/3} + 70\sqrt{3/7}) = 1/60$.)

## Problem 4 — Reading $\alpha$

**(i)** $\alpha_m = \tfrac12\ln\!\left(\frac{1-e_m}{e_m}\right) > 0 \iff \frac{1-e_m}{e_m} > 1 \iff 1 - e_m > e_m \iff \boxed{e_m < 0.5}$.

**(ii)** $e_m = 0.5 \Rightarrow \boxed{\alpha_m = 0}$: the classifier contributes *nothing* to the weighted vote — a coin-flip stump is ignored.

**(iii)** If $e_m \ge 0.5$ then $\alpha_m \le 0$ — the round adds no positive signal (or votes against itself), so reweighting cannot steer the ensemble toward the hard points.

## Problem 5 — Bagging vs boosting, sourced

**(i)** **False.** Boosting does *not* use bootstrap sampling; each tree is fit on a modified (reweighted) version of the original data (ISL §8.2.3; §34.5(i), §34.8(i)).

**(ii)** **False.** Unlike bagging, boosting *can* overfit if $B$ is too large — $B$ is chosen by cross-validation (ISL §8.2.3; §34.8(iii)).

**(iii)** **True.** Under squared-error loss, averaging reduces variance and leaves bias unchanged (ESL §8.7; §34.2).

**(iv)** **False.** Boosting trees are grown *sequentially*, each using information from the previously grown trees (ISL §8.2.3; §34.8(i)).

## Problem 6 — The $m$-knob

**(i)** Decreasing the features per split **reduces** the tree-to-tree correlation (wider variety of trees).

**(ii)** Increasing the features per split **increases** each individual tree's strength (more features to pick the split from).

**(iii)** Decorrelated trees are what make averaging actually reduce variance — fully correlated trees vote identically and bagging gains nothing (§34.3).
