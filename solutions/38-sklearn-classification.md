# Solutions — Chapter 38: scikit-learn: classification workflows

## Problem 1 — The contract

(i) `fit(X_train, y_train)` learns $(w, b)$ from the training rows (e.g. the perceptron's mistake-driven updates, §31.4; logistic regression's cross-entropy minimization, §31.10). `predict(X_test)` returns one label per row: $\hat y = \mathrm{sign}(w^T x + b)$ for the linear family (§22.9). `decision_function(X_test)` returns the *un-snapped* score $w^T x + b$ per row — the confidence behind the sign. `score(X_test, y_test)` returns the mean accuracy, i.e. $1 - L$ with $L$ = §22.9's 0/1 loss.

(ii) A regressor's `score` does not return accuracy; it returns $R^2 = 1 - u/v$ (§37.5) — "what fraction of the label's variance the model explains."

(iii) `RidgeClassifier` never models $P(y \mid x)$: §38.3's construction converts labels to $\{-1,+1\}$ and runs *regression*; its output is a real number whose sign is read as the class. There is no probability to report — `predict_proba` would be a fiction. `LogisticRegression` genuinely fits §31.8's $P(y=1 \mid x) = \sigma(\theta^T x + \theta_0)$, so its probabilities are the model itself.

## Problem 2 — Confusion matrix by hand

(i) $n = 40 + 50 + 10 + 20 = 120$.
$$\text{accuracy} = \frac{40 + 50}{120} = \frac{90}{120} = \boxed{0.75}, \qquad
\text{precision} = \frac{40}{40 + 10} = \frac{40}{50} = \boxed{0.80},$$
$$\text{recall} = \frac{40}{40 + 20} = \frac{40}{60} \approx \boxed{0.6667}, \qquad
F_1 = \frac{2 \times 0.8 \times 0.6667}{0.8 + 0.6667} = \frac{1.0667}{1.4667} \approx \boxed{0.7273}.$$

(ii) The disease is rare, so $TN = 50$ dominates the accuracy numerator — a classifier that calls nearly everyone healthy still scores well. Recall $0.67$ says a third of the sick are missed (the costly error); precision $0.80$ says one in five alarms is false. Accuracy hides both behind the healthy majority.

## Problem 3 — The `C` translation

(i) Slide: minimize $\text{penalty} + C \cdot \text{loss}$ over $w$. Divide the whole objective by the positive constant $C$ (scaling an objective doesn't move its minimizer):
$$\frac{1}{C}\,\text{penalty} + \text{loss} \quad\Longleftrightarrow\quad \text{loss} + \frac{1}{C}\,\text{penalty}.$$
Matching against the book's §28.3 form $\text{loss} + \lambda \cdot \text{penalty}$ gives $\boxed{\lambda = 1/C}$.

(ii) $C$ multiplies the *loss*: shrinking $C$ shrinks the loss term's weight, so the penalty term dominates the trade-off — the optimizer prefers smaller weights, i.e. stronger regularization.

(iii) §37.7's ridge `alpha` *is* the book's $\lambda$, so for the identical regularization strength: $\boxed{C = 1/\mathrm{alpha}}$ (with the caveat that the two estimators' objectives normalize the loss differently — logistic regression averages nothing out front the way the book's $J$ does — so treat the reciprocal as the dial's direction and scale, exact only up to that normalization).

## Problem 4 — Course-sourced (`LabelBinarizer`)

(i) Classes sorted: apple, orange, pear ($k = 3$). One-hot per row:
$$\begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & 1 & 0
\end{bmatrix}$$
`'apple'` → column 0, `'pear'` → column 2, `'orange'` → column 1. This matches the recorded output `[[1 0 0] [0 0 1] [1 0 0] [0 1 0]]` exactly.

(ii) $n = 4$ examples (rows), $k = 3$ distinct classes (columns) — one column per class, one row per example.

## Problem 5 — Threshold moving

(i) Lowering the threshold can only flip predictions from negative to positive, never the reverse — so the predicted-positive count is non-decreasing, $TP$ is non-decreasing while the positive pool ($TP + FN$) is fixed, and $\text{recall} = TP/(TP+FN)$ is non-decreasing. (Precision typically falls: the extra positives are the model's least confident calls.)

(ii) Threshold **up** from $0.5$: fewer positive ("spam") calls, each surer. Protected metric: **precision** of the spam verdict — "of the emails I flagged, (almost) all were truly spam." (The cost — missed spam, lower recall — is explicitly accepted.)

## Problem 6 — Spot the leak

(i) The violated discipline is §37.8's Note (and §36.7's imputation rule): *fit on train, transform everywhere*. `pca.fit(x_test1)` learns the projection's mean and principal directions from the **test** rows — test statistics leak into the fitted transformer. Beyond the leak, something worse breaks: train and test are projected onto *different* bases (two independent PCAs), so the classifier trains in one coordinate system and is tested in another — the $0.5564$ accuracy is the wreckage.

(ii) One fit, on train only:
```python
p = pca.fit(x_train1)
x_train1_reduced = p.transform(x_train1)
x_test1_reduced  = p.transform(x_test1)
```
or structurally leak-proof:
```python
pipe = make_pipeline(PCA(n_components=10, random_state=1), clf)
pipe.fit(x_train1, y_train1)     # every step learns from train only
pipe.predict(x_test1)            # test rows only ever see transform
```

## Problem 7 — OVR vs OVO

(i) $k = 5$: `OneVsRestClassifier` trains $\boxed{5}$ classifiers (one per class, $c$-vs-rest). `OneVsOneClassifier` trains $\binom{5}{2} = \frac{5 \times 4}{2} = \boxed{10}$ classifiers (one per pair).

(ii) Each OVO duel trains on only the two classes' rows — roughly $2/k$ of the data per classifier — so a base estimator whose training cost blows up with $n$ (the slide's "does not scale with the data") sees small, cheap subproblems instead of one big one. (OVR's $k$ classifiers each still see all $n$ rows.)

## Problem 8 — Diagnose the fit

(i) Positive rows: $TP + FN = 185$. Recall $= TP/185 = 0.0703$ → $TP \approx 0.0703 \times 185 = 13$, $FN = 172$. Precision $= TP/(TP+FP) = 1.0$ → $FP = 0$. Accuracy $= (TP+TN)/381 = 0.5486$ → $TN = 0.5486 \times 381 - 13 = 209 - 13 = 196$. So $(TN, FP, FN, TP) = (196, 0, 172, 13)$ — exactly §38.10's recorded matrix.

(ii) The suspect is `shuffle=False`. The course stacked the training matrix as all $+1$s then all $-1$s; with shuffling off, SGD (§38.6: "the gradient of the loss is estimated with one sample at a time") sees 863 fives first and walks far into positive territory, then 1032 threes drag the weights back — the final model sits in negative territory and predicts $-1$ almost everywhere ($FN = 172$). Shuffling (`shuffle=True`) interleaves the classes so no single class dominates any stretch of the walk — the slide's "it is important to permute (shuffle) the training data before fitting."
