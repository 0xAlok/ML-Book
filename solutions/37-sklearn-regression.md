# Solutions — Chapter 37: scikit-learn: regression workflows

## Problem 1 — The contract

(i) In §23.1's notation: `fit(X_train, y_train)` computes $\hat\theta$ — the parameter vector minimizing the training loss (for `LinearRegression`, §23.5's $(X^TX)^{-1}X^Ty$). `predict(X_test)` computes $\hat y = X_{\text{test}}\hat\theta$, one prediction per row. `score(X_test, y_test)` returns $R^2 = 1 - u/v$ on the held-out pair (§37.5).

(ii) A transformer has no `predict` — it transforms features, it doesn't model labels. Instead of `predict` it has `transform(X)`: apply the *learned* transformation. (`fit` learns e.g. $\mu, \sigma$; `transform` maps $x' = (x - \mu)/\sigma$; `fit_transform` does both.)

## Problem 2 — $R^2$ by hand

(i) Residuals $y - \hat y = (1, -1, 1, 0, -1)$: $u = 1 + 1 + 1 + 0 + 1 = 4$. Mean $\bar y = 20/5 = 4$; deviations $(-2, 0, 1, 0, 1)$: $v = 4 + 0 + 1 + 0 + 1 = 6$. $R^2 = 1 - 4/6 = 1/3 \approx 0.333$.

(ii) It scores exactly $0.0$. The dummy predicts $\bar y = 4$ for every point, so its residual sum of squares *is* $v = 6$ (each residual is $y_i - \bar y$), and $R^2 = 1 - v/v = 0$. That is the §37.5 landmark: the constant model sits at $R^2 = 0$ by construction — $v$ is literally "how far the labels miss their own mean."

## Problem 3 — Pick the metric

(i) `median_absolute_error` — the median shrugs at the misrecorded mansions ("robust to outliers"); MSE would square them and let them dominate.

(ii) `max_error` — the worst-case metric, by definition the single largest $|y - \hat y|$.

(iii) `mean_squared_log_error` — the slide's pick for "targets with exponential growths"; it also "penalizes under-estimation heavier than the over-estimation," which is the right asymmetry when missing growth hurts more than overshooting it.

(iv) `mean_absolute_percentage_error` — "sensitive to relative error": $|y - \hat y|/|y|$ averaged, i.e. "typically off by $x$\%".

## Problem 4 — Course-sourced

(i) $[x_1, x_2] \to [1, x_1, x_2, x_1^2, x_1x_2, x_2^2]$: row $[0,1] \to [1, 0, 1, 0, 0, 1]$; row $[2,3] \to [1, 2, 3, 4, 6, 9]$; row $[4,5] \to [1, 4, 5, 16, 20, 25]$ — matches the deck's recorded output exactly.

(ii) `interaction_only=True` drops the pure powers: $[x_1, x_2] \to [1, x_1, x_2, x_1x_2]$: $[1, 0, 1, 0]$, $[1, 2, 3, 6]$, $[1, 4, 5, 20]$.

(iii) §23.8's $\phi(x) = (1, x, x^2, \dots)^T$ generalized: full variant $\phi(x_1, x_2) = (1, x_1, x_2, x_1^2, x_1x_2, x_2^2)^T$; interaction-only $\phi(x_1, x_2) = (1, x_1, x_2, x_1x_2)^T$. Both are "linear in $\theta$" models (§23.8's Note) — the curvature lives in $\phi$, not in the fit.

## Problem 5 — Score or error?

(i) `cross_val_score` (and `GridSearchCV`) always *maximize* the scoring value. An error is "lower is better," so maximizing raw MSE would pick the *worst* model; the `neg_` prefix flips it into a score where higher is better, and the argmax then does the right thing.

(ii) MSE $= 16.04$ — negate back.

(iii) $R^2$ is already a score (higher is better: $1.0$ best, $0.0$ baseline, negative bad), so negating it would invert the ordering — `scoring='r2'` is used as-is.

## Problem 6 — Spot the leak

(i) It violates the never-learn-on-full-data discipline (§36.7's imputation note, §37.8's Note): `fit_transform(X)` learns the column medians from *all* rows, including the ones that will become the test set. Test information (the medians) leaks into the training data — the test set is no longer unseen.

(ii)
```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('regressor', LinearRegression())])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe.fit(X_train, y_train)   # medians learned on train only
pipe.score(X_test, y_test)   # test rows only ever see transform
```
The split happens on the raw $X$ (NaNs and all); the imputer is fitted inside `pipe.fit` on the training rows alone. The leak is now unwritable — there is no call sequence that fits the imputer on the test rows.

## Problem 7 — The `alpha` translation

(i) The book minimizes $J(w) = \tfrac{1}{2}\lVert Xw - y\rVert^2 + \tfrac{\lambda}{2}\lVert w\rVert^2$; sklearn minimizes $J_{\text{sk}}(w) = \lVert Xw - y\rVert^2 + \alpha\lVert w\rVert^2$. But $J(w) = \tfrac{1}{2}J_{\text{sk}}(w)$ when $\alpha = \lambda$ — multiplying an objective by the positive constant $\tfrac{1}{2}$ does not move its minimizer. Same argmin, so book-$\lambda$ = sklearn-`alpha` exactly (for ridge).

(ii) sklearn's `Lasso` minimizes $\tfrac{1}{2n}\lVert y - Xw\rVert^2 + \alpha\lVert w\rVert_1$ — the squared error is *averaged over the $n$ samples* before the penalty is added. The book's §28.7 objective, $\tfrac{1}{2}\lVert Xw - y\rVert^2 + \tfrac{\lambda}{2}\lVert w\rVert_1$, has no $1/n$. Dividing the book's objective by $n$ shows the two coincide only when $\alpha = \lambda/(2n)$ — so the same numeric dial is a $2n$-times-stronger penalty in the book's formulation. You cannot copy the number across; re-tune it.

## Problem 8 — Diagnose the fit

(i) Overfitting, textbook (§23.9): training error $0.000$ — the degree-9 curve memorizes the ten points — while test error $0.226$ is an order of magnitude worse, and $\lVert w\rVert_2 = 84442$ is the §28.1 signature of fitting noise with huge canceling weights.

(ii) Fix 1: shrink the hypothesis class — `GridSearchCV` over `poly__degree` (tunes `degree`; §23.9's "choose a smaller $m$ by validation"). Fix 2: keep degree 9 but penalize the weights — `RidgeCV` over `ridge__alpha` (tunes `alpha`, the book's $\lambda$; §28.10's validation discipline). Either is legitimate; eg 5 shows Fix 2 dropping test MSE from $0.226$ to $0.044$.
