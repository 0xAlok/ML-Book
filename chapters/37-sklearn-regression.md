# Chapter 37: scikit-learn: regression workflows

Everything in §§37.1–37.7 comes from the MLP course's regression weeks — the Week-3 "Linear Regression" and Week-4 "Polynomial Regression" slide decks (Dr. Ashish Tendulkar, IIT Madras) — and §§37.8–37.9 from the Week-2 "Data Preprocessing" slides, the Week-2 programming-questions notebook, and the "Notes by Sejal" Week-2 summary (pipeline construction, `set_params`, grid search with pipelines). (The MLT TA colab's "Week 2 Programming Assignment" was checked — it is about kernel matrices, not sklearn workflows, and is not used.) This is the code for Chapter 23's equations: §23.5 derived $\hat\theta = (X^TX)^{-1}X^Ty$ on paper; here something calls `.fit()` and returns it.

**A word on what was actually run.** sklearn is not installed on this machine (see the chapter task note), so no `sklearn` call below was executed here. The code blocks are transcribed from the course slides and notebooks — verbatim unless a slide typo forced a fix (each fix is logged in `reviews/37-sklearn-regression.md`). Every *number* beside the code is either **[recorded]** — printed as output in the course material itself — or **[verified-NumPy]** — computed in this session through the mathematically identical closed form (§23.5's normal equations, §28.4's ridge form, the scaler/imputer definitions the slides give), which is provably what the sklearn call computes. The two kinds are marked wherever they could be confused.

**Notation.** The book's standing convention (chapters 31–36): $X \in \mathbb{R}^{n \times d}$ holds the points $x^{(1)}, \dots, x^{(n)}$ as rows, $y \in \mathbb{R}^n$ holds the labels. sklearn uses the same convention — its Week-12 slide writes it out: training data shape $\to$ `(n_samples, n_features)`, labels $\to$ `(n_samples,)`.

## 37.1 The estimator contract: `fit`, `predict`, `score`

**Def (estimator).** An **estimator** = any sklearn object that learns parameters from data. Every estimator honors the same three-method contract:

i) `fit(X_train, y_train)` — learn the parameters from the training data. For regression, this is §23.5's job: find $\hat\theta$.
ii) `predict(X_test)` — apply the learned parameters to new data: $\hat y = X_{\text{test}}\hat\theta$ (§23.1's model, row by row).
iii) `score(X_test, y_test)` — evaluate on held-out data (§37.5: for regressors this returns $R^2$).

The Week-3 slide's remark, kept because it is the whole point of the chapter: *"These code snippets work for both LinearRegression and SGDRegressor, and for that matter to all regression estimators that we will study in this module. Why? All of them are estimators."* Learn the contract once; every regressor from here on is a new object with the same three verbs.

**eg 1 (the contract, on the §36.8 house data).** Features = area, rooms; label = price. The 10-row table of §36.8, rooms imputed with the median $4.0$:

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()      # i)   choose the estimator
lin_reg.fit(X_train, y_train)     # ii)  learn:  theta-hat = (X^T X)^{-1} X^T y
lin_reg.predict(X_test)           # iii) apply:  y-hat = X_test @ theta-hat
lin_reg.score(X_test, y_test)     # iv)  evaluate: R^2 on the test set
```

Replace `LinearRegression()` with `Ridge(alpha=1e-3)` or `SGDRegressor()` and nothing else changes — the contract is the API.

**Basically, ...** "An sklearn estimator is a machine with three buttons: `fit` learns the weights, `predict` uses them, `score` grades them. `LinearRegression`, `Ridge`, `Lasso`, `SGDRegressor` are all the same machine with different engines inside — which is why the Week-3 slide keeps reusing one code shape for all of them."

## 37.2 The workflow: split first, then fit

The slide's four steps, in order — and the order is load-bearing:

i) **Split** the data into train and test *before anything learns anything*:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```
`random_state=42` fixes the shuffle, the same reproducibility discipline as §36.6's seed (the SGD slide repeats it: "It's a good idea to use a random seed of your choice... it helps us get reproducible results").
ii) **Fit** the estimator on the *training* set only.
iii) **Training error** (a.k.a. *empirical error*) — how the model does on data it saw.
iv) **Test error** (a.k.a. *generalization error*) — how it does on data it didn't. Compare the two: close together is healthy; train tiny + test large is §23.9's overfitting signature.

**Note (the baseline habit).** Before any real model, the slide builds a `DummyRegressor`:
```python
from sklearn.dummy import DummyRegressor
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
dummy_regr.predict(X_test)
dummy_regr.score(X_test, y_test)
```
It predicts the training-set mean (strategies: `mean`, `median`, `quantile`, `constant`) for every input — the "constant model" of §37.5. Any model that cannot beat this on the test set is not a model, it is a rounding error with extra steps.

**Note (the discipline, again).** The imputation rule of §36.7 — learn the imputer on the training set, apply it everywhere — generalizes: *every* learning step (imputer, scaler, polynomial expansion, the regressor itself) is fitted on the training split and merely applied to the test split. §37.8 shows the machinery (`fit` vs `transform`) that enforces this; Chapter 39 makes it a full evaluation doctrine.

## 37.3 Two engines for the same line

The slide offers two estimators for linear regression — same contract, different internals:

**Option A — the normal equation.** `LinearRegression()` solves §23.5's closed form. The appendix is explicit: it uses an SVD-based solve, cost $O(m^2)$ in the number of features $m$ — "if we double the number of features, the training computation grows roughly 4 times" — and it is linear in $n$ "as long as the training set fits in the memory." Same optimum §23.3 proves, computed directly.

**Option B — iterative optimization.** `SGDRegressor()` walks downhill the way Chapter 10 does, one sample (or mini-batch) at a time: cost $O(knp)$ for $k$ epochs and $p$ average nonzero features per sample — linear in $n$, which is why the slide says *"Use for large training set up (> 10k samples)"*, and adds: *"For learning problems with small number of training examples, sklearn user guide recommends Ridge or Lasso."*

**Note (SGD's one demand: scale your features).** "SGD is sensitive to feature scaling, so it is highly recommended to scale input feature matrix." The slide's fix is a two-step pipeline (full pipeline machinery in §37.8):
```python
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
sgd = Pipeline([
    ('feature_scaling', StandardScaler()),
    ('sgd_regressor', SGDRegressor())])
sgd.fit(X_train, y_train)
```
(Exceptions the slide carves out: word frequencies and indicator features "have intrinsic scale" and skip this.)

The SGD knobs, as the slides record them (defaults per the slides — `random_state` is deliberately set in real code even though the slides omit it "for sake of brevity"):
i) `random_state=42` — reproducibility (§36.6's whole sermon, in one argument).
ii) `max_iter=100` — one epoch is one full pass over the training data; default is 1000. The slide's rule of thumb: "SGD converges after observing approximately $10^6$ training samples," so a first guess is `max_iter = np.ceil(1e6 / n)`.
iii) `learning_rate` — `'invscaling'` (the slide's stated default), `'constant'`, or `'adaptive'`; `eta0` sets the starting rate. Invscaling decays as $\eta(t) = \eta_0 / t^{\text{power\_t}}$.
iv) Stopping — either watch the training loss (`tol`, `n_iter_no_change`: stop when the loss fails to improve by `tol` for `n_iter_no_change` epochs) or hold out a validation fraction and watch the *validation score* (`early_stopping=True, validation_fraction=0.2`).
v) `loss='squared_error'` — "studied in this course"; `'huber'` is the robust alternative. `average=True` switches on averaged SGD (the returned `coef_` is the average of the iterates, which "works best with a larger number of features and a higher eta0").
vi) `warm_start=True` — resume from the previous run's weights instead of re-initializing; the slide's monitoring recipe fits one epoch at a time in a loop and evaluates the validation MSE after each.

**Basically, ...** "`LinearRegression` does the algebra (§23.5) in one shot — exact, but the matrix inverse gets expensive as features grow. `SGDRegressor` does Chapter 10's hill-walking — cheap per step, scales to millions of rows, but needs scaled features, a seed, and a few knobs set. Same line out the other end."

## 37.4 Model inspection: reading the weights back out

After `fit`, the learned parameters live in two attributes — for *every* linear estimator in this chapter:

```python
linear_regressor.coef_       # w1, ..., wm  (the weights)
linear_regressor.intercept_  # w0          (the bias term)
```

The model is §23.1's, written the slide's way: $\hat y = w_0 + w_1x_1 + w_2x_2 + \cdots + w_mx_m = w^Tx$. And inference is one line on a properly shaped matrix — "Step 1: arrange data for prediction in a feature matrix of shape (#samples, #features)" — the $(n, d)$ convention doing its job:

```python
# Predict labels for feature matrix X_test
linear_regressor.predict(X_test)
```

**eg 2 (the §36.8 houses, inspected) [verified-NumPy].** OLS on all 10 rows (rooms NaN $\to$ median $4.0$, per §36.8), solved through §23.5's normal equations — the identical computation `LinearRegression().fit(X, y)` performs:

```python
lin_reg.coef_       # [0.0643, 1.7601]   (area, rooms)
lin_reg.intercept_  # -14.805
```

Read it as the book reads equations: each extra square foot adds $0.0643$ to the price, each extra room adds $1.7601$, from a base of $-14.805$. A $1000$-sqft, $3$-room house: $\hat y = -14.805 + 0.0643 \times 1000 + 1.7601 \times 3 = 54.8$. (Units are §36.8's; the arithmetic is what matters — `coef_` *is* §23.5's $\hat\theta$, `intercept_` its $\theta_0$.)
## 37.5 Evaluation: scores, errors, and $R^2$

**Def (score vs error).** The slide draws the line once, and it never moves: a **score** is a metric where *higher is better*; an **error** is one where *lower is better*. sklearn converts any error into a score by negating it — the `neg_` prefix: `neg_mean_squared_error`, `neg_mean_absolute_error`, `neg_root_mean_squared_error`, and so on (needed because cross-validation and grid search always *maximize*). The slide's table:

| error function | scoring name |
|---|---|
| `mean_absolute_error` | `neg_mean_absolute_error` |
| `mean_squared_error` | `neg_mean_squared_error` / `neg_root_mean_squared_error` |
| `mean_squared_log_error` | `neg_mean_squared_log_error` |
| `median_absolute_error` | `neg_median_absolute_error` |

**Def ($R^2$, the coefficient of determination).** `estimator.score(X_test, y_test)` returns
$$\boxed{R^2 = 1 - \frac{u}{v}}, \qquad u = (Xw - y)^T(Xw - y), \quad v = (y - \hat y_{\text{mean}})^T(y - \hat y_{\text{mean}}).$$
$u$ = residual sum of squares (how far the model's predictions miss); $v$ = total sum of squares (how far the labels miss their *own mean*). So $R^2$ = "what fraction of the label's variance the model explains." Three landmarks, straight from the slides:
i) Best possible: $1.0$ — $u = 0$, the model nails every point.
ii) $0.0$ — $u = v$: the model is exactly as good as always predicting the mean. (This is why §37.2's `DummyRegressor(strategy="mean")` is the baseline to beat.)
iii) Negative — "the model can be arbitrarily worse" than predicting the mean. A negative $R^2$ is the model telling on itself.

**eg 3 ($R^2$ by the numbers, on eg 2's fit) [verified-NumPy].** Training residuals $u = 160.4$, label variance $v = 4214.1$: $R^2 = 1 - 160.4/4214.1 = 0.9619$. Mean squared error $= u/n = 16.04$. Ninety-six percent of the price variance is explained by area and rooms — on the data it was fitted to, so read it as a goodness-of-fit, not a generalization claim (§37.2's step iv is what earns that).

The `sklearn.metrics` shelf, with the slide's one-line character for each:
i) `mean_absolute_error(y_test, y_predicted)` — average $|y - \hat y|$, in the label's own units.
ii) `mean_squared_error` — the slide's appendix lists MSE and RMSE as *the* evaluation measures; note the slide text once misspells the import as `mean_squarred_error` — the real name is `mean_squared_error` (fix logged in the review).
iii) `r2_score` — "same as output of score."
iv) `mean_squared_log_error` — "useful for targets with exponential growths like population, sales growth"; "penalizes under-estimation heavier than the over-estimation."
v) `mean_absolute_percentage_error` — "sensitive to relative error."
vi) `median_absolute_error` — "robust to outliers" (the median shrugs at the one wild point).
vii) `max_error` — the worst case, "only for single output regression."

**Basically, ...** "$R^2$ answers 'how much better than just guessing the average?' — $1$ is perfect, $0$ is the average-guesser, negative is worse than guessing. Everything else on the shelf is a different way of averaging the misses: squared (MSE) punishes big misses hardest, absolute (MAE) reads in plain units, median (MedAE) ignores outliers, max tells you the worst single embarrassment."

## 37.6 Polynomial regression: expand the features, keep the machine

§23.8's move, as code: polynomial regression = polynomial *transformation* + linear regression. "The only difference with respect to regular linear regression is that we are transforming the features and then performing regression" — §23.8's lecture line, which the MLP slide re-implements:

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
# Two steps:
# 1. Polynomial features of desired degree (here degree=2)
# 2. Linear regression
poly_model = Pipeline([
    ('polynomial_transform', PolynomialFeatures(degree=2)),
    ('linear_regression', LinearRegression())])
# Train with feature matrix X_train and label vector y_train
poly_model.fit(X_train, y_train)
```

Swap the engine to SGD — "notice that there is a single line code change in two code snippets" (the slide's words):
```python
poly_model = Pipeline([
    ('polynomial_transform', PolynomialFeatures(degree=2)),
    ('sgd_regression', SGDRegressor())])
poly_model.fit(X_train, y_train)
```

**eg 4 (what `PolynomialFeatures` actually builds) [recorded].** The slide's example, output printed in the deck:
```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
X = np.arange(6).reshape(3, 2)
poly = PolynomialFeatures(degree=2)
poly.fit_transform(X)
# Data matrix:
# [[0 1]
#  [2 3]
#  [4 5]]
# After transformation:
# [[ 1.  0.  1.  0.  0.  1.]
#  [ 1.  2.  3.  4.  6.  9.]
#  [ 1.  4.  5. 16. 20. 25.]]
```
$[x_1, x_2] \to [1, x_1, x_2, x_1^2, x_1x_2, x_2^2]$ — exactly §23.8's $\phi(x) = (1, x, x^2, \dots)^T$, now with the interaction term $x_1x_2$. Two variants: `interaction_only=True` gives $[1, x_1, x_2, x_1x_2]$ ("$[x_1^2, x_2^2]$ are excluded" — the slide's note); `include_bias=False` drops the leading $1$ when the estimator already fits its own intercept.

**Note (capacity bites back).** The slide's appendix warns: "Since the polynomial regression uses more parameters (due to polynomial representation of the input), it is more prone to overfitting. We will study how to detect overfitting with learning curves and use of regularization to mitigate the risk of overfitting." §23.9 said it on paper; here is the code version.

**eg 5 (overfitting, caught red-handed) [verified-NumPy].** Ten noisy points from the smooth curve $f(x) = 4x^3 - 6x^2 + 3x$ on $[0, 1]$ (seed $0$; noise $\sigma = 0.15$), judged on $500$ fresh points. Each row is the sklearn pipeline shown above, its numbers verified through the identical closed form:

| model | $\lVert w\rVert_2$ | train MSE | test MSE |
|---|---|---|---|
| linear (degree 1) | $0.85$ | $0.027$ | $0.048$ |
| poly, degree 9 | $84441.82$ | $0.000$ | $0.226$ |
| poly 9 + Ridge($\alpha=0.5$) | $0.38$ | $0.031$ | $0.044$ |
| poly 9 + Ridge($\alpha=50$) | $0.03$ | $0.088$ | $0.056$ |

Read it the §23.9 way: degree 9 memorizes the ten training points (train MSE $0.000$) with absurd weights ($\lVert w\rVert_2 = 84442$ — the wiggle between points from §23.9's figure, now as a number) and pays on fresh data. Ridge with $\alpha = 0.5$ shrinks the weights two-hundred-thousand-fold and wins on the test set; $\alpha = 50$ squeezes too hard and underfits. This is §28.6's "$\lambda$ dial" with the dial labeled `alpha` — §37.7 names the translation.

**Basically, ...** "A degree-9 polynomial on 10 points is a student who memorized the practice test: $0.000$ training error, nonsense elsewhere, weights in the tens of thousands. Regularization is the teacher confiscating the cheat sheet — smaller weights, slightly worse practice score, much better real score."

## 37.7 Regularization in sklearn: the `alpha` dial

Chapter 28 derived ridge and lasso with the penalty dial called $\lambda$. sklearn calls it **`alpha`** — same role, and for ridge the translation is exact:

**Note (the `alpha` $\leftrightarrow$ $\lambda$ convention).** The book's ridge (§28.3) minimizes $\tfrac{1}{2}\lVert Xw - y\rVert^2 + \tfrac{\lambda}{2}\lVert w\rVert^2$; sklearn's `Ridge` minimizes $\lVert Xw - y\rVert^2 + \alpha\lVert w\rVert^2$. Multiply the book's objective by 2 and the two are the same problem — so **book-$\lambda$ = sklearn-`alpha`, exactly**, for ridge. (For lasso the objectives differ in normalization — sklearn's `Lasso` divides the squared error by $2n$ — so treat its `alpha` as its own dial, not a port of the book's $\lambda$.)

```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1e-3)
# fit, score, predict work exactly like other linear regression estimators
```
or, equivalently, the SGD route (`penalty='l2'` is even SGDRegressor's default):
```python
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(alpha=1e-3, penalty='l2')
```

Lasso, the same shape (§28.7's feature-selector):
```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1e-3)
# or: SGDRegressor(alpha=1e-3, penalty='l1')
```
Both at once — elastic net, "a convex combination of L1 (Lasso) and L2 (Ridge)": `SGDRegressor(penalty='elasticnet', l1_ratio=0.3)` means $0.3 \times \text{L1} + 0.7 \times \text{L2}$ ("L2 takes higher weightage in this formulation" — the slide's words).

Regularization meets the polynomial pipeline — the slide's recipe for §37.6's overfit:
```python
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
poly_model = Pipeline([
    ('polynomial_transform', PolynomialFeatures(degree=2)),
    ('ridge', Ridge(alpha=1e-3))])
poly_model.fit(X_train, y_train)
```
(Replace `Ridge` with `Lasso` for the lasso version; the slide gives both.)

**Choosing `alpha`.** Two course-sanctioned routes (§28.10's validation discipline, as code): the built-in path — `RidgeCV` / `LassoCV` / `ElasticNetCV`, which "fit data for a range of values of some parameter almost as efficiently as fitting the estimator for a single value" (`RidgeCV(alphas=[...])` tries a list; the winner is reported in its `alpha_` attribute) — or plain cross-validation over a `Ridge`/`Lasso` grid. The full cross-validation machinery is Chapter 39's subject; the one-line version lives here because the dial is useless without a way to set it.

**Basically, ...** "sklearn's `alpha` *is* the book's $\lambda$ wearing a different name tag (exactly, for ridge). `RidgeCV` is the lazy-good way to pick it: hand it a list of candidates, it cross-validates each, and tells you the winner. The polynomial that memorized your data in eg 5 gets fixed by exactly this."

## 37.8 Transformers and pipelines: learn on train, apply everywhere

The Week-2 preprocessing slides give the second contract of the chapter — the **transformer** contract:

**Def (transformer).** `fit(X)` *learns* parameters from data; `transform(X)` *applies* the learned transformation to (new) data; `fit_transform(X)` does both at once. The slide's discipline, verbatim in spirit: the same preprocessing must be applied to train and test, so fit on the training set and transform both.

i) **Scaling.** `StandardScaler` learns $\mu, \sigma$ per feature and maps $x' = (x - \mu)/\sigma$ [recorded]:
```python
ss = StandardScaler()          # (the slide prints StandardScalar(); the class is StandardScaler)
x = [[4],[3],[2],[5],[6]]      # mu = 4, sigma = sqrt(2)
ss.fit_transform(x)            # [[0],[-1/sqrt(2)],[-2/sqrt(2)],[1/sqrt(2)],[2/sqrt(2)]]
```
transformed mean $0$, std $1$. Siblings: `MinMaxScaler` ($x' = (x - x_{\min})/(x_{\max} - x_{\min})$, range $[0,1]$), `MaxAbsScaler` (range $[-1,1]$).
ii) **Imputation.** `SimpleImputer(strategy='mean'|'median'|'most_frequent'|'constant')` [recorded]:
```python
si = SimpleImputer(strategy='mean')
# X = [[7,1],[nan,8],[2,nan],[9,6]]  ->  col means 6 and 5
si.fit_transform(X)   # [[7,1],[6,8],[2,5],[9,6]]
```
`KNNImputer(n_neighbors=k)` fills each gap with the mean of the $k$ nearest rows by Euclidean distance (the slide works one by hand: $[1, 2, \text{nan}]$'s two nearest rows give $(3+5)/2 = 4$); `MissingIndicator` adds the binary "was missing" columns (§36.7's discipline: *recorded-but-absent* $\to$ impute, and learn the statistics on train only).

**Def (pipeline).** A `Pipeline` chains transformers and ends in an estimator: `fit` calls `fit_transform` down the chain and `fit` on the last step; `predict`/`score` transform-then-predict. The Week-2 programming notebook's real pipeline (blood-donation data, 5 features; verbatim, lightly trimmed):
```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

num_pipe = Pipeline([("colselect", ColumnTransformer([("select", "passthrough", [0,1,2,3])])),
                     ("impute", SimpleImputer()),
                     ("scale", StandardScaler())])
cat_pipe = ColumnTransformer([("encode", OrdinalEncoder(), [4])])
pipe = FeatureUnion([("numerical", num_pipe),
                     ("categorical", cat_pipe)])
```
`ColumnTransformer` applies *different* transformers to *different* columns and concatenates; `FeatureUnion` runs transformers in parallel and glues the outputs side by side. Steps are reached by name — `pipe.set_params(pca__n_components=2)` — the double underscore addressing that §37.9's grid search relies on.

**Note (the leak this machinery exists to prevent).** `fit_transform(X_all)` before the split learns the scaler's $\mu, \sigma$ (or the imputer's medians) from the test points too — test information leaking into training statistics. The pipeline fixes it structurally: `pipe.fit(X_train)` then `pipe.predict(X_test)` — the test set only ever sees `transform`. This is §36.7's imputation note grown into an architecture.

**eg 6 (the full workflow, one object).** Impute $\to$ scale $\to$ polynomial $\to$ ridge, on the §36.8 houses (the NaN in row 9's rooms is real this time), fitted on the §36.6 split (seed 42, `test_size=0.2`: test = ids 9 and 2 — "the same two, every run"):
```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('impute',  SimpleImputer(strategy='median')),   # learns medians on TRAIN only
    ('scale',   StandardScaler()),                    # learns mu, sigma on TRAIN only
    ('poly',    PolynomialFeatures(degree=2)),        # [1, x1, x2, x1^2, x1*x2, x2^2]
    ('ridge',   Ridge(alpha=1e-3))])                   # book-lambda = 1e-3
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)      # R^2 on the held-out rows
```
What each step learned on the 8 training rows [verified-NumPy]: rooms median $4.0$ (area median $1250.0$, unused — no NaNs there); scaler $\mu = (1243.75,\ 4.125)$, $\sigma = (236.43,\ 0.78)$; the first training row's polynomial expansion $[1,\ -0.61,\ -0.16,\ 0.37,\ 0.10,\ 0.03]$. None of these numbers saw the test rows — that is the entire point of the object.

**Basically, ...** "A pipeline is the §37.2 workflow with the discipline built in: `fit` learns every step's parameters from the training data in order, `predict` replays the same transformations on new data. `ColumnTransformer` says 'different columns, different treatment'; `FeatureUnion` says 'run these side by side and glue.' Write the workflow once, and the leak the book keeps warning about becomes un-writable."

## 37.9 Choosing the dial: validation, briefly

Two recurring questions now have code answers; both get their full treatment in Chapter 39, and both already appeared on paper in §§23.9 and 28.10:

i) **Which degree?** `GridSearchCV` over the pipeline's own parameter — the double underscore reaches inside the named steps (the slide's code, typo `POlynomialFeatures` fixed):
```python
from sklearn.model_selection import GridSearchCV
param_grid = [{'poly__degree': [2, 3, 4, 5, 6, 7, 8, 9]}]
pipeline = Pipeline(steps=[('poly', PolynomialFeatures()),
                           ('sgd', SGDRegressor())])
grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(X_train, y_train)
```
ii) **Which `alpha`?** `RidgeCV`/`LassoCV` (built-in, §37.7), or the same grid search with `'ridge__alpha'`.
iii) **Am I over- or underfitting?** `learning_curve` plots training vs cross-validated error against training-set size — the code version of §23.9's train/test comparison: both errors high means underfit, a gap that won't close means overfit.

**Note (the HPT data split).** The slide is explicit about the three-way split this machinery assumes: tune on train/validation, retrain the winner on train+validation, and report *once* on the test set — "note that the test set was not used in hyper-parameter search and model retraining." The test set is spent exactly once.

## 37.10 Where this goes next

i) **Classification, same contract.** Chapter 38 replays this chapter with labels instead of numbers: the Week-2 notebook already used `LogisticRegression()` inside `RFE`/`SequentialFeatureSelector` — `fit`/`predict`/`score` unchanged, only the estimators and the metrics change.
ii) **The dial-setting machinery, properly.** Chapter 39 takes §37.9's sketch — cross-validation iterators (`KFold`, `ShuffleSplit`, `LeaveOneOut`), `cross_val_score`/`cross_validate`, grid vs randomized search — and makes it a subject.
iii) **The neural turn.** The Week-12 slide's `MLPRegressor` "trains using backpropagation... uses the square error as the loss function, and the output is a set of continuous values" — a regressor honoring the same `fit`/`predict` contract, whose insides are Chapter 41's neurons. The contract outlives linear models.
iv) **Debugging the workflow.** Chapter 40: when `score` is negative and the pipeline looks right, the discipline for finding out why.

## Problem set

1. **The contract.** The Week-3 slide claims the `fit`/`predict`/`score`/`coef_`/`intercept_` snippets "work for both LinearRegression and SGDRegressor, and for that matter to all regression estimators that we will study in this module." (i) In one line each, say what `fit`, `predict`, and `score` compute for a regressor, in §23.1's notation ($\hat\theta$, $\hat y = X\hat\theta$). (ii) Which of the three methods does a *transformer* (e.g. `StandardScaler`) not have, and which does it have instead?
2. **$R^2$ by hand.** $y = (2, 4, 5, 4, 5)$, predictions $\hat y = (1, 5, 4, 4, 6)$. (i) Compute $u$ and $v$ and hence $R^2$. (ii) What would a `DummyRegressor(strategy="mean")` score on this data, and why — answer from the definition of $v$?
3. **Pick the metric.** For each situation, name the slide's metric and say why in one line: (i) house prices with a few misrecorded mansions in the data; (ii) a model whose contract penalizes the single worst prediction; (iii) forecasting user counts that grow exponentially; (iv) reporting error as "typically off by $x$\%".
4. **Course-sourced.** The slide's `PolynomialFeatures` example (eg 4) transforms `[[0,1],[2,3],[4,5]]` with `degree=2`. (i) Write the transformed matrix by hand and check against the recorded output. (ii) Repeat with `interaction_only=True`. (iii) In §23.8's notation, what is $\phi(x)$ for each variant?
5. **Score or error?** `cross_val_score` is maximizing something. (i) Why does sklearn need the `neg_` versions of error metrics at all? (ii) If `cross_val_score(..., scoring='neg_mean_squared_error')` returns $-16.04$, what is the MSE? (iii) Why is there no `neg_r2_score`?
6. **Spot the leak.** A classmate writes:
   ```python
   X_imp = SimpleImputer(strategy='median').fit_transform(X)   # X has NaNs
   X_train, X_test, y_train, y_test = train_test_split(X_imp, y, random_state=42)
   ```
   (i) Name the discipline this violates (§36.7 / §37.8's Note) and say what leaks where. (ii) Rewrite it as a `Pipeline` so the leak is structurally impossible.
7. **The `alpha` translation.** The book's ridge (§28.3) minimizes $\tfrac{1}{2}\lVert Xw-y\rVert^2 + \tfrac{\lambda}{2}\lVert w\rVert^2$; sklearn's `Ridge` minimizes $\lVert Xw-y\rVert^2 + \alpha\lVert w\rVert^2$. (i) Show the two minimizers coincide when $\alpha = \lambda$. (ii) Why can't you port a tuned `alpha` from sklearn's `Lasso` straight into the book's §28.7 formula as $\lambda$?
8. **Diagnose the fit.** A degree-9 polynomial pipeline reports train MSE $0.000$, test MSE $0.226$, $\lVert w\rVert_2 = 84442$ (eg 5's numbers). (i) Diagnose in §23.9's vocabulary. (ii) Name two fixes from this chapter and say which hyperparameter each one tunes (`degree` via `GridSearchCV`, or `alpha` via `RidgeCV`).
