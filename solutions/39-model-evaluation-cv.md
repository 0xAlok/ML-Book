# Solutions — Chapter 39: Model evaluation, cross-validation, hyperparameter tuning

Worked solutions for the problem set in `chapters/39-model-evaluation-cv.md`. Numbers marked **[recorded]** are printed in the course material; **[verified]** are hand arithmetic from this session (numpy used as a calculator only — sklearn was never run).

---

**1. Split variance, by hand.** $y = (1, 2, 10, 11)$; model = training mean.

(i) Test MSEs:

- Split A, train $\{1, 2\}$: $\hat y = 1.5$. $\mathrm{MSE} = \frac{(10-1.5)^2 + (11-1.5)^2}{2} = \frac{72.25 + 90.25}{2} = \boxed{81.25}$.
- Split B, train $\{1, 10\}$: $\hat y = 5.5$. $\mathrm{MSE} = \frac{(2-5.5)^2 + (11-5.5)^2}{2} = \frac{12.25 + 30.25}{2} = \boxed{21.25}$.
- Split C, train $\{1, 11\}$: $\hat y = 6$. $\mathrm{MSE} = \frac{(2-6)^2 + (10-6)^2}{2} = \frac{16 + 16}{2} = \boxed{16.00}$. [verified]

(ii) Leave-one-out. The four hold-outs:

| held out $y_i$ | mean of other three | squared error |
|---|---|---|
| $1$ | $23/3$ | $(1 - 23/3)^2 = (20/3)^2 = 400/9$ |
| $2$ | $22/3$ | $(2 - 22/3)^2 = (16/3)^2 = 256/9$ |
| $10$ | $14/3$ | $(10 - 14/3)^2 = (16/3)^2 = 256/9$ |
| $11$ | $13/3$ | $(11 - 13/3)^2 = (20/3)^2 = 400/9$ |

$$\mathrm{MSE}_{\mathrm{LOO}} = \frac{1}{4}\cdot\frac{400 + 256 + 256 + 400}{9} = \frac{1312}{36} = \frac{328}{9} \approx \boxed{36.44}.$$ [verified]

(iii) Any single split's number is hostage to which rows fell into test — here $16$ vs $81$ for the same model. Averaging over many splits (LOO: $36.44$) washes out that luck; the slide's point is that the *average* is the honest estimate of generalization, not any one draw.

---

**2. Fold arithmetic.**

(i) `KFold(n_splits=3)`, $n = 6$, no shuffling → contiguous blocks: test sets $\{0, 1\}$, $\{2, 3\}$, $\{4, 5\}$. Each row is tested exactly once and trained on exactly twice. [verified]

(ii) `LeaveOneOut` = `KFold(n_splits=n)` = $6$ fits: test sets $\{0\}, \{1\}, \dots, \{5\}$, each fit training on the other five rows.

(iii) `[0 2 1 3]` = `np.append(train, test)` for the *final* loop iteration: rows $\{0, 2\}$ trained, rows $\{1, 3\}$ tested in that split. Only one array is returned because the course's function rebinds `train, test` on every iteration and calls `np.append` *after* the loop — the first three splits' indices are silently discarded. (The graded answer's option (a).) [recorded]

---

**3. Two functions, two returns.**

(i) An array of length $5$ holding the five per-fold *test* scores (default scoring for the estimator — $R^2$ for a regressor).

(ii) `fit_time`, `score_time`, `test_score` — always; `estimator` — only with `return_estimator=True`; `train_score` — only with `return_train_score=True`.

(iii) Mean of the neg-scores: $\frac{-16.04 - 18.90 - 15.02}{3} = \frac{-49.96}{3} \approx -16.6533$; negate for the error: $\boxed{\mathrm{MSE} \approx 16.65}$. [verified]

(iv) `cross_val_score` is the thin convenience wrapper: one metric in, one array out. `cross_validate` is the instrumented worker — it loops the folds once and can accumulate as many metrics per fold as you name.

---

**4. Course-sourced metric.** $y = [7, 4, 9, 4]$, $\hat y = [8, 7, 12, 5]$.

(i) Residuals $r = y - \hat y = [-1, -3, -3, -1]$, $\bar r = -2$: $\mathrm{Var}(r) = \frac{1^2 + (-1)^2 + (-1)^2 + 1^2}{4} = 1$. $\bar y = 24/4 = 6$: $\mathrm{Var}(y) = \frac{1^2 + (-2)^2 + 3^2 + (-2)^2}{4} = \frac{18}{4} = 4.5$.
$$\mathrm{EVS} = 1 - \frac{1}{4.5} = 1 - \frac{2}{9} = \boxed{\frac{7}{9} \approx 0.7778},$$
matching the recorded $0.7777777777777778$ to every shown digit. [verified]

(ii) Perfect predictor: every residual $0$ → $\mathrm{Var}(r) = 0$ → EVS $= 1$. Mean-predictor: $\hat y_i = \bar y$ → $r = y - \bar y$ → $\mathrm{Var}(r) = \mathrm{Var}(y)$ → EVS $= 0$. (This is why the slide's $R^2$ page gives the constant model $0.0$ — same floor.)

(iii) $R^2 = 1 - u/v$ uses the *raw* residual sum of squares; explained variance subtracts the residual mean first ($\mathrm{Var}(r) = \frac{1}{n}\sum r_i^2 - \bar r^{\,2}$). On the *test* set the residuals need not sum to zero (no intercept was fit to them), so the bias correction makes EVS a hair higher: $0.6605500501742703 > 0.6605140591531992$ [recorded].

---

**5. Read the curves.**

(i) **Underfit** (§23.9's $m = 1$ line): both errors high and no train/CV gap, so the model lacks capacity — more data will not move either curve. Fix direction: richer model (more features, weaker regularization), not more rows.

(ii) **Overfit with a closing gap**: train error $0.10 \ll$ CV error $0.45$ is §23.9's overfitting signature, but the CV curve is still falling at max $n$ — the gap is data-starved, not capacity-wrong. Buy *data* first; reach for regularization only if the gap plateaus open.

(iii) `train_size, train_scores, test_scores = results[:3]`; `train_errors, test_errors = -train_scores, -test_scores` (the negation undoes the `neg_` scoring). With $5$ training sizes and $4$ folds, `train_scores` (and `test_scores`) have shape $\boxed{(5, 4)}$.

---

**6. Grid vs randomized.**

(i) Verbatim [recorded]:
```
UserWarning: The total space of parameters 8 is smaller than n_iter=10.
Running 8 iterations. For exhaustive searches, use GridSearchCV.
```
sklearn ignored the `n_iter=10` request and ran all $8$ combinations ($2$ `fit_intercept` $\times$ $4$ `alpha`) — random search degenerated into exhaustive grid search, and the warning says to use `GridSearchCV` for that.

(ii) $4^3 = 64$ combinations $\times$ $5$ folds $= \boxed{320}$ fits. [verified]

(iii) Randomized search wins when the space is large or continuous — the budget (`n_iter`) stays fixed while a grid explodes exponentially. Grid search wins when the space is small and discrete enough to enumerate — then it *is* the exhaustive answer, and random sampling can only miss combinations.

---

**7. The three-way doctrine.**

(i) 1. Split into train/validation/test. 2. Train every hyperparameter combination on train (`n_jobs=-1` to parallelize; `error_score=0`/`NaN` so one bad combination doesn't crash the search). 3. Score each on validation; pick the best. 4. Retrain the winner on train+validation combined. 5. Evaluate once on test.

(ii) The validation rows' job was *selection*; once the hyperparameters are frozen, there is no reason to withhold those rows from *parameter* learning. Retraining on train+validation gives the final model strictly more data at zero selection cost — the choice is already made.

(iii) It honors **"spend the test set exactly once"** (§37.9): the test split was never touched during the search, and this single `.score` call is its one and only use. The STEP-4 retraining was done by `GridSearchCV`'s default `refit=True`, which refits the best combination on the full training data before returning.

---

**8. Spot the leak.**

(i) The learn-on-train-only discipline (§36.7's imputation note, §37.8's transformer contract, §38.10's PCA story). `scaler.fit_transform(X_train)` fixes $\mu, \sigma$ *before* the search, so inside the CV loop every fold's training rows are scaled with statistics that saw the *other* folds' rows — the fold boundary is porous. (Mild here, since it's still train-only data; fatal in §38.10's version, which fit on the test set.)

(ii) Move the scaler inside the searched object so each fold refits it on its own training rows; address the estimator's dial through the step name:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline(steps=[('scale', StandardScaler()),
                           ('sgd', SGDRegressor(random_state=1))])
search = GridSearchCV(pipeline,
                      {'sgd__alpha': [0.1, 0.01, 0.001]},
                      cv=4)
search.fit(X_train, y_train)
```
`sgd__alpha` is the double-underscore addressing (§37.8): step `sgd`, hyperparameter `alpha`.

---

**9. Refit semantics.**

(i) `refit=True`: average each candidate's scores across folds, take the hyperparameter values behind the best average score, and **refit once** with them — one winning model. `refit=False`: **average the coefs, intercepts, and $C$** of the per-fold winners instead — the answer is a blend, with no final single fit. (The slide's words, §38.5.)

(ii) `alphas` is the **constructor's** candidate list — the values you ask it to try. `alpha_` is the **fitted attribute** — the value cross-validation actually chose. The slide names the input side; §37.7 names the output side. Both are right about different objects.

(iii) `GridSearchCV` over a `Pipeline` of `PolynomialFeatures` + `Ridge` — e.g. `param_grid = [{'poly__degree': [2, 3, 4, 5, 6, 7, 8, 9], 'ridge__alpha': [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]}]`. `RidgeCV` alone is insufficient because its built-in search covers only the regularization strength — it cannot vary the polynomial degree (or any second dial); the slide sanctions exactly this "grid over the pipeline" shape for multi-dial tuning.
