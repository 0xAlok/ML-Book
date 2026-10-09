# Chapter 40 solutions: End-to-end model training and debugging

Full worked solutions for the problem set in `chapters/40-end-to-end-debugging.md`. Numbers marked **[recorded]** are printed as output in the course material itself; **[verified]** are worked by hand in this session; **[verified-NumPy]** were re-run with numpy 1.26.4.

## 1. The eight steps

(i) The deck's eight steps, in order: 1. Look at the big picture. 2. Get the data. 3. Discover and visualize the data to gain insights. 4. Prepare the data for ML algorithms. 5. Select a model and train it. 6. Fine-tune your model. 7. Present your solution. 8. Launch, monitor and maintain your system. [recorded]

(ii) Step 3 (discover/visualize). The deck's definition: "When we look at the test set, we are likely to notice patterns in that and based on that we may select certain models. This leads to biased estimation on test set, which may not generalize well in practice. This is called data snooping bias." [recorded] — the remedy is to separate the test set *before* exploring and never look at it.

(iii) "A lot more time is spent on capturing and processing data needed for ML and taking decisions based on output of ML module" — data plumbing and downstream decisions, with "strong collaboration with domain experts, product managers and eng-teams." [recorded]

## 2. The broadcast that tells on you

(i) `X = [[0,1],[2,3],[4,5]]`; column means: $(0+2+4)/3 = 2$, $(1+3+5)/3 = 3$ → `mu = [2., 3.]`. [verified-NumPy]

(ii) `X - mu`: `(3,2) - (2,)` — the `(2,)` aligns on the trailing axis and is stretched down all 3 rows:
```
[[0-2, 1-3],    [[-2., -2.],
 [2-2, 3-3],  =  [ 0.,  0.],
 [4-2, 5-3]]     [ 2.,  2.]]
```
[verified-NumPy]

(iii) `mu[:, np.newaxis]` has shape `(2,1)`. Broadcasting `(3,2)` against `(2,1)`: trailing axes $2$ vs $1$ → the $1$ stretches to $2$; next axes $3$ vs $2$ → neither is $1$, so it errors: `ValueError: operands could not be broadcast together with shapes (3,2) (2,1)`. [verified-NumPy] It is the good outcome because the alternative — a *silent* wrong-alignment subtraction — would poison every downstream number with no warning; the `ValueError` points at exactly the line where the convention was mixed up (§36.3's "that error is your friend").

## 3. The shuffle case

(i) Accuracy $= (TP+TN)/(TP+TN+FP+FN) = (13+196)/381 = 209/381 \approx 0.5486$ [verified]. Precision $= TP/(TP+FP) = 13/(13+0) = 1.0$ [verified]. Recall $= TP/(TP+FN) = 13/(13+172) = 13/185 \approx 0.0703$ [verified]. They match the recorded numbers to every shown digit.

(ii) With `shuffle=False`, SGD sees all 863 $+1$ rows first: the perceptron walks deep into five-territory, racking up positive predictions it believes in — then the 1032 $-1$ rows arrive and it over-corrects, ending up predicting $-1$ almost everywhere. The 13 rows it still calls positive are ones it is sure about (hence precision $1.0$), but it finds only 13 of the 185 true positives (hence recall $0.0703$).

(iii) `shuffle=True` in the `Perceptron` constructor (it is the default; the course's eg set it `False` explicitly).

## 4. Spot the leak

(i) `fit_transform` on the test set re-learns everything the pipeline learns: the `SimpleImputer`'s per-feature medians and the `StandardScaler`'s per-feature $\mu, \sigma$ — computed from the *test* rows this time.

(ii) The deck's own slide-57 law: "Note that all these transformers are learnt on the training data and then applied on the training and test data to transform them. Never learn these transformers on the full dataset." [recorded]

(iii) `wine_features_test_tr = transform_pipeline.transform(wine_features_test)` — replaying the train-learned statistics on test. This restores §37.8's discipline: `fit` learns on the training set, `transform` applies the learned transformation to new data; the test set only ever sees `transform`.

## 5. Read the dashboard

(i) Train MSE $0.0$, test MSE $0.5813$: textbook **overfitting** — the model memorized the training rows (zero training error) and fails on fresh data. The deck's own verdict: "Note that the training error is $0$, while the test error is $0.58$. This is an example of an overfitted model." [recorded]

(ii) $R^2 = -0.4$: §37.5's third landmark — the model is *worse than always predicting the mean* ("the model can be arbitrarily worse" than the constant predictor). This is not an untuned model; it is a broken pipeline (sign error, leak, or nonsense features). Debug first — start at §40.2's case 1 — don't tune.

(iii) Both errors high ($\approx 0.8$) and flat over a $10\times$ increase in $n$: **underfit** (§39.4's first reading) — the model class can't express the pattern, and more data will not help. Buy capacity (more/better features, a richer model) or reduce regularization — not more data.

## 6. Search arithmetic

(i) Grid 1: $3$ values of `n_estimators` $\times$ $4$ values of `max_features` $= 12$ combos. Grid 2: $2$ values of `n_estimators` $\times$ $3$ values of `max_features` $= 6$ combos. Total $12 + 6 = 18$ combinations; with `cv=5`, each combination is trained $5$ times: $18 \times 5 = 90$ fits. [verified]

(ii) The warning, verbatim [recorded]:
```
UserWarning: The total space of parameters 8 is smaller than n_iter=10.
Running 8 iterations. For exhaustive searches, use GridSearchCV.
```
sklearn ran only the $8$ existing combinations instead of the requested $10$ samples.

(iii) Randomized search is better on large or continuous spaces, where a fixed budget of `n_iter` samples beats an exponential grid; grid search is better on small discrete spaces where exhaustive enumeration is affordable.

## 7. Spend the test set once

(i) §39.6's five steps: 1. Split train/validation/test. 2. Train every hyperparameter combination on train (`n_jobs=-1`; `error_score` to survive bad combos). 3. Evaluate each on validation; pick the best. 4. Retrain the winner on train+validation. 5. Evaluate once on test.

(ii) The CI belongs to step 5 — the final, once-spent test evaluation. The point estimate ($0.3535$) says where the error landed; the 95% interval $(0.2916, 0.4153)$ says how much that number could wobble on a different test draw — the same "one split is a noisy number" honesty as §39.1, quantified.

(iii) Step 4: with the default `refit=True`, `GridSearchCV` automatically refits the winning combination on the whole training data, so `best_estimator_` is ready to predict.

## 8. Error analysis, the deck's way

(i) "It is also useful to analyze the errors in prediction and understand its causes and fix them." [recorded]

(ii) The sorted `(importance, feature)` list ranks features by how much the forest relied on them; the deck's decision: "Based on this information, we may drop features that are not so important." [recorded]

(iii) The deck's verdict: "LinReg has better MSE and more precise estimation compared to DT." [recorded] It rests on both numbers per model: the *mean* CV MSE (linear $0.4316$ < tree $0.6852$ — better) and the *standard deviation* (linear $0.0836$ < tree $0.1667$ — more precise). The forest beats both ($0.3457 \pm 0.0736$), which is why the deck tunes it next.

## 9. The suspiciously-good audit

(i) §40.4's checklist, in order: (1) was something `fit` on the test set? (2) were statistics learned on the full dataset? (3) was the test set spent more than once? (4) are there duplicated rows across the split? (5) is a feature secretly the label?

(ii) Suspect (1): grep for `fit(` applied to anything named `test` — `pca.fit(x_test)`, `scaler.fit_transform(X_all)` before the split. Suspect (2): any `fit`/`fit_transform` on the whole `X` before `train_test_split`, or a transformer `fit` outside the CV fold loop.

(iii) Suspect (iv): near-duplicate rows leaking the same information into both train and test, so the score measures memorization, not generalization. The two-second check: `df.duplicated().sum()` before the split (the book's addition — flagged as such in the review log).
