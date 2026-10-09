# Solutions — Chapter 28: Regularization: ridge and lasso

## Problem 1 — Ridge gradient, by hand

(i) Expand. $(Xw - y)^T(Xw - y) = (w^TX^T - y^T)(Xw - y) = w^TX^TXw - w^TX^Ty - y^TXw + y^Ty$. The middle two terms are scalars and transposes of each other: $y^TXw = (w^TX^Ty)^T = w^TX^Ty$. So
$$J(w) = \tfrac{1}{2}\big(w^TX^TXw - 2w^TX^Ty + y^Ty + \lambda w^Tw\big).$$

(ii) Differentiate term by term (§10.1's recipe; $X^TX$ is symmetric):
- $\nabla_w\, \tfrac{1}{2}w^TX^TXw = X^TXw$,
- $\nabla_w\, \tfrac{1}{2}(-2w^TX^Ty) = -X^Ty$,
- $\nabla_w\, \tfrac{1}{2}y^Ty = 0$,
- $\nabla_w\, \tfrac{1}{2}\lambda w^Tw = \lambda w$.

Adding: $\boxed{\nabla_w J = X^TXw - X^Ty + \lambda w}$.

(iii) Stationarity $\nabla_w J = 0$ gives $X^TXw - X^Ty + \lambda w = 0$, i.e. $\boxed{(X^TX + \lambda I)w = X^Ty}$.

## Problem 2 — The two endpoints

(i) $\lambda \to 0$: the $\lambda I$ term vanishes, so $(X^TX + \lambda I)^{-1} \to (X^TX)^{-1}$ (here we use the assumed invertibility), and $\hat w \to \boxed{(X^TX)^{-1}X^Ty}$ — ordinary least squares.

(ii) $\lambda \to \infty$: factor $\lambda$ out, $(X^TX + \lambda I)^{-1} = \frac{1}{\lambda}\big(\tfrac{1}{\lambda}X^TX + I\big)^{-1} \to \frac{1}{\lambda}I \to 0$ as a matrix, so $\hat w = (X^TX + \lambda I)^{-1}X^Ty \to \boxed{0}$. The penalty term dominates the data term.

## Problem 3 — Soft-thresholding in 1-D

(i) $J(w) = \tfrac{1}{2}(w^2\lVert x\rVert^2 - 2w\,x^Ty + \lVert y\rVert^2) + \tfrac{\lambda}{2}\lvert w\rvert$. Write $\rho = x^Ty$ and $s = \lVert x\rVert^2 > 0$. For $w \ne 0$ the (sub)derivative is $sw - \rho + \tfrac{\lambda}{2}\mathrm{sign}(w)$; at $w = 0$ the subdifferential of $\tfrac{\lambda}{2}\lvert w\rvert$ is the interval $[-\lambda/2,\, \lambda/2]$.

- If $\rho > \lambda/2$: optimum at $w > 0$ with $sw - \rho + \lambda/2 = 0$, i.e. $w = (\rho - \lambda/2)/s$.
- If $\rho < -\lambda/2$: optimum at $w < 0$ with $sw - \rho - \lambda/2 = 0$, i.e. $w = (\rho + \lambda/2)/s$.
- If $|\rho| \le \lambda/2$: $w = 0$ is optimal, because $0$ lies in the subdifferential: $- \rho \in [-\lambda/2, \lambda/2]$.

Together: $\boxed{\hat w = \mathrm{sign}(\rho)\,(|\rho| - \lambda/2)_+/s}$ with $s = \lVert x\rVert^2$.

(ii) Ridge's 1-D answer $\rho/(s + \lambda)$ is a *smooth rescaling*: it equals $0$ only if $\rho = 0$ exactly — otherwise merely smaller than the OLS value $\rho/s$. Lasso's answer has a **dead zone**: whenever $|\rho| \le \lambda/2$ (the feature's correlation with $y$ is weaker than half the tax rate), the coefficient is *exactly* $0$, not just small. That dead zone is the algebraic face of the diamond's corners (§28.7).

## Problem 4 — One GD step vs the closed form

$X = (1, 2)^T$, $y = (2, 3)^T$: $X^TX = 5$, $X^Ty = 1\cdot 2 + 2\cdot 3 = 8$.

(i) $w_1 = w_0 - \alpha\big(X^T(Xw_0 - y) + \lambda w_0\big) = 0 - 0.1\big(X^T(-y) + 0\big) = 0.1 \cdot 8 = \boxed{0.8}$.

(ii) Exact ridge: $\hat w = X^Ty/(X^TX + \lambda) = 8/(5 + 2) = \boxed{8/7 \approx 1.143}$. Yes — $w_1 = 0.8$ moved from $0$ toward $1.143$ (about $70\%$ of the way in one step, since this 1-D problem's optimal step would be $1/(5+2) = 1/7 \approx 0.143$).

## Problem 5 — Read a $\lambda$ table

(i) The lecture's step 2 picks the $\lambda$ with the **least validation error**: $\boxed{\lambda^* = 10^{-1}}$, validation MSE $0.47$ (the bowl's bottom).

(ii) Training MSE is smallest at $\lambda = 10^{-4}$ ($0.20$), but training error is *optimistic* — §23.9's trap: the $\lambda \to 0$ end memorizes the training sample (§28.1, §28.9's high-variance end), as the validation column proves ($0.92$, the worst). Never select on training error.

(iii) Step 3: retrain with $\lambda^* = 0.1$ on the **entire training set** (train + validation pooled). Step 4: report performance on the **test set**, untouched until now.

## Problem 6 — Why the diamond zeroes and the ball doesn't

Lasso's 1-D solution (Problem 3) is $\mathrm{sign}(\rho)\,(|\rho| - \lambda/2)_+/\lVert x\rVert^2$: the $(\cdot)_+$ builds in a dead zone — any coordinate whose data-pull $|\rho|$ falls below the tax threshold $\lambda/2$ is set to *exactly* zero rather than merely shrunk. Geometrically (§28.7's figure), minimizing the lasso objective is minimizing squared error inside the diamond $\lVert w\rVert_1 \le t$, and the error ellipses generically first touch a diamond at a *corner* — a corner sits on an axis, i.e. some $w_j = 0$ exactly. Ridge's 1-D answer $\rho/(\lVert x\rVert^2 + \lambda)$ has no dead zone (zero only if $\rho = 0$ exactly), and its budget region is a smooth ball with no corners, so the ellipse touch-point keeps every coordinate nonzero: shrinkage without selection.

## Problem 7 — sklearn practice (MLP slides)

(i) `alpha` **is** this chapter's $\lambda$: `Ridge(alpha=1e-3)` sets the regularization rate to $10^{-3}$ (same for `SGDRegressor(alpha=1e-3, penalty='l2')`).

(ii) They automate step 2 of §28.10's procedure: `RidgeCV` / `LassoCV` run the cross-validated search over candidate $\lambda$ values internally and return the $\lambda$ with the least CV error — no hand-written grid loop needed. (`GridSearchCV` / `RandomizedSearchCV` generalize the same pattern to any hyperparameter.)

(iii) The lecture's third type is **elastic net** — the $\ell_1$ + $\ell_2$ combination — and the MLP slides' CV estimator is `ElasticNetCV`.
