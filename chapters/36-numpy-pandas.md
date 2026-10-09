# Chapter 36: NumPy and Pandas for ML

Everything in §§36.1–36.6 comes from the MLT TA colab notes (the "Intro to Libraries" notebooks — vectors, matrices, more-on-numpy, random-sampling — plus the Week-1 programming-assignment solution for the MNIST data-matrix example), and §§36.7–36.9 from the MLP Week-1 material (Lecture 7 "Data loading", the Lecture 1–5 end-to-end wine-quality deck, and the Week-1 programming questions notebook with its solutions). Every code block below was re-run in this session (numpy 1.26.4, pandas 2.1.4) and shows the actual output — except the explicitly marked spots where the course notebooks used sklearn, which is not installed on this machine (§36.7's diabetes/breast-cancer notes and its `SimpleImputer` call). Those are reported as recorded in the course notebooks, not re-run.

This is the chapter where the book's math becomes code. Parts I–IV worked with $x^{(i)}$, $X$, $w^T x$ on paper; from here on, those live inside arrays.

**Notation.** The book's standing convention (chapters 31–35): the design matrix $\boxed{X \in \mathbb{R}^{n \times d}}$ holds the points $x^{(1)}, \dots, x^{(n)}$ as its rows in order, and $y \in \mathbb{R}^n$ holds the labels — $n$ points, $d$ features. Zero-indexed, so `X[0]` is $x^{(1)}$ and `X[i]` is $x^{(i+1)}$.

**Note (the convention clash).** Two older sources disagree with this. §22.6 wrote the points as $x^i$ (superscript, no parens), and the MLT TA colabs stack data as $(d, n)$ — features as rows, points as columns ("we transpose so that the data-matrix is $(d, n)$ and not $(n, d)$"). Chapters 23–24 and the whole sklearn ecosystem use $(n, d)$. This chapter uses the book's $(n, d)$ everywhere and flags the TA colab's $(d, n)$ when it matters (§36.4). Reading either convention into the other is the single most common shape bug in ML code.

## 36.1 The ndarray: the book's data structure

**Def (ndarray).** The TA colab's one-liner: "Everything in `NumPy` is an array." An **ndarray** = an $n$-dimensional grid of numbers of one type. A vector is a 1-D array, a matrix a 2-D array, an image a 3-D array (MNIST: $(60000, 28, 28)$ in the TA colab's Week-1 assignment).

i) **Create:** `np.array([...])`, `np.zeros(n)`, `np.ones((r, c))`, `np.arange(a, b)`, `np.linspace(a, b, k)`, `np.eye(n)`, `np.diag([...])`.
ii) **Inspect:** `.shape` (the book's dimensions: $(n, d)$), `.ndim` (array-dimension, not vector-space dimension — the TA colab warns these differ), `.size`, `.dtype`.

**eg 1 (the book's dataset, as code).** §22.3's six houses, features = rooms, area, distance-to-metro; label = price:

```python
X = np.array([[3,9,1.9],[2,7,2.1],[4,12,2.8],
              [5,16,0.9],[5,15,3.1],[4,11,1.6]])
y = np.array([50,32,66,98,85,69])
X.shape, y.shape, X.ndim, X.size, X.dtype
# (6, 3) (6,) 2 18 float64
X[2]          # row 2 = x^(3) = [ 4.  12.   2.8]
```

$X$ is the feature matrix of §23.1 ($n \times d$, rows are points as row vectors); $y$ is the label vector. `X[2]` is $x^{(3)}$ — zero-indexed, so row $i$ holds $x^{(i+1)}$.

## 36.2 Vectorized ops vs Python loops

The TA colab contrasts the two ways to add vectors: a Python loop with `zip` + `append`, versus `z = x + y`. Everything element-wise is supported directly:

i) `x + y`, `3 * x` (scaling), `x * y` = Hadamard product $x \odot y$ (§2.3's ops), `x ** 2`, `np.log10(x)` — element-wise functions of vectors (the colab's $f(x) = x^2$, $f(x) = 5x - 2$ examples).
ii) `x @ y` = the dot product $x^T y = \sum_j x_j y_j$ (§2.5); the colab "shall stick to this" since it matches the math.
iii) `np.linalg.norm(x)` ($L_2$ default; `ord=1` for $L_1$), `np.abs(A)` element-wise (§2.4's norms).

**eg 2 (why ML code never loops).** Sum of squares of $10^6$ numbers — Python loop vs vectorized, each timed 5 times, best of 5:

```python
a = np.arange(10**6, dtype=float)
t_loop = 0.191      # s :  s = 0.0
                    #      for v in a: s += v*v
t_vec  = 0.00078    # s :  s = (a**2).sum()
# speedup: ~246x; both sums 3.3333e+17, relative difference 1.12e-12
```

**Basically, ...** "A Python loop does one number at a time through the interpreter; a NumPy op hands the whole array to compiled C in one call. On a million elements that's the difference between $0.19$ s and under a millisecond — $\sim 246\times$ here — for the *same* answer. Every training loop in this book (Chapters 10, 37–42) runs millions of such ops, so the vectorized form is not style, it's the only thing that finishes."

## 36.3 Broadcasting: the silent shape rule

The TA colab's question: "Sometimes we would have to add a vector to each row or column of a matrix" — but $M + b$ is, strictly, "a slight abuse of notation as we can't add a matrix and a vector together." NumPy resolves the abuse with one rule:

**Def (broadcasting).** Align shapes on their *trailing* axes. An axis of size $1$ (or a missing axis) is *stretched* — repeated — to match. No copying happens; it's a view trick.

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])          # (2, 3)
M + np.array([1, 2, 3])            # (3,) -> stretched down each row
# array([[2, 4, 6],
#        [5, 7, 9]])
M + np.array([1, 2]).reshape(2, 1) # (2, 1) -> stretched across columns
# array([[2, 3, 4],
#        [6, 7, 8]])
```

**eg 3 (centering a data matrix).** §24.2's covariance is built from *centered* rows $x^{(i)} - \bar x$. Broadcasting does all $n$ subtractions at once:

```python
mu = X.mean(axis=0)      # column means: [ 3.83333333 11.66666667  2.06666667]
Xc = X - mu              # (6,3) - (3,) -> mu stretched down every row
Xc[0]                    # [-0.83333333 -2.66666667 -0.16666667]
Xc.mean(axis=0)          # [-0.  0. -0.]  -> columns now average to 0
```

**Basically, ...** "Broadcasting = NumPy pretends the small array is as big as the big one, by repeating it. $(3,)$ against $(6, 3)$: the 3 numbers get copied down all 6 rows. The rule is always 'line up the *right* edges, stretch the $1$s'. Once you see it, `X - mu` reads as the math $x^{(i)} - \bar x$ for all $i$ at once."

**Note.** Broadcasting fails loudly when nothing stretches: `(2,3) + (4,)` is a `ValueError`. That error is your friend — it means you mixed up the $(n, d)$ convention (the Note at the top).

## 36.4 Indexing, slicing, and the axis semantics

i) **Indexing:** `M[1, 2]` = element (row 1, col 2), zero-indexed (the colab prefers this over `M[1][2]`). **Slicing:** `M[2, :]` = row 2, `M[:, 1]` = column 1, `M[1:3, 2:4]` = submatrix — start included, end excluded, like Python:

```python
M = np.array([[1, 3, 0],[4, 2, 5],[9, 8, 7]])
M[1, 2], M[2, :], M[:, 1]
# 5  array([9, 8, 7])  array([3, 2, 8])
```

ii) **Fancy indexing:** arrays as indices — `x[[1, 3, 6]]` picks positions $1, 3, 6$; **boolean masks** — `x[x > 0]` keeps the positives. The colab's payoff example: ReLU, three lines,

```python
relu = lambda t: np.where(t > 0, t, 0)
relu(np.array([1, -2, 1, 3, -4, -3]))
# array([1, 0, 1, 3, 0, 0])
```

the activation function Chapter 41 runs inside every neural net — a boolean mask wearing a one-line disguise.

iii) **Reshape:** `x.reshape(3, 2)`, `M.reshape(6)`, `M.reshape(3, -1)` (NumPy infers the $-1$); `.T` transposes. `x[:, np.newaxis]` turns a $(d,)$ vector into a $(d, 1)$ column — the colab's bridge between the "flat" vector and the matrix forms. **Note:** this is also where you convert the TA colab's $(d, n)$ habit to the book's $(n, d)$: `X_dn.T` is $X$.

iv) **Axis semantics** — the one everyone gets backwards once. `axis=k` *collapses* axis $k$: the result has one entry per position of every *other* axis.

```python
A = np.arange(1, 9).reshape(2, 4)   # [[1 2 3 4],[5 6 7 8]]
A.sum(axis=0)  # [ 6  8 10 12]  -> rows collapsed: one value per COLUMN
A.sum(axis=1)  # [10 26]        -> columns collapsed: one value per ROW
M = np.arange(1, 10).reshape(3, 3)
M.mean(axis=0) # [4. 5. 6.]    -> column means (eg 3's mu)
M.var(axis=1)  # [0.66666667 ...] -> variance within each row
```

**Note.** The TA colab describes `axis=0` as "top-bottom operations … done on row-vectors" and `axis=1` as "left-right … done on column-vectors" — loose wording (its own "left-right" example sums *rows*). The collapse rule above is the correct one and matches the verified outputs; the chapter uses it, not the colab's phrasing.

**Basically, ...** "`axis=0` eats the rows: you're left with one number per column. `axis=1` eats the columns: one number per row. Want the mean of each feature? Features are columns, so `X.mean(axis=0)`. Want the length of each data point? Points are rows, so `np.linalg.norm(X, axis=1)`."

## 36.5 Reductions, sorting, stacking, comparing

The colab's "misc functions", verified:

```python
x = np.array([10, -3, 2, 15, 5])
np.max(x), np.argmax(x)    # 15, 3   -> value and its POSITION
np.min(x), np.argmin(x)    # -3, 1
np.sort(x)                 # [-3  2  5 10 15]
np.argsort(x)              # [1 2 4 0 3]  -> the permutation that sorts
```

i) `argmax`/`argsort` return *indices*, not values — the standard "which class / which example" query (the classification chapters will use `argmax` over score vectors to extract predicted classes).
ii) **Stacking:** `np.concatenate((A, B), axis=0)` piles rows ($4 \times 2$ from two $2 \times 2$s — the "add more data points" direction); `axis=1` glues columns ($2 \times 4$ — "add more features").
iii) **Comparing:** `np.array_equal(x, y)` for exact equality; `x == y` gives a boolean *array*, unusable in an `if`. For floats — which are never exact — `np.allclose(x, y, rtol=0, atol=1e-2)`: $|x - y| \le \text{atol} + \text{rtol}\cdot|y|$ element-wise. **Note:** eg 2's loop-vs-vectorized check used exactly this idea (relative difference $1.12\times 10^{-12}$). Never test trained weights with `==`; Chapter 40's debugging discipline starts here.

## 36.6 Random numbers: seeded and reproducible

Two APIs appear in the course materials, and they are *not* interchangeable calls:

i) **Legacy:** `np.random.seed(42)` then `np.random.permutation(10)`, `np.random.randn(...)` — global state (the MLP slide's choice).
ii) **Modern:** `rng = np.random.default_rng(seed=42)` then `rng.random(3)`, `rng.normal(...)`, `rng.choice(...)`, `rng.multivariate_normal(...)` — an explicit generator object (the TA colab's choice).

Same seed $\Rightarrow$ same stream, every run:

```python
np.random.default_rng(42).random(3)
# [0.77395605 0.43887844 0.85859792]
np.random.seed(42); np.random.permutation(10)
# [8 1 5 0 7 2 9 4 3 6]
```

**Note.** Pseudo-random, not random: the generator is a deterministic algorithm, so fixing the seed fixes the whole sequence. That is the entire point — reproducibility.

**eg 4 (train/test split, the MLP slide's function).** The slide: "Make sure to set the seed so that we get the same test set in the next run." `np.random.seed(42)` + `np.random.permutation(len(data))`, first $20\%$ = test:

```python
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```

(On §36.8's 10-row table: train $(8, 4)$, test $(2, 4)$, test rows = ids 9 and 2 — the same two, every run.) The slide's sklearn one-liner is the same idea packaged: `train_test_split(data, test_size=0.2, random_state=42)`.

**Basically, ...** "Randomness in ML is a controlled substance. You *want* shuffling and sampling (Chapter 39 lives on it), but you want the *same* shuffle every time you re-run, or you can never debug. Set the seed once, and 'random' becomes a fixed, replayable script."

## 36.7 pandas: the labeled table

**Def.** A **DataFrame** = a 2-D labeled table (rows = examples, columns = named features); a **Series** = one labeled column. If an ndarray is the math, a DataFrame is the math *with the metadata attached* (§22.3: column $j$ means the same thing in every row — the labels are that meaning, written down).

i) **Load:** `pd.read_csv(url, sep=";")`. The MLP slide's wine-quality flow: 1599 rows, 12 columns — 11 features + label `quality`, all `float64`, label `int64` (verified against the same UCI file the slide downloads).
ii) **First look:** `data.head()` (first 5 rows), `data.info()` (1599 entries, dtypes, non-null counts), `data.describe()` (count, mean, std, quartiles per numeric column), `data.shape`.
iii) **Select:** `data['quality']` (Series), `data[['fixed acidity','alcohol']]` (DataFrame), `data.iloc[train_indices]` / `data.loc[train_index]` (positional vs label — the slide uses both), boolean filter `data[data['price'] > 80]`.
iv) **Summarize:** `data['quality'].value_counts()` (the label distribution: quality $3$: $10$, $4$: $53$, $5$: $681$, $6$: $638$, $7$: $199$, $8$: $18$ — "lots of samples of average wines", the slide's words); `data.groupby('diagnosis').describe()` (the W1 programming notebook: breast-cancer, 569 rows × 33 cols; benign $B = 357$, malignant $M = 212$ — recorded in the notebook, the data file wasn't in this dump to re-run).
v) **Correlate:** `corr_matrix = exploration_set.corr()`; `corr_matrix['quality']` — alcohol $0.4762$, volatile acidity $-0.3906$ (verified; the slide rounds to $0.48$ / $-0.38$). Standard correlation $\in [-1, 1]$; "only captures linear relationship" (slide's warning).

**NaN semantics.** A column with one missing value becomes `float64` (NaN is a float) — verified in §36.8. And NaN is *contagious and unfindable by equality*:

```python
np.nan == np.nan   # False
3 + np.nan         # nan
```

So: `df.isna().sum()` counts missing per column (the slide's check). The slide's three-way discipline, verbatim in spirit: (a) *recorded-but-absent* → impute (`fillna`, or sklearn's `SimpleImputer(strategy="median")` — its `fit` learns the medians on the training set, `transform` applies them; recorded, sklearn not installed here); (b) *too broken* → `dropna()`; (c) *doesn't exist* (a pregnancy test for a male patient) → keep as NaN. Chapter 39 takes up imputation properly; the rule to carry forward is the slide's: never learn the imputer on the full dataset — train statistics on train only.

**Back to NumPy.** The round trip the slide shows both ways: `wine_features_tr = pd.DataFrame(tr_features, columns=wine_features.columns)` (ndarray → labeled), and the reverse every model needs —

```python
X = df[['area','rooms']].to_numpy()   # (10, 2) ndarray
y = df['price'].to_numpy()            # (10,)   ndarray
```

— the $X, y$ of §36.1, ready for Chapter 37.

**Basically, ...** "NumPy is the engine room; pandas is the loading dock with labels. You *receive* data as a DataFrame (named columns, mixed types, missing values visible), *explore* it with `head/describe/value_counts/groupby/corr`, *clean* it (`isna`, fill or drop), and *hand it over* as plain $X, y$ arrays the moment the model starts. Every sklearn `fit(X, y)` in Chapters 37–38 is the end of a pandas pipeline."

## 36.8 Worked end-to-end: load, clean, describe, extract

One small table, the whole §36.7 pipeline, every number re-run. (The CSV is inline so the example is self-contained.)

```python
import pandas as pd, numpy as np, io
csv = """id,area,rooms,price
1,900,3,50
2,700,2,32
3,1200,4,66
4,1600,5,98
5,1500,5,85
6,1100,4,69
7,950,3,55
8,1300,4,72
9,800,,40
10,1400,5,90
"""
df = pd.read_csv(io.StringIO(csv))
```

**Step 1 — look.** `df.shape` = $(10, 4)$. `df.dtypes`: `id, area, price` int64; `rooms` **float64** — the empty cell in row 9 forced it. `df.head(3)` shows rows 1–3; row 9's rooms cell is `NaN`.

**Step 2 — count the damage.** `df.isna().sum()`: rooms = 1, everything else 0.

**Step 3 — clean.** Median of rooms = $4.0$; `df['rooms'] = df['rooms'].fillna(4.0)`; `df.isna().sum().sum()` = 0. (This is the slide's `SimpleImputer(strategy="median")` by hand, on a table small enough to check.)

**Step 4 — describe.** `df.describe()`: mean price $65.70$, std $21.64$, area mean $1145.0$; `df['price'].corr(df['area'])` = $0.98$ — area nearly determines price, as built. `df.groupby('rooms')['price'].mean()`: 2→$32.00$, 3→$52.50$, 4→$61.75$, 5→$91.00$.

**Step 5 — extract.** `X = df[['area','rooms']].to_numpy()` → $(10, 2)$; `y = df['price'].to_numpy()` → $(10,)$. Done: the labeled table is now the book's $(X, y)$, and Chapter 37 can `fit` on it.

**Note.** The wine-quality deck runs exactly this pipeline at scale — `read_csv` → `head` → `info`/`describe` → `value_counts` → `corr` → split — and every one of its reported numbers re-verified here: 1599 rows, quality $\in [3, 8]$, alcohol–quality correlation $0.4762$, volatile-acidity–quality $-0.3906$.

## 36.9 Where this goes next

i) Chapter 37 feeds this chapter's $(X, y)$ to sklearn regressors — `fit`, `predict`, and the transformers of the MLP Lecture-7 slide (`fit` learns parameters on the training set, `transform` applies them; `fit_transform` does both). The slide's warning stands as the chapter's last word: *never learn a transformer on the full dataset* — the same discipline as §36.7's imputation note.
ii) The vectorization habit (§36.2) is what makes every gradient-descent loop from Chapter 10 onward actually run; broadcasting (§36.3) is how a bias vector joins a score matrix in Chapter 41 without a single loop.
iii) The $(n, d)$ convention is now load-bearing: Chapter 38's classifiers expect `X.shape == (n_samples, n_features)`. When a shape error bites, check the convention first (§36.1's Note) — it is the cheapest bug to find and the most common one to make.

## Problem set

1. **Broadcast or break.** For each pair, give the result shape or say it errors, with one line of the broadcasting rule: (i) $(4, 5) + (5,)$; (ii) $(4, 5) + (4, 1)$; (iii) $(4, 5) + (4,)$; (iv) $(6, 1, 5) + (5,)$.
2. **Reproduce the speedup.** On $n = 10^5$ numbers: (i) time a Python loop computing $\sum v^3$ vs `(a**3).sum()` (best of 5 each) and report the speedup; (ii) use `np.allclose` (not `==`) to check the two answers agree and say why `==` is the wrong test.
3. **Read the slice.** `M = np.arange(1, 17).reshape(4, 4)`. By hand, then check in code: (i) `M[1::2, :]`; (ii) `M[:, -1]`; (iii) `M[M > 12]`; (iv) write the mask expression that selects the *even* entries.
4. **Axis by hand.** `M = np.arange(1, 13).reshape(3, 4)`. (i) Compute `M.mean(axis=0)` and `M.sum(axis=1)` by hand. (ii) In one line each: which gives a per-feature statistic and which a per-point statistic? (iii) The TA colab calls `axis=0` "top-bottom"; give the collapse-rule reading that replaces it.
5. **Clean the table.** Load the §36.8 CSV in code. (i) Which column has missing values, and why did its dtype become `float64`? (ii) Fill with the median and verify zero NaNs remain. (iii) Report mean price per `rooms` via `groupby`, and the area–price correlation.
6. **Seeds.** (i) Run `np.random.default_rng(42).random(3)` twice — the two calls give identical output. Explain in one line why, and contrast with calling `.random(3)` twice on *one* `rng = np.random.default_rng(42)` object. (ii) The MLP slide's `split_train_test` calls `np.random.seed(42)` *inside* the function: what breaks if you call the function twice and expect different splits? (iii) In one line, why does any of this matter for Chapter 39's cross-validation?
7. **Course-sourced.** The MLP Week-1 programming notebook's Q5: `load_diabetes(as_frame=True, return_X_y=True)` returns the feature matrix as a `DataFrame` and the target as a `Series` (recorded in the notebook's solution output). (i) State both types. (ii) In one line, what does the DataFrame buy you over a bare ndarray here — answer in §22.3's vocabulary.
