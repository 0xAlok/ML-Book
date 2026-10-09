# Review log — Chapter 36: NumPy and Pandas for ML

## Sources actually used (verified by content, not folder name)

- **MLT TA colab notes** (`MLT-20261009T001607Z-1-001/MLT/Live session slide - 2024 Sep/TA notes/Colab/Intro to Libraries/`), extracted via python json (not opened as text): `Copy of vectors in numpy.ipynb` (§36.1–36.2: Python-loop vs `x + y`, Hadamard $\odot$ / `x * y`, scaling, element-wise $f(x) = x^2/\log(5x-2)$, `x @ y` dot product ("we shall stick to this"), `np.zeros/ones/arange`, L1/L2 norms, `.shape`/`.ndim` incl. the "array-dimension vs vector-space dimension" warning); `Copy of matrices in numpy.ipynb` (§36.3–36.4: `M + b` row/column vector with the "slight abuse of notation" quote, indexing `M[1][2]` vs `M[1,2]`, row/col/submatrix slices, transpose, `reshape`/`np.newaxis`, products, `np.linalg` ops); `Copy of more on numpy.ipynb` (§36.3–36.5: reshape with `-1`, boolean-mask filtering, ReLU via `np.where(x > 0, x, 0)`, axis ops — see the loose-wording flag below, `np.concatenate`, max/argmax/sort/argsort, `np.array_equal`/`np.allclose`, the MNIST `(60000, 28, 28)` reshape-to-`(784, 60000)` example); `Copy of random_sampling.ipynb` (§36.6: `rng = np.random.default_rng(seed=42)`, Bernoulli/Gaussian/multivariate sampling, the $(d,n)$ transpose quote, sample-covariance code). The same folder's `Week 1 Programming Assignment/Prog_Assignment_Solution_by_TA.ipynb` supplied the MNIST `(d, n)` centering → covariance → PCA workflow shape (used only for the §36.1 MNIST-shape citation and §36.3's centering motivation; the PCA itself belongs to Chapter 24).
- **MLP Week-1 Lecture 7** (`MLP/Slides/Week 1/MLP Week 1 Lecture 7.pdf`, 14 pp., text-extractable): dataset loaders/fetchers/generators; "pandas.io provides tools to read from common formats like CSV, excel, json, SQL"; the transformer `fit/transform/fit_transform` methods and "Never learn these transformers on the full dataset" (carried into §36.7 and §36.9).
- **MLP Week Lecture 1–5 deck** (`MLP/Slides/Week 1/MLP Week Lecture 1-5.pdf`, 89 pp., text-extractable — the end-to-end wine-quality project): `pd.read_csv(url, sep=";")`, `head()`, `info()`, `describe()`, `value_counts()`, `split_train_test` (np.random.seed(42) + np.random.permutation), `train_test_split(data, test_size=0.2, random_state=42)`, stratified sampling via `StratifiedShuffleSplit`, `corr()`, `scatter_matrix`, `isna().sum()`, the three-way missing-value discipline (impute / dropna / keep-as-NaN) with `SimpleImputer(strategy="median")` `fit` on train then `transform`, `pd.DataFrame(tr_features, columns=...)`. All its reported numbers were re-verified against the same UCI file it downloads (see below).
- **MLP Week-1 programming notebook** (`Practice and Graded Assignment Solutions/Week 1/MLP_Week_1_Programming_questions.ipynb`): Q1–Q6 shape questions (used only in passing for the pandas-type Q5), the breast-cancer `pd.read_csv` → `data.head()` → `data.shape` = `(569, 33)` → `groupby('diagnosis')` = 357 B / 212 M → histograms (recorded, not re-run — sklearn/pandas_profiling not installed and the data file isn't in the dump).
- **Sejal's Week_1 notes**: read; only the 8-step ML-project framing and sklearn API taxonomy — nothing chapter-specific; not cited.
- **ISL-with-Python.pdf**: not used (the course sources covered everything; per instructions, not cited).
- Earlier book chapters as cited: §2.3 (vector ops), §2.4 (norms), §2.5 (dot product), §22.3 (data as vectors + metadata), §22.6 (notation), §23.1 (feature matrix $n \times d$), §24.2 (centered covariance). Chapters 37–38 do not exist yet; forward pointers to them are aspirational (what they *will* build on), never §x.y refs.

## Numbers recomputed (by hand + numpy, this session; numpy 1.26.4, pandas 2.1.4)

- **eg 1**: `X.shape (6,3)`, `y.shape (6,)`, `X.ndim 2`, `X.size 18`, `dtype float64`, `X[2] = [4., 12., 2.8]` ✓.
- **eg 2 timing** ($n=10^6$, best of 5): loop $0.191$ s vs vectorized $0.00078$ s → $\sim$246$\times$; sums both $3.3333\times 10^{17}$, relative difference $1.12\times 10^{-12}$ ✓. First attempt's single-shot timing gave only 25.6$\times$ (cold-cache) — re-ran best-of-5; the chapter reports the stable number.
- **§36.3 broadcasting**: `M + [1,2,3]` → `[[2,4,6],[5,7,9]]`; `M + [[1],[2]]` → `[[2,3,4],[6,7,8]]` ✓ (matches the TA colab's recorded outputs).
- **eg 3 centering**: column means `[3.83333333, 11.66666667, 2.06666667]`; `Xc[0] = [-0.83333333, -2.66666667, -0.16666667]`; centered column means `[-0., 0., -0.]` ✓.
- **§36.4 indexing**: `M[1,2]=5`, `M[2,:]=[9,8,7]`, `M[:,1]=[3,2,8]` (caught a transcription error: first draft said `[0,5,7]` — fixed); `x[[1,3,6]]=[0,3,1]`, `x[x>0]=[3,1,5,1,5]`, ReLU → `[1,0,1,3,0,0]` ✓; `A.sum(axis=0)=[6,8,10,12]`, `A.sum(axis=1)=[10,26]`; `M.var(axis=1)=[0.66666667]*3` ✓.
- **§36.5**: max 15/argmax 3, min −3/argmin 1, sort `[-3,2,5,10,15]`, argsort `[1,2,4,0,3]`; concatenate axis=0 → $(4,2)$, axis=1 → $(2,4)$; `array_equal` True/False; `allclose(...,rtol=0,atol=1e-2)` True ✓.
- **§36.6 RNG**: `default_rng(42).random(3) = [0.77395605, 0.43887844, 0.85859792]`; legacy `seed(42); permutation(10) = [8,1,5,0,7,2,9,4,3,6]` ✓. **eg 4 split** on the §36.8 table: train $(8,4)$, test $(2,4)$, test ids `[9, 2]` ✓.
- **Wine-quality** (downloaded from the slide's exact UCI URL to /tmp): `(1599, 12)`; 11 float64 + 1 int64; quality $\in [3,8]$; value_counts $3$:10, $4$:53, $5$:681, $6$:638, $7$:199, $8$:18; alcohol–quality corr $0.4762$; volatile-acidity–quality $-0.3906$ — the slide's $0.48$/$-0.38$ are correct roundings ✓.
- **§36.8 mini-CSV**: `rooms` dtype float64 (NaN upcast); `isna().sum()` rooms=1; median $4.0$; post-fill NaNs 0; `describe` mean price $65.70$, std $21.64$, area mean $1145.0$; area–price corr $0.98$; groupby means $32.00/52.50/61.75/91.00$; `X (10,2)`, `y (10,)` ndarrays ✓.
- **Problem set**: P1 via `np.broadcast_shapes` ($(4,5)+(5,)\!\to\!(4,5)$; $(4,5)+(4,1)\!\to\!(4,5)$; $(4,5)+(4,)\!\to\!$ValueError; $(6,1,5)+(5,)\!\to\!(6,1,5)$); P2 ($n=10^5$): loop $0.0224$ s vs vec $0.00037$ s → $60.9\times$; `allclose` True, `==` False ✓; P3/P4/P5 all recomputed by hand and in code ✓.

## Cross-references verified against the actual files

- Script-checked: all 14 `§x.y` cites in the chapter resolve to real `## x.y` headers (§2.3, §2.4, §2.5, §22.3, §22.6, §23.1, §24.2, and eight self-refs §36.x).
- Forward mentions of Chapters 37–42 are chapter-number only, never section-level (those chapters don't exist yet).

## Thin / contradictory source points (flagged, not invented)

1. **$(n,d)$ vs $(d,n)$**: the TA colabs use $(d,n)$ with explicit "transpose so the data-matrix is $(d,n)$"; the book (ch 23–24) and sklearn use $(n,d)$. The chapter adopts the book's convention and carries an explicit `Note:` — the clash is the chapter's stated #1 shape-bug source.
2. **$x^{(i)}$ vs $x^i$**: §22.6 defines superscript $x^i$; chapters 31–35 use $x^{(i)}$ citing §22.9. The chapter follows 31–35 and flags the drift in the Notation note.
3. **TA colab's axis wording**: "top-bottom/row-vectors" vs "left-right/column-vectors" is loose (its own left-right example sums rows). The chapter uses the collapse rule (`axis=k` eats axis $k$) and notes the correction — outputs verified identical either way.
4. **sklearn-dependent notebook outputs not re-run**: breast-cancer `(569,33)` / 357 B / 212 M; diabetes DataFrame/Series types; `SimpleImputer(strategy="median")` behavior. sklearn isn't installed (pip blocked: externally-managed env; venv install failed: no disk space) and the data files aren't in the dump. All three are explicitly marked "recorded in the course notebook, not re-run".
5. **No figures**: the TA colab's broadcasting PNG is embedded base64 in the notebook; the chapter carries broadcasting with tables + verified outputs instead — no diagram genuinely needed, none added (per the no-padding rule).
6. **eg 2 presentation**: times shown as commented constants rather than live `time.perf_counter()` output, since wall-clock numbers vary run to run; the method (best of 5) and machine-relative speedup are what the text claims.

## Fixes made during self-review

- `M[:, 1]` output corrected: `[0, 5, 7]` → `[3, 2, 8]` (was the column-2 value; caught by re-running).
- Notation box contradicted itself ("point $i$ = row $i$" vs zero-indexing); now reads "rows are $x^{(1)}\dots x^{(n)}$ in order; `X[0]` is $x^{(1)}$".
- eg 3's §24.2 cite said "$C = \frac{1}{n}A^TA$" (that's §24.9's form); reworded to the centering fact that is actually in §24.2.
- Intro dropped unverified notebook cell counts ("41, 83"); MNIST-shape citation re-attributed to the Week-1 programming-assignment colab (it isn't in Intro-to-Libraries).
- §36.5's "Chapter 38's predicted-class extraction is argmax" softened to a forward-looking statement (ch 38 unwritten).
- Intro's "§36.9's SimpleImputer call" corrected to §36.7 (where it actually is).
- Problem 6(i) rewritten (was self-contradictory about "fresh interpreters").
