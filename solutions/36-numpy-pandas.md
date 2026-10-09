# Solutions — Chapter 36: NumPy and Pandas for ML

## Problem 1 — Broadcast or break

Rule: align on trailing axes; size-$1$ or missing axes stretch; anything else errors.

i) $(4, 5) + (5,) \to (4, 5)$. The $(5,)$ aligns with the trailing axis and stretches down the 4 rows (the §36.3 row-vector case).
ii) $(4, 5) + (4, 1) \to (4, 5)$. Trailing: $5$ vs $1$ → the $1$ stretches; leading: $4$ vs $4$ → fine. (The §36.3 column-vector case.)
iii) $(4, 5) + (4,) \to$ **ValueError**. Trailing axes $5$ vs $4$: neither is $1$, nothing stretches.
iv) $(6, 1, 5) + (5,) \to (6, 1, 5)$. $(5,)$ aligns with the trailing $5$; the middle $1$ stretches to $6$ (and the leading $6$ is untouched).

All four confirmed with `np.broadcast_shapes`.

## Problem 2 — Reproduce the speedup

$n = 10^5$, best of 5 runs (numpy 1.26.4):

```python
a = np.arange(10**5, dtype=float)
# loop:      s = 0.0
#            for v in a: s += v**3          -> 0.0224 s
# vectorized: s = (a**3).sum()              -> 0.00037 s
# speedup: 60.9x
```

(i) $\approx 61\times$ on this machine (eg 2's $n = 10^6$ run gave $\sim$246$\times$ — the gap widens with $n$ because the per-element interpreter overhead dominates).

(ii) `np.allclose(s1, s2)` → `True`; `s1 == s2` → `False`. Both sums print as $2.499950\times 10^{19}$ but differ in the last bits: the loop accumulates left-to-right in Python floats while NumPy sums in a different (blocked/pairwise) order, so the round-off differs. `==` demands bit-identical floats — the wrong test for any two independently computed floating results; `allclose` (relative + absolute tolerance) is the right one (§36.5, §36.2's $1.12\times 10^{-12}$ relative difference).

## Problem 3 — Read the slice

$M = \begin{bmatrix} 1&2&3&4\\ 5&6&7&8\\ 9&10&11&12\\ 13&14&15&16 \end{bmatrix}$.

(i) `M[1::2, :]` = rows $1, 3$ (start at row 1, step 2): $\begin{bmatrix} 5&6&7&8\\ 13&14&15&16 \end{bmatrix}$.
(ii) `M[:, -1]` = last column: $[4, 8, 12, 16]$.
(iii) `M[M > 12]` = flat array of entries $> 12$, in row-major order: $[13, 14, 15, 16]$. **Note:** a boolean mask always returns a *flat* (1-D) array — the selected positions don't form a rectangle, so no shape is preserved.
(iv) Even entries: `M[M % 2 == 0]` → $[2, 4, 6, 8, 10, 12, 14, 16]$.

## Problem 4 — Axis by hand

$M = \begin{bmatrix} 1&2&3&4\\ 5&6&7&8\\ 9&10&11&12 \end{bmatrix}$.

(i) `M.mean(axis=0)`: collapse rows — column means $[5, 6, 7, 8]$ (e.g. column 0: $(1+5+9)/3 = 5$). `M.sum(axis=1)`: collapse columns — row sums $[10, 26, 42]$ (e.g. row 0: $1+2+3+4 = 10$).
(ii) `mean(axis=0)` is the per-*feature* statistic (one number per column); `sum(axis=1)` is the per-*point* statistic (one number per row).
(iii) Collapse-rule reading: `axis=0` eats the row axis, leaving one value per column; `axis=1` eats the column axis, leaving one value per row. ("Top-bottom" is just a loose name for the same thing — the collapse is what the code does.)

## Problem 5 — Clean the table

```python
df = pd.read_csv(io.StringIO(csv))   # the §36.8 CSV
```

(i) `df.isna().sum()` → `rooms: 1`, rest 0. The empty cell in row 9 is read as `NaN`; since `NaN` is a float, pandas upcasts the whole column — `rooms` is `float64` while `id/area/price` stay `int64` (§36.7's NaN note).

(ii) `df['rooms'].median()` = $4.0$; `df['rooms'] = df['rooms'].fillna(4.0)`; then `df.isna().sum().sum()` = $0$. Verified.

(iii) `df.groupby('rooms')['price'].mean()`:
- rooms $2.0 \to 32.00$ (one house: id 2)
- rooms $3.0 \to 52.50$ (ids 1, 7: $(50+55)/2$)
- rooms $4.0 \to 61.75$ (ids 3, 6, 8, 9: $(66+69+72+40)/4 = 247/4$)
- rooms $5.0 \to 91.00$ (ids 4, 5, 10: $(98+85+90)/3 = 273/3$)

`df['price'].corr(df['area'])` = $0.98$ — area nearly determines price, as the table was built.

## Problem 6 — Seeds

(i) Identical output both times. Each call builds a *fresh* generator from the same seed, and a seeded generator is a fixed deterministic stream — same seed, same first three draws: $[0.77395605, 0.43887844, 0.85859792]$. Contrast: on *one* `rng` object, the second `.random(3)` continues the stream (draws 4–6), so the outputs differ.

(ii) Nothing varies between the two calls: the seed is reset to 42 *inside* the function, so `np.random.permutation` replays the identical shuffle and both calls return the same train/test split. Expecting different splits breaks; expecting the *same* split (reproducibility) is exactly what the slide wants.

(iii) Cross-validation re-trains on $k$ different folds and compares scores — if the fold assignment isn't seeded, every re-run shuffles differently and the scores wobble for no reason; you can't tell a better model from a luckier shuffle. Seeding makes the comparison fair and replayable.

## Problem 7 — Course-sourced

(i) Feature matrix: `pandas.DataFrame`; target: `pandas.Series` (the W1 programming notebook's recorded solution output for `load_diabetes(as_frame=True, return_X_y=True)`; sklearn was not installed on this machine, so this is the notebook's output as recorded, not re-run).

(ii) In §22.3's vocabulary: the DataFrame carries the **metadata** — the column names say what each coordinate means (`age`, `bmi`, …) — so the table is human-readable while the numbers stay machine-usable; a bare ndarray keeps the numbers but drops the meaning.
