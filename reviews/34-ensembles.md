# Review log — Chapter 34: Ensembles — bagging, boosting, AdaBoost

## Sources actually used (verified by content, not folder name)

- **MLT Week-11 practice assignment** (`MLT-20261009T001607Z-1-001/MLT/Practice Assignment/Week_11.pdf`): Q1 (random-forest feature subsetting: decreasing features lowers tree correlation; increasing improves individual trees), Q3 (bagging reduces *variance*; inefficient if classifiers fully correlated), Q5 + its worked solution (AdaBoost update: $D_0(i)=1/n$, $\alpha=\ln\sqrt{(1-e)/e}$, $w_1=w_0e^{\pm\alpha}$, normalize). Formulas read off the rendered PDF page as an image (pdftoppm, /tmp/pa11p-5.png) because pdftotext dropped the math.
- **MLT Week-11 revision slides** (`MLT-.../MLT/PPT/revision/Week 11_Revision.pdf`, image-only, read via pdftotext which extracted the text layer): bagging definition, RF steps (bootstrap → tree on random feature subset → repeat, hundreds of trees; start with $\sqrt d$ features; "wide variety of trees"), boosting = "build a model, then a second model to rectify the errors of the first", weak learner = slightly better than random, stump = one split, AdaBoost performance $= \tfrac12\ln((1-\text{error})/\text{error})$, increase weights of misclassified / decrease of correct.
- **ISL §8.2.1** (bagging: $\sigma^2/n$ averaging argument, deep unpruned trees = high variance/low bias, majority vote, $B$ not critical / no overfitting, OOB ~2/3 in-bag + ~B/3 OOB predictions, OOB $\approx$ LOOCV), **ISL §8.2.3** (boosting: sequential not parallel, no bootstrap, fit-to-residuals + shrinkage $\lambda$, "learn slowly", smaller trees suffice, stumps $\to$ additive model, 3 tuning params $B,\lambda,d$, boosting *can* overfit if $B$ too large $\to$ CV).
- **ESL §8.7** (bagging: averaging reduces variance, leaves bias unchanged under squared-error; population aggregation never increases MSE; 0–1 loss caveat — bagging a bad classifier can make it worse; wisdom of crowds / independence caveat; stable procedures like nearest neighbours barely affected; interpretability lost), **ESL ch. 14** (AdaBoost.M1 algorithm box, weak learner def, weighted majority vote $G(x)=\mathrm{sign}(\sum\alpha_mG_m)$, hard points get ever-increasing weight, stump experiment $45.8\%\to5.8\%$ beating a 244-node tree at $24.7\%$, "boosting is fundamentally different" from bagging).
- **MITx 6.036 ch. 14 notes** (bagging: $B$ bootstrap datasets of size $n$, regression average, classification proportion + $\arg\max$, reduces estimation error, interpretability lost).
- The PPT/Week 11 "Lectures_1,2,3 / Lectures_4,5,6" PDFs turned out to be **soft-margin SVM derivations, not ensembles** (rendered as images to confirm) — not used.

## Numbers recomputed (twice: by hand + numpy)

- **§34.7 AdaBoost worked example** — all three rounds derived by hand in exact fractions and verified with `/tmp/adaboost_check.py` (exhaustive stump search over all thresholds × both orientations confirmed each round's argmin):
  - R1: $e_1=1/8$, $\alpha_1=\tfrac12\ln7\approx0.9730$, $Z=\sqrt7/4$, $D_1(4)=1/2$, others $1/14$ (sums to 1).
  - R2: $e_2=3/14$, $\alpha_2=\tfrac12\ln(11/3)\approx0.6496$, $Z=(11/7)/\sqrt{11/3}$, weights $1/6, 1/22, 7/22$ (sums to 1). The reused $G_1$ would err $1/2$, so $G_2$ (flipped stump) wins — shown in chapter.
  - R3: $e_3=7/22$, $\alpha_3=\tfrac12\ln(15/7)\approx0.3811$, $Z=(15/11)/\sqrt{15/7}$, weights $11/90, 1/30, 1/2$ (sums to 1).
  - Final vote scores: $-0.7044$ ($x\le-0.5$), $2.0037$ (middle), $+0.7044$ ($x>3.5$) — verified in numpy; $x=4$ still misclassified, as the chapter states.
- **§34.2 eg 1**: $3(0.3)^2(0.7)+(0.3)^3=0.216$ — verified.
- **§34.4 eg 2**: $(1-1/100)^{100}\approx0.3660$, so in-bag $\approx0.634$ — verified in numpy.
- **Problem 1**: $10(0.008)(0.64)+5(0.0016)(0.8)+0.00032=0.05792$ — verified.
- **Problem 3**: $\alpha=\tfrac12\ln(7/3)\approx0.4236$; normalized misclassified weight $=s^2/(30s^2+70)=(7/3)/140=1/60\approx0.016667$ exactly — verified symbolically and numerically. Matches the practice assignment Q5 answer (c) form.
- **Problem 2**: $B/3=200$; $150(1-1/e)\approx94.8\approx95$ — verified.

## Derivations double-checked

- Normalization algebra in all three AdaBoost rounds re-derived via the $Z=(as+b/s)$ pattern using $s^2=(1-e)/e$; each round's $Z$ simplifies exactly (R1: $(7/4)/\sqrt7$; R2: $(11/7)/\sqrt{11/3}$; R3: $(15/11)/\sqrt{15/7}$) and the fractions $1/2,1/14,1/6,1/22,7/22,11/90,1/30$ follow exactly — no rounding in the chapter's fractions.
- $\alpha_m>0\iff e_m<0.5$ (Problem 4) — trivial from the log, checked.

## Cross-references verified against the actual files

- §21.5 (weak law; $\mathrm{Var}(\bar X_n)=\sigma^2/n$) — real section, grep-confirmed.
- §22.9 (classification, $y^i\in\{+1,-1\}$) — real, quoted verbatim.
- §28.9 (bias–variance trade-off, high-variance = sample-sensitive) — real.
- §29.21(ii) ("High variance — small data changes can reshuffle the splits; mitigated by ensembles (Chapter 34's topic)") — real, quoted; §29.22(i) (ensembles "directly attack trees' high-variance weakness") — real.
- §33.10 ("where the soft margin *tolerates* them via slack, AdaBoost *reweights* them and tries again. Same enemy, opposite tactic.") — real, quoted; §33.3 (slack $\xi_i$) — real.
- All "Basically, ..." blocks present under every complex topic (§§34.1–34.9); math in LaTeX; no emojis; English only.

## Thin / contradictory source points (flagged, not invented)

1. **The $\tfrac12$ in $\alpha_m$.** ESL's AdaBoost.M1 box uses $\alpha_m=\log((1-e_m)/e_m)$ *without* $\tfrac12$; the MLT revision slides and practice-assignment Q5 solution both use $\tfrac12\ln((1-e)/e)$. The chapter follows the **course's** form and carries an explicit `Note:` about the discrepancy. (The $\tfrac12$ rescales all $\alpha_m$ equally, so the vote's sign is unaffected either way.)
2. **Q5 solution wording.** It says "Weights assigned for creating the 1st bag" inside an *AdaBoost* question — sloppy ("bag" belongs to bagging; boosting has no bootstrap). The chapter uses correct terminology (weights for training the next stump) and does not repeat the wording.
3. **Random forests.** Sources support only: bootstrap + random feature subset per split, hundreds of trees, $\sqrt d$ starting rule, the correlation/strength trade-off. The chapter keeps RF to §34.3's decorrelation role and does not present it as a full topic (per the outline: bagging, boosting, AdaBoost).
4. **AdaBoost noise/outlier sensitivity** (the classic "weights blow up on mislabeled points" caveat) is **not** stated in any source I read — so the chapter does *not* claim it. §34.7's outlier is presented as "hard point," not as a warning about label noise.
5. **The $1-1/e\approx0.632$ OOB fraction** is my arithmetic from the bootstrap definition, not a source quote; ISL only says "around two-thirds." Presented as a computation in eg 2.
6. **Boosting for classification** details are explicitly "omitted" in ISL §8.2.3 — the chapter covers classification-boosting via AdaBoost (ESL + course materials) instead, and uses ISL's regression-boosting only for the sequential/shrinkage intuition.
7. **MITx 6.036's boosting one-liner** ("decreases both estimation and structural error") was too vague to anchor any claim — not used.
8. **Figures** (`chapters/assets/34-bagging-bootstrap.png`, `chapters/assets/34-adaboost-weights.png`) are original matplotlib plots generated for this chapter (scripts in /tmp: `fig_bagging2.py`, `fig_adaboost.py`); the AdaBoost figure's bar heights are the exact fractions from §34.7. HTML comments in the chapter mark them original.

## Fixes made during self-review

- First bagging figure (shallow trees on clean data) showed no visible disagreement between bootstrap trees — regenerated with smaller, noisier data and deeper trees so the "high variance → calmer average" story is actually visible.
- matplotlib `bar(..., linewidths=)` → `linewidth=` (API error in figure script).
- Checked that every problem in the problem set has a full solution in `solutions/34-ensembles.md` and none of the solutions leak into the chapter.

## Coordinator's pre-merge fix (2026-10-09, independent review)
- eg 2 (OOB counting): "point $i$ is OOB in $\approx 300/3 = 100$ trees" → "$\approx 300 \times 0.368 \approx 110$ trees". The exact expectation is $300(1-1/100)^{100} \approx 109.8$; "100" understated it by ~10%.
- All §34.7 AdaBoost numbers independently recomputed in numpy: rounds (1/14×7, 1/2), (1/6×3, 1/22×4, 7/22), (11/90×3, 1/30×4, 1/2); ensemble votes −0.7044 / 2.0037 / 0.7044 — all match.
