# Review log — Chapter 13. Duality and KKT conditions

## Sources used (all committed under `sources/course materials/MLF-20261009T001559Z-1-001/MLF/`)

1. `PPT/Week 9/9.3.Revisiting Constrained Optimization.pdf` (lecturer's handwritten slides, 3 pp.) — Lagrangian $L(x,\lambda)=f(x)+\lambda h(x)$; $\max_{\lambda\ge0}L(x,\lambda)$ = $f(x)$ if $h(x)\le0$ else $\infty$; primal $\min_x\max_{\lambda\ge0}L$; swap to dual $\max_{\lambda\ge0}\min_x L$; dual function $g(\lambda)$ concave ("exercise"); dual typically easier (unconstrained inner, $\lambda\ge0$ outer).
2. `PPT/Week 9/9.4.Relation between Primal and Dual Problem, KKT Conditions.pdf` (7 pp.) — $J(x)$ definition; weak duality proof ($L\le J \Rightarrow \max\min L \le f(x^\star)$); strong duality "if $f,h$ convex (upto some regularity)"; C1 stationarity $\nabla f(x^\star)+\lambda^\star\nabla h(x^\star)=0$ and C2 $\lambda^\star h(x^\star)=0$ derived from $f(x^\star)=g(\lambda^\star)$; four KKT conditions; sufficiency $\Rightarrow$ local optima (modulo regularity, not proved); general form with $m$ inequalities ($u_i\ge0$) and $n$ equalities ($v_j$ free).
3. `PPT/Week 9/9.5.KKT conditions continued.pdf` (2 pp.; p.1 duplicates 9.4 p.7) — SVM example: $\min_w \tfrac12\lVert w\rVert^2$ s.t. $w^Tx_i y_i \ge 1\ \forall i$; quadratic $\Rightarrow$ convex; linear constraints $\Rightarrow$ convex; strong duality $\Rightarrow$ solve the dual; dual helps "go from linear models to non-linear models" (kernels).
4. `PPT/Week 9/Week 9 Tutorial.pdf` — canonical worked KKT example (LP: min $3x_1+x_2$ s.t. $x_1-x_2+4\le0$, $-3x_1+2x_2+10\le0$, $x_1,x_2\ge0$; solution $u_1=9,u_2=4,x_1=18,x_2=22,f^\star=76$ — every number re-derived below); KKT four conditions (stationarity/CS/primal/dual feasibility); diet problem primal+dual pair; LP solution-possibility types; primal/dual possibilities table.
5. `Transcripts/Week 9/Revisiting Constrained Optimization.pdf` — full verbal derivation of the primal-as-minmax and the swap; authoritative wording for "can be hard to solve" and "typically easier".
6. `Transcripts/Week 9/Relation between Primal and Dual Problem, KKT Conditions.pdf` — weak-duality proof via $J(x)$; strong duality "upto some regularity conditions" (never named); C1/C2 derivations; necessity vs sufficiency ("completely characterize the optimal solution" for convex); general KKT form. Treated as authoritative where slides are terse.
7. `Transcripts/Week 9/KKT conditions continued..pdf` — SVM motivation; KKT "both necessary and sufficient for most convex optimization problems"; dual $\Rightarrow$ kernel methods. SVM objective stated here as "norm $w$ squared" (no $1/2$) — see discrepancies.
8. `Transcripts/Week 9/Applications of Optimization in Machine Learning.pdf` and `PPT/Week 9/9.1, 9.2` — scanned for duality/KKT content: none (linear regression / convex-function properties; Ch 12 territory).
9. `PPT/Week 10/*.pdf` + `Week_10 tutorial.pdf` — all Chapter-6 probability; unrelated to this chapter (see note 1 below). `Transcripts/Week 10/` likewise probability. "WEEK 6 by jimmi.pdf" ignored per instructions.

## What was checked

- **Weak duality proof** (§13.6): re-derived from the transcript's $J(x)$ argument; the step $\min_x L \le \min_x J$ and the max-over-$\lambda$ step both verified.
- **Concavity of $g$** (§13.5): proof direction checked — $\min_x[\theta A+(1-\theta)B] \ge \theta\min A+(1-\theta)\min B$ is correct, giving the concave ($\ge$) chord inequality.
- **KKT derivation** (§13.8): C1 uses "unconstrained minimizer $\Rightarrow$ zero gradient" (§10.8, necessity direction only — correct usage); C2 sandwich argument re-checked: $f(x^\star)\le f(x^\star)+\lambda^\star h(x^\star)\le f(x^\star)$ forces the product to 0.
- **LP example** (§13.12, tutorial's): recomputed by hand — $u_2=4$, $u_1=9$ from stationarity; $x_2=22$, $x_1=18$ from CS; feasibility residuals $h_1=0$, $h_2=-54+44+10=0$, $x_i\ge0$; $f^\star=3\cdot18+22=76$. All match the tutorial.
- **Quadratic example** (§13.13): $u^\star=2$, $x^\star=1$, $f^\star=1$; $g(u)=u-u^2/4$, $g(2)=1$; gap 0. Figure panel (b) plots exactly these curves/values.
- **Diet problem** (§13.14): primal $(2,1.5)$ feasible, value 220; dual $(10/3,20,0)$ feasible ($50\le50$, $80\le80$), value 220; certificate chain $220\le d^\star\le p^\star\le220$ valid. Complementary-slackness consistency cross-checked (binding primal constraints 1,2 $\leftrightarrow$ free $y_1,y_2$; slack primal 3 $\leftrightarrow$ $y_3=0$; $x_i>0$ $\leftrightarrow$ binding dual constraints).
- **Problem-set solutions**: P1 ($g=2\lambda-\lambda^2/4$, $\lambda^\star=4$, $d^\star=4=p^\star$), P5 ($9$ vs $\infty$), P6 ($g=\lambda-\lambda^2/2$, $d^\star=1/2=p^\star$ via Cauchy–Schwarz), P9 ($x^\star=(4,0)$, $u^\star=(2,0,1)$, $f^\star=8$; sanity-checked on the constraint line), P10 certificate chain — all recomputed by hand.
- **Cross-references**: every § ref verified against the actual chapter files — §10.6/10.8/10.9, §11.3/11.6/11.7/11.8, §12.7/12.9/12.10/12.11 exist; internal §13.x refs (13.2–13.15) all exist. Ch 32–33 (SVM) per OUTLINE.md.
- **Figure**: original matplotlib (`chapters/assets/13-duality-kkt.png`); panel (a) curves $L(x,\lambda)=x^2+\lambda(1-x)$ for $\lambda=0,2,4$ lie below $J(x)$ (spot-checked: at $x=2$, $L=4,2,0 \le J=4$); panel (b) $g(\lambda)=\lambda-\lambda^2/4$ max $1$ at $\lambda=2$. (Build quirk: this matplotlib rejects `\le`/`\ge` in mathtext — used `\leq`/`\geq` in the script only.)
- **Style**: `=` definitions, i)/ii)/iii), `Note:` callouts, `eg` blocks, "Basically, ..." after every complex topic (12 total), all math in LaTeX, no emojis, English.

## What was fixed during review

- §13.12 step iv): draft said "Try $h_1,h_2$ active ($u_1,u_3>0$...)" — wrong multiplier index; corrected to ($u_1,u_2>0$) before commit.
- Backslash scare: `muse.write` output was suspected of doubling `\` in LaTeX; verified byte-level with Python — file contains single backslashes throughout (the doubling was JSON display escaping). The one genuine `\\` is the intended line break inside `\begin{cases}` (§13.3). No change needed.

## Thin / contradictory source points (not invented; reported here)

1. **Week 10 is probability, not duality.** All of `PPT/Week 10/` and `Transcripts/Week 10/` cover MLF Chapter 6 (probability). The OUTLINE.md mapping "MLF W9–W10" for Ch 13 is in practice just W9; the chapter's scope was therefore derived from Week 9 alone. Flagging in case the outline's W10 tag misleads a future writer.
2. **Slater's condition is never named.** Sources say only "upto some regularity conditions" / "modulo regularity". The chapter states strong duality exactly so, with a Note that the standard literature calls these constraint qualifications (Slater's condition), cross-checked against standard references — kept to one sentence, clearly marked as beyond-lecture.
3. **SVM objective form.** The 9.5 slide writes $\min_w \tfrac12\lVert w\rVert^2$; the transcript says "minimise over $w$ of norm $w$ squared" (no $1/2$). Immaterial (positive rescaling), but the chapter follows the slide's $1/2$ form.
4. **No worked quadratic KKT example in sources.** The tutorial's canonical example is the LP (§13.12). §13.13's quadratic ($\min x^2$ s.t. $x\ge1$, with explicit dual) is constructed by applying the sourced machinery — a derivation, not a source claim.
5. **Diet-problem optima computed by the writer.** The tutorial gives the primal/dual pair but solves neither; the $(2,1.5)$ / $(10/3,20,0)$ / $220$ values are hand computations verified above (and reused as Problem 3).
6. **Sufficiency direction — CORRECTED by coordinator.** Lectures assert (not prove) that KKT $\Rightarrow$ local optimum in general; the chapter reported this faithfully as "asserted without proof". On independent review this claim is false in full generality: $\min_x -x^2$ s.t. $x \le 0$ satisfies all four KKT conditions at $(0,0)$, yet $x=0$ is a local *maximum*. Chapter fixed pre-merge: §13.10(ii) now states the standard result (KKT necessary for local optimum under CQ; sufficiency needs convexity, per (iii)), with the counterexample included.

## Concerns for the coordinator

- None blocking. The one judgment call to be aware of: §13.7's Slater's-condition Note goes one sentence beyond the lectures (standard-literature cross-check, as GOAL.md permits); if the house rule is "sources only", that sentence can be cut without loss.
