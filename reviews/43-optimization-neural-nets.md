# Review log — Chapter 43: Optimization for neural nets

Branch: `draft/optimization-neural-nets`. Self-review pass completed 2026-10-09. Not merged — awaiting independent review.

## Sources actually used

1. **GenAI Week 5 "Optimization Strategies" deck** (Balaji Srinivasan, Ganapathy Krishnamurthi) — `sources/course materials/GenAI-20261009T001550Z-1-001/GenAI/Slides/Week - 5/Optimization Strategies - Lecture Slides.pdf`, 78 slides, text extracted via `pdftotext` (fully extractable, no render needed): ravine/saddle-point challenges, the ill-conditioned demo $f = 0.1x_1^2+2x_2^2$ (slide 58, GD $\eta = 0.4$, momentum $\eta = 0.6$), convex primer (weight decay as $\tfrac{\lambda}{2}\lVert w\rVert^2$ penalty), batch/SGD/mini-batch theorem + vectorization, five LR schedules + warmup, momentum theorem + leaky-average derivation ($1/(1-\beta)$ effective step), AdaGrad/RMSProp/AdaDelta/Adam theorems, closing summary ("minibatch SGD is the workhorse", "Adam is a robust, default choice").
2. **Gradient_Descent.ipynb** — `.../Week - 5/Gradient_Descent.ipynb`, 39 cells extracted via python json: BGD/SGD/mini-batch implementations and trade-off tables, batch-size experiment ($1,8,32,64,128,256$ at $\eta = 0.01$), LR sensitivity ($0.0001$–$0.1$), PyTorch DataLoader batch-size vocabulary, manual-vs-PyTorch comparison. All torch/pandas code transcribed as [recorded] (torch not installed; plots not re-run).
3. **Advanced_Optimization.ipynb** — `.../Week - 5/Advanced_Optimization.ipynb`, 66 cells extracted via python json: optimizer implementations and update equations (Vanilla GD, momentum, **Nesterov**, AdaGrad, RMSProp, AdaDelta, Adam, **AdamW** with the decoupled-decay derivation), LR starting-point table, scenario selection guide, pitfalls table, 2025 best practices (AdamW default, gradient clipping snippet, mixed precision). Rosenbrock/Beale/neural-net comparison cells read but their numeric results not cited (plots not re-runnable here). All torch code [recorded].
4. **GenAI Notes `Week5-Convex Optimization,Gradient Decent.pdf`** — text-extracted and diffed against the deck: same content modulo page-ordering noise, plus one line ("An optimization algorithm's updates are based on local gradient information"). No unique material; not cited beyond this note.
5. Book chapters as anchors, every §-reference verified against the actual files: §10.5 (GD update, $\alpha$), §21 (variance-of-average logic), §28 (ridge $\lambda/2$ penalty, $\nabla J = X^TXw-X^Ty+\lambda w$), §36.6 (seeding), §37.3 (StandardScaler sermon verbatim, `invscaling` $\eta_0/t^{\text{power\_t}}$, SGDRegressor knobs), §40.5(i) (debugging playbook verbatim), §41.7 (activation table), §41.10 (numpy class), §41.14(i) (zero-init symmetry verbatim), §41.15(ii) (this chapter's promise), §42.4 ($\delta$ recursion), §42.8(ii) (vanishing-gradient reading).

## Re-run / recorded ledger

| Item | Status | Evidence |
|---|---|---|
| GD $\eta=0.4$ 8-step trace on $0.1x_1^2+2x_2^2$ from $(-4,1)$ | [verified-NumPy] | `/tmp/w5/demo/demo.py` |
| GD $\eta=0.6$ 6-step divergence (f: 3.6 → 113.73) | [verified-NumPy] | same script |
| Momentum $\eta=0.6,\beta=0.9$ 30-step trace | [verified-NumPy] | `/tmp/w5/demo/demo2.py` |
| Steps to $f<10^{-6}$: GD(0.4)=86, Momentum=120, AdaGrad(2.0)=13, RMSProp(0.1)=64, Adam(0.01)=1227 | [verified-NumPy] | demo.py + demo2.py |
| RMSProp $\eta=0.01$ crawls / $\eta=0.5$ oscillates | [verified-NumPy] | demo2.py |
| Velocity weight sum $\sum_{k=0}^{60}0.9^k = 9.984 \approx 10 = 1/(1-0.9)$ | [verified-NumPy] | demo.py |
| Five-schedule table at $t=0,100,500,1000$ | [verified-NumPy] | demo.py |
| Adam one-step: $\hat v_1=0.5,\ \hat s_1=0.25$, step $=1.0\eta$; uncorrected $3.162\eta$ | [verified-NumPy] | demo.py |
| All optimizer update equations, tables (LR ranges, selection guide, pitfalls, best practices), Nesterov reformulation, AdamW derivation, `clip_grad_norm_` snippet | [recorded] | notebook JSON extraction |
| `assets/43-ill-conditioned-trajectories.png` | original matplotlib figure, drawn from the re-run traces | `/tmp/w5/demo/fig.py` |

## Thin / contradictory points

1. **No weight-initialization schemes in any GenAI W5 source.** The deck, both notebooks, and the notes never name Xavier/He or any variance-scaling rule; the deck's only init-adjacent line is the pitfalls table's "poor initialization → use proper initialization" [recorded]. §43.10 therefore derives the *criterion* (keep $z$ where $g'(z)$ is healthy, via the §42.8(ii) $\delta$ recursion) from book-internal material and explicitly flags the missing formulas instead of inventing them.
2. **Batch normalization is used but never explained.** `nn.BatchNorm1d` appears in the Advanced notebook's classifier architecture with no formula or discussion; the deck is silent. §43.9 covers only input/feature scaling (§37.3) and flags batch norm as absent, proposing it as a later-pass topic with Chapter 44.
3. **Nesterov is notebook-only.** The deck covers momentum but not Nesterov; §43.4 says so explicitly and marks the equations [recorded].
4. **Notes PDF ≈ deck.** Byte-distinct files, same content — verified by diff, not assumed (same check as ch42's log item 3).
5. **sklearn `invscaling` vs deck inverse-time.** sklearn's $\eta_0/t^{\text{power\_t}}$ (§37.3) vs the deck's $\eta_0/(1+kt)$ — same $1/t$ family, differing only in the $+1$ (finite at $t=0$). §43.8's Note states the difference honestly rather than equating them.
6. **Momentum is not the step-count winner on the demo problem.** At its best $\eta$, plain GD (86 steps) beats momentum (120 steps) here; momentum's verified win is *surviving* $\eta = 0.6$ where GD diverges, and AdaGrad (13 steps) wins outright via preconditioning. §43.11's Note says this explicitly so the chapter doesn't overclaim the deck's "momentum fixes oscillation" story.
7. **Adam's slowness on the demo (1227 steps at $\eta = 0.01$)** is a property of the default LR on this toy problem, not a verdict on Adam. §43.11's Note frames it as "robustness across problems, not speed on this one."

## Fixes made during review

- §43.5(iii): AdaDelta boxed equations checked term-by-term against the deck's 4-step theorem (the $\Delta x_{t-1}$ update-average maintenance is described in prose rather than boxed; the three displayed lines match the slide).
- §43.8: polynomial-decay formula written as $\eta_0(\beta t+1)^{-\alpha}$ after re-reading the slide's garbled extraction ("$\sqrt{\ }$" was the $\alpha = 0.5$ case: $\eta_0/\sqrt{\beta t+1}$); the eg's numbers use $\alpha = 0.5, \beta = 1$ consistently.
- Solution 8(ii): $(0.0451)^3$ recomputed as $9.2\times 10^{-5}$ (was going to be written loosely as "$\sim 10^{-4}$" — now exact).
- §43.12 troubleshooting table: "Adam generalizes poorly" row sourced to the notebook's exact cause/fix ("Weight decay interaction → Switch to AdamW"), not invented.
- Problem 10(iii): weight decay $0.001$ chosen from the notebook's *fine-tuning* row ($0.0001$–$0.001$), not the vision/NLP rows — checked against the table.
- Figure: axis limits fixed so the diverging GD trace doesn't dominate; legend labels match the §43.11 eg settings verbatim.

## Files

- `chapters/43-optimization-neural-nets.md` — the chapter (13 sections + problem set)
- `solutions/43-optimization-neural-nets.md` — full worked solutions, separate volume
- `chapters/assets/43-ill-conditioned-trajectories.png` — original matplotlib figure (contours + three traced trajectories), HTML comment notes it was drawn for this chapter, not reused from any URL
- `reviews/43-optimization-neural-nets.md` — this log

## Concerns for independent review

1. **§43.10's init-scale argument is book-original** (criterion from §42.8(ii), no course formula). Please confirm the framing "criterion stated, gap flagged" is the right call vs. omitting the section — the chapter title promises "init" and §41.15(ii) points here.
2. **Batch norm punted** (§43.9 + concern above): confirm the one-paragraph flag is sufficient and the "later pass with Chapter 44" proposal is acceptable.
3. **The steps-to-$10^{-6}$ comparison** (eg 6): hyperparameter choices are the deck's/notebook's own ($\eta = 2.0$ AdaGrad from the deck's demo; notebook LR table for the rest) — confirm the table reads as illustrative, not as a benchmark claim.
4. **AdaDelta's boxed equations** condense the deck's 4-step theorem to 3 lines; verify nothing load-bearing was dropped.
5. **Problem 4's RMSProp numbers** ($\approx 0.3953/0.0791$ effective LRs) — hand-computed; please spot-check.
