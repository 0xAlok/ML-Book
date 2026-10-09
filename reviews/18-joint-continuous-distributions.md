# Review log — Chapter 18. Joint continuous distributions

## Sources used (all committed under `sources/course materials/`)

1. `Stats2_VOL2_JOINTCONTSDISTRIBUTIONS.pdf` (Nikita Kumari & Mayur Gundal, IITM BS; "Statistics for Data Science - II"). Chapter 3 ("Jointly continuous random variables"): §3.1 2-D uniform distribution — Q1 triangle $D = \{x+y<2, x>0, y>0\}$ with $P(X+Y<1)=1/4$ via area ratio and via integration, $P(X+2Y>1)$ asked; Q2 $f_{X,Y}(x,y)=x+y$ on the unit square (validity, $P(X<1/2,Y<1/2)$, $P(X+Y<1)$); §3.2 marginal density — Q2 uniform on $[0,1/2]^2\cup[1/2,1]^2$ giving Uniform$[0,1]$ marginals; §3.3 independence — $X\sim\mathrm{Exp}(\lambda_1), Y\sim\mathrm{Exp}(\lambda_2)$ independent, joint density and $P(X>Y)=\lambda_2/(\lambda_1+\lambda_2)$; §3.4 conditional density — unit-square uniform conditionals, $f=x+y$ conditionals $f_{Y\mid X=a}(y)=(a+y)/(a+1/2)$; §3.5 problems (MCQs, used for scope mapping and numerics cross-check: Q3 marginals of $\frac65(x+y^2)$, Q4 $f_{Y\mid X=1}(1/4)=2$ on the $x=2y$ triangle, Q5 $P(1/5<X<2/5\mid Y=1/2)=11/75$, Q6/Q9 independence tests, Q7 $(X\mid Y=1)\sim\mathrm{Exp}(1)$, Q8 conditionals on $\{0<x<1, 0<y<3x\}$, Q10 $e^{-(x+y)}$ independence). All deck numerics used were recomputed independently.
2. `Stats2_VOL1_JOINTDISTRIBUTIONS.pdf` §§2.3.4–2.3.5 (integer-sum convolution, min/max) — scope pointers only, per the ch-17 review log; the continuous versions in this chapter were derived by hand.
3. Prior book chapters as cross-check anchors: §§15.12/15.15 (binomial theorem, Poisson PMF), §§16.7–16.10 (uniform/exponential/normal, 1-D CDF method and monotone formula), §§17.5–17.14 (joint toolkit), §9.7 (multivariable chain rule / Jacobian intuition).

## What was checked

- **All § cross-references (automated):** every `§N.M` in chapter + solutions resolves to a real section header in the actual chapter files — 0 missing. Every `eg N` text reference matches an existing `eg` header (1–11, sequential). Forward `Chapter 19/20/21` refs match OUTLINE.md (files not yet written — same convention as the ch-17 review).
- **Every derivation re-derived:** the CDF-method recipe; the convolution derivation $F_Z(z)=\int F_Y(z-x)f_X(x)\,dx \to f_Z = f_X * f_Y$ (Leibniz step); the triangular and Gamma$(2,\lambda)$ convolutions with mean cross-check $2/\lambda$; the discrete Poisson convolution via the binomial theorem; $F_{\max}=\prod F$, $F_{\min}=1-\prod(1-F)$; min-of-exponentials $\mathrm{Exp}(\lambda_1+\lambda_2)$; $P(X>Y)=\lambda_2/(\lambda_1+\lambda_2)$; both Jacobian determinants ($-1/2$ and $-u$) and both support transformations (diamond; $u>0, 0<v<1$); the marginal $f_U(u)=\min(u,2-u)$ matching the convolution triangle; the $U\perp V$ factorization with $V\sim\mathrm{Uniform}[0,1]$; the MGF product rule and the normal-sum identification.
- **Numerics (all recomputed independently, exact fractions + mpmath):** $1/4$, $7/8$ (deck's Q1; second value has no printed deck solution — computed here); deck Q2's $1/3$ ✓ but its $3/16 \to 1/8$ (see thin point 2); marginals $\frac65(x+1/3)$, $\frac65(\frac12+y^2)$ integrating to $1$; diagonal-squares marginals $=1$ with zero-cell $(1/4,3/4)$ dependence; $f_{\max}=2z$, $f_{\min}=2(1-z)$ integrating to $1$; $f_{U,V}=1/2$ on the diamond (area $2$); $f_U$ integrating to $1$; Poisson$(5)$ PMF sums to $1$ with convolution/direct agreement at $z=2$.
- **Figure:** original matplotlib (`chapters/assets/18-joint-continuous-distributions.png`, 1875×1275, matching chs 15–17); panels (a)–(c) verified against the rendered image and against the text (triangular $f_Z$, $2z$/$2(1-z)$, diamond with density $1/2$). Marked with the original-work HTML comment. No external image URLs used (HTTP-200 rule not triggered).
- **Scope boundary:** no multivariate normal content — the normal-sum example is a univariate distribution statement with an explicit forward pointer to Chapter 19; no MLE (§20); CLT only as a forward pointer (§21). Convolution/min-max/Jacobian/MGF are general two-variable machinery, per the task scope.
- **Style:** `=` definitions, i)/ii)/iii) points, `Note:` callouts, 11 `eg` blocks, "Basically, ..." after every complex section (6; §18.7 is summary, same convention as chs 16–17), all math in LaTeX, no emojis (byte-checked), English.

## What was fixed during review

- Wrong section pointer in eg 10: "§18.4's eg-4 density" → "§18.3's eg 4 density" (the Gamma density lives in §18.3).
- Vague pointer "(§9's multivariable chain rule)" → "(§9.7)".
- Inserted eg 5 (discrete Poisson convolution) shifted numbering: a scripted renumber hit the new eg 5 instead of the old one — caught on re-read; headers, all in-text `eg` refs, the figure caption, and all five solution `eg` pointers corrected and re-verified.
- Figure panel (c) rebuilt twice: the first inset version overlapped labels and clipped the title; final version uses a nested gridspec (square → diamond side by side with arrow), verified visually.

## Thin / contradictory source points (not invented; reported here)

1. **The VOL2 deck does not cover convolution, min/max CDFs, the Jacobian method, or MGFs.** These four topics (the core of this chapter) were built from standard results at the task's explicit direction — Chapter 17's §§17.8/17.15 promised them, and the task scope named them. Every formula was derived by hand from the CDF method and numerically verified; nothing was copied from an uncommitted source.
2. **Deck arithmetic error (§3.1.0.1 Q2).** The printed $P(X<1/2,\ Y<1/2)=3/16$ integrates $x/2$ to $x^2/2$ instead of $x^2/4$; the correct value is $1/8$ (verified two ways: direct integration and $E[x+y]$ over the quarter square). The chapter and solutions use $1/8$ and flag the slip inline.
3. **Deck's $P(X+2Y>1)$ (Q1) has no printed solution** in the extraction — only the question. Answered here as $7/8$ via the complement triangle (area $1/4$), verified by area arithmetic.
4. **MGF uniqueness and the normal MGF formula** are quoted as standard facts, not proved — a full justification is measure-theoretic and out of this book's scope. The chapter says so.
5. **The deck's §3.5 MCQs were used for scope/numerics cross-check only** (Q4 $=2$, Q5 $=11/75$, Q7 $=e^{-x}$, Q8 (b)+(c), Q10 (b)(c)(d) all recomputed and confirmed); only Q3's setup was adapted into the problem set, with full working.

## Concerns for the coordinator

- None blocking. Two judgment calls to confirm: (a) convolution, min/max, Jacobian, and joint-MGF sections are hand-derived standard material rather than deck-verbatim — the deck covers none of them, but the task scope and Chapter 17's forward promises required them; (b) the discrete convolution is given one worked example (Poisson+Poisson, eg 5) plus the stated formula, since §17.8 already contains a worked discrete-sum example — if the coordinator wants the discrete case expanded, it is one more eg.
