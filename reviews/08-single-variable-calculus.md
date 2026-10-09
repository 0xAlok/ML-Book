# Review log — Chapter 8: Single-variable calculus

## Sources used

1. **Primary:** `sources/course materials/Maths1_VOL2_CALCULUS.pdf` (99 pp., "Mathematics for Data Science — Calculus of one variable"). Read via `pdftotext` to `/tmp/vol2.txt`. Used: §1.5 (limits, one-sided limits, sequential definition, limit at infinity, Examples 1.5.1–1.5.5: $x^2$ at $1$, floor at $-1$, Dirichlet-type at $\sqrt2$, $\sin(1/x)$ at $0$, $1/x$ at $\infty$), §1.6 (continuity Defs 1.6.1–1.6.2, Examples 1.6.1–1.6.4: $|x|$ at $0$, floor at $-1$, piecewise jump at $2$, $\sin(x^2)/(x^2-1)$ at $0$; Theorems 1.6.1–1.6.2), §2.1 (differentiability Def 2.1.1–2.1.2, truck average-vs-instantaneous example, Examples 2.1.2–2.1.6, 2.1.7–2.1.15, Theorem 2.1.1 with proof, Propositions 2.1.2–2.1.4, standard-derivative table), §2.3 (tangent as limit of secants, equation $y=f(a)+f'(a)(x-a)$, linear approximation Def 2.3.1, Examples 2.3.1–2.3.4: $\cos x$ tangent at $\pi/3$, $x^3$ at $1$, $\sec x$ at $0$), §2.4 (turning points, Theorem 2.4.1 with proof, critical/saddle points Defs 2.4.1–2.4.2, second derivative test, Examples 2.4.4, 2.4.7, 2.4.8: $x^3-12x$, $x^2$ on $[-1,1]$, Theorem 2.4.2).
2. **Secondary (MLF Week 2):** Transcript `Transcripts/Week 2/3. Univariate Calculus- Derivatives and Linear Approximations.pdf` — the linear-approximation derivation ($f'(x^*)\approx \Delta f/\Delta x \Rightarrow f(x)\approx f(x^*)+f'(x^*)(x-x^*)$), tangent-vs-linear-approximation distinction, worked list ($\sin x\approx x$, $e^x\approx1+x$, $\ln(1+x)\approx x$, $(1+x)^r\approx1+rx$, all at $0$), the $0.99^7\approx0.93$ MCQ, and the "key expression in first-order calculus… will drive most of optimization" framing. PPTs 2–4 text-extracted with `pdftotext` (both `muse.read` and `pdftotext` render them garbled — they are image/math-heavy slides); salvageable fragments confirm: sequential continuity definition ("for all sequences converging to $a$, $f(x_n)$ converges to $f(a)$"), differentiability at $x$ via the difference-quotient limit, "$f$ not continuous at $x \Rightarrow$ not differentiable at $x$", and $|x|$-at-$0$ as the worked non-differentiable example.
3. **Chapter 1 cross-reference:** `chapters/01-sets-functions-preliminaries.md` §1.11 ($\forall$/$\exists$ table and "Basically" line) — the $\varepsilon$–$\delta$ definition is written to hook into that notation.
4. **Cross-check anchors (correctness only, nothing copied):** web check of the $\varepsilon$–$\delta$ limit formulation against mecmath-linked material and standard references — the chapter's statement matches the standard form.

## Source gaps handled

- **Transcript PDFs 2 and 4 are 0-byte files in the repo** ("2. Univariate Calculus- Continuity and Differentiability.pdf" and "4. Univariate Calculus- applications and advanced rules.pdf" under `Transcripts/Week 2/`). Continuity/differentiability MLF-deck content was recovered from PPT 2 fragments + the primary book; the deck-4 topics (product/chain rules, higher-order approximations) are covered by the primary book's Propositions 2.1.3–2.1.4.
- **The primary book defines limits via sequences, not $\varepsilon$–$\delta$.** The $\varepsilon$–$\delta$ definition was added as standard material (task-required, cross-checked); the chapter explicitly notes the equivalence with the source's sequential form rather than pretending it came from the book.
- **Intermediate value theorem:** the book only invokes it in an exercise hint (Ex 1.2.1, range of $x^3+5$). Included as one paragraph, flagged as such; the root-guarantee example ($x^3+x-5$ on $[1,2]$) is original.
- **Concavity/inflection:** the book names "inflection point" once (in the second-derivative-test list) and notes $f''$ checks monotonicity of $f'$. Covered briefly per the task's "brief" instruction; nothing beyond that was invented.

## What was double-checked

- Every derivative recomputed: $(x^2)'=2x$, $(\sin x)'=\cos x$ (limit steps), $(x^7\sin x)'=7x^6\sin x+x^7\cos x$, $(\tan x)'=\sec^2x$ (numerator $\cos^2+\sin^2=1$), $(\tan2x)'=2\sec^2 2x$, $(\sin x^2)'=2x\cos x^2$, $(3x^3-2x^2+x-5)'=9x^2-4x+1$, $(1/x)'=-1/x^2$ (quotient steps), $((2x^3-1)^4)'=24x^2(2x^3-1)^3$, $(x^2e^x)'=xe^x(x+2)$, $(x/(x^2+1))'=(1-x^2)/(x^2+1)^2$.
- Every tangent/linear approximation: $\cos x$ at $\pi/3$ $\to$ $y=-\frac{\sqrt3}{2}(x-\pi/3)+\frac12$; $x^3$ at $1$ $\to$ $3x-2$; $\sec x$ at $0$ $\to$ $1$; $0.99^7\approx0.93$.
- $\varepsilon$–$\delta$ proof for $x^2$ at $1$: $\delta=\min(1,\varepsilon/3)$ verified; Solution 1 ($\delta=\varepsilon/2$) verified.
- Max/min: $x^3-12x$ critical points $\pm2$, $f''(\pm2)=\pm12$, values $f(2)=-16$, $f(-2)=16$; global check on $[-3,3]$: $f(-3)=9$, $f(3)=-9$ — max $16$ at $-2$, min $-16$ at $2$. Solution 11: $f''(-1)=-4$, $f''(1/3)=4$, $f(1/3)=130/27\approx4.815$ (matches book's Ex 2.4.6 value $4.814$ approx.). Solution 12 ($x^3-3x$ on $[-2,2]$): values $-2,2,-2,2$ — max $2$ at $-1,2$; min $-2$ at $-2,1$. Solution 9: $f(2)=4$, $f'(2)=10$, $y=10x-16$. Solution 10: $\sqrt{1.02}\approx1.00995$, error $\approx5\times10^{-5}$. Solution 3: $3/2$. Solution 7(b): $(1-x^2)/(x^2+1)^2$.
- Figure `chapters/assets/ch08-secant-tangent-abs-corner.png` rendered and visually inspected: secants $h=0.9,0.45,0.18$ converge to $y=2x-1$; $|x|$ panel shows the corner with slopes $-1$/$+1$.

## Discrepancies / judgment calls

- The book's truck example says the $260$ km Maharashtra stretch was covered in "4 hours with about an hour's break", i.e. $3$ driving hours $\to$ $260/3\approx86.67$ km/h — the chapter follows the book's arithmetic (driving time, not total time).
- First-derivative (sign-change) test is not stated verbatim in the book, but follows directly from its turning-point definition ("increase ending at $x$ and decrease beginning at $x$"); stated as the sign-change reading of the same idea.
- Increasing/decreasing via the *sign* of $f'$ is stated as the working rule; the book defines monotonicity interval-wise and says "$f'$ checks the monotonicity of $f$". Standard and cross-checked.
- ReLU mention in §8.4 is a one-sentence ML pointer, not sourced content — clearly framed as a pointer.

## Deliberate omissions

- **L'Hôpital's rule (§2.2)** — not in the task's content list; omitted.
- **Higher-order approximations (PPT 4)** — Taylor territory; out of scope.
- **Multivariate calculus (MLF decks 5–6)** — reserved for Chapter 9.
- **Optimization/gradient descent** — reserved for Chapter 10 (only forward pointers).
- **Integration (book ch. 3)** — not part of this chapter's brief.
- **Mean value theorem** — the task asked for it "if present"; it is not present in the sources, so it is omitted rather than invented.

## Uncertainties

- None material. The garbled PPT extraction means deck-4's exact worked examples could not be recovered, but all of them are covered by equivalent book examples used above.
