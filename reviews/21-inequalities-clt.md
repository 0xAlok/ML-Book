# Review log — Chapter 21: Inequalities and the central limit theorem

Branch: `draft/inequalities-clt`. Reviewer: chapter author (self-review pass, 2026-10-09).
Sources re-read before writing: MLF transcript "Laws of large numbers Markov, Chebyshev,
Hoeffding, Central limit.docx.pdf" (full read), `week12-part4.pdf` (pdftotext; slides are
image-based, OCR garbled — transcript used as the authority), `Week 12 tutorials_sol.pdf`
(Q1 binomial–Markov, Q4 geometric–Chebyshev), MLF `MLF_PAS_W12.pdf` (Q1 coin–Markov,
Q2 binomial–Chebyshev, Q8 kσ–Chebyshev). MLE/multivariate-normal/GMM parts of the Week-12
deck were deliberately not used (Chapters 19/20/25 territory).

## What was checked

1. **Every derivation re-derived by hand**: Markov proof (split integral at $t$, drop
   $[0,t)$ part, replace $x$ by $t$ on $[t,\infty)$); Chebyshev as Markov on
   $(X-\mu)^2$; WLLN via Chebyshev on $\bar{X}_n$ with
   $\mathrm{Var}(\bar{X}_n) = \sigma^2/n$; $\mathrm{Var}(Y_n) = \sigma^2$ for the CLT
   scaling; the $k\sigma$ rule; Hoeffding formula
   $2\exp(-2n\varepsilon^2/(b-a)^2)$ matched against the transcript's wording
   ("2 times exp -2n epsilon squared by b-a the whole square").
2. **All numerics recomputed independently** (Python cross-check, all match):
   eg 5 peak $1/(\sigma\sqrt{2\pi}) = 1.3819$ with $\sigma = 1/\sqrt{12}$;
   §21.8: $1-\Phi(2) = 0.02275$, Markov $50/60 = 0.8333$, Chebyshev $25/100 = 0.25$,
   exact $\sum_{k=60}^{100}\binom{100}{k}/2^{100} = 0.02844$;
   Problem 3: $(1/4)/(8\cdot1/16) = 0.5$; Problem 4: $E = 2$, $\mathrm{Var} = 2$,
   $P(X\le 4) = 15/16$; Problem 6: $n \ge 10{,}000$; Problem 7: $1-\Phi(1.5) =
   0.06681$; Problem 8: $\Phi(-1.6) = 0.05480$ vs exact $0.06661$ (error $0.0118$,
   $\approx 18\%$ relative); Problem 9: $E[X] = t\cdot\mu/t = \mu$.
3. **Every § cross-reference verified programmatically against the actual files**
   (script extracted all `## X.Y` headers from `chapters/*.md` and all § refs from
   the chapter): §15.5/§15.7/§15.8/§15.9/§15.11/§15.12/§15.13, §16.4/§16.7/§16.9,
   §17.11, §19.9, §20.1/§20.6/§20.7/§20.8/§20.10/§20.12, §21.2–§21.9 — all valid.
   Forward refs ("Ch 23") match OUTLINE.md; the Part II closer mirrors §13.15's
   "Part I, in one paragraph" shape.
4. **Figure inspected visually**: n=1 flat, n=2 triangular, n=3 rounded, n=10
   matching the red $N(0,1/12)$ curve; marked original-matplotlib in the HTML comment.

## What was fixed during review

- Nothing substantive: the draft's derivations and numerics all checked out on the
  first re-read. One wording tightening: the CLT theorem header now reads
  "quoted, not proved" so no reader mistakes the statement for a proved result.

## Thin / contradictory source points

1. **No CLT proof in the sources.** The lecture states the CLT and moves to
   intuition + the uniform demo; the deck has no proof either. The chapter states
   it as quoted (header says so explicitly). Flag: the ch-21 author should not be
   cited as a source for a CLT proof.
2. **No continuity correction in the sources.** The binomial CLT example uses the
   plain $z = (60-50)/5$ standardization; neither the transcript, deck, tutorial,
   nor practice assignment mentions the $\pm 0.5$ correction. The chapter omits it
   and says so (§21.8, Problem 8). The approximation is still decent (0.0228 vs
   0.0284; Problem 8: 0.0548 vs 0.0666).
3. **Hoeffding stated, not proved — as in the lecture.** The inequality is quoted
   verbatim in form; the chapter mirrors the lecture's "stated, not proved"
   treatment and does not derive it or use it in any problem.
4. **Strong law omitted deliberately.** The transcript explicitly sets almost-sure
   convergence aside ("we will not be worried about that"); the chapter notes the
   omission in §21.1 rather than inventing coverage.
5. **week12-part4.pdf slides are image-based** (pdftotext output garbled); all
   statements were taken from the transcript, which covers the same lecture
   1:1. No contradictions found between the two.
