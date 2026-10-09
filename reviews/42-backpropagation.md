# Review log — Chapter 42: Backpropagation, worked end-to-end

Branch: `draft/backpropagation`. Self-review pass completed 2026-10-09. Not merged — awaiting independent review.

## Sources actually used

1. **GenAI "Introduction to ANN" deck** (Balaji Srinivasan, Ganapathy Krishnamurthi) — `sources/course materials/GenAI-20261009T001550Z-1-001/GenAI/Slides/Week - 1 & 2/Introduction to ANN - Lecture Slides.pdf`, slides 59–70: the chain-rule engine, computational graph, scalar hand example, and the full MLP weight-update derivation (three gradient components, δ recursion, parameter gradients). Text extracted via pdftotext; slide 64 verified via `pdftoppm` render.
2. **XOR Problem notebook** — `.../GenAI/Slides/Week - 1 & 2/XOR Problem.ipynb`, cells 11/14/17/18: hand-calc weights, the full backward pass by hand, the `backward` implementation, and its verification. Numbers transcribed as [recorded], every one re-run in numpy as [verified-NumPy].
3. **`GenAI/Notes/Week-1,2-ANN.pdf`** — byte-distinct from the deck (md5 differs: `d1489897...` vs `e03ee6a2...`) but verified to carry the same content; no unique backprop material found. Not cited beyond this note.
4. **MLT Week-12 ANN slides** — name backprop as the sklearn trainer; add no new derivation. Cited once for that fact.
5. Book chapters as anchors: §2.10 (matmul), §8.3 (derivative-as-limit), §9.7 (multivariate chain rule), §10.5/§10.7 (GD), §31.10/§31.12 (logistic gradient, softmax), §36.6 (seeding), §40.5(i) (debugging playbook), §41.4/§41.7/§41.9/§41.10/§41.11/§41.13/§41.14. Every §-reference was checked against the actual file — §9.7 (not §10.7) holds the multivariate chain rule; the task brief's "§10.5–10.7" shorthand was corrected in the chapter.

## Re-run / recorded ledger

| Item | Status | Evidence |
|---|---|---|
| Deck slide 64 scalar example forward/backward | [recorded], values corrected (deck typo) | PNG render verified |
| XOR backward pass, all 9 gradients + α=0.1 update | [verified-NumPy] | `/tmp/reverify_42.py` output |
| Gradient check: analytic vs centered finite differences, 9 params | [verified-NumPy], max |diff| = 8.18e-10 | same script |
| BCE+sigmoid per-example cancellation (−2.0387×0.2499 = −0.5095) | [verified-NumPy] | same script |
| Mean-loss BCE+sigmoid δ = (g−y)/m vs FD | [verified-NumPy], 5.0e-10 | separate numpy run |
| Softmax+CE δ = ŷ−y vs FD (4-class) | [verified-NumPy], 1.17e-9 | separate numpy run |
| Problem 4 (0,1) forward/backward | [verified-NumPy] | solutions build script |
| Problem 6 softmax numbers, Problem 7 1-1 net + FD, Problem 9 sigmoid'(3)^4 | [verified-NumPy] | solutions build script |
| Notebook `backward` code, hand-calc weights, cell-18 insight quotes | [recorded] | cells 11/14/17/18 extracted |
| torch snippets | none — no torch code in this chapter | n/a |

## Gradient-check results (the high-value check)

Centered finite differences (ε=1e-7) against the hand-derived backward pass for the §42.3 net, all 9 parameters: analytic `[−0.200336, 0.127367, 0, 0, −0.200336, 0.127367, −0.372452, −0.254735, −0.509470]` vs finite-difference (identical to 6 s.f.), max absolute difference **8.18e-10**. The two extra pairing checks (softmax+CE, mean-BCE) agree to ~1e-9. Conclusion: the chapter's derivations are the true gradients.

## Thin / contradictory points

1. **Deck typo, slide 64:** "𝓛 = d·c = 5·4 = 30" with c=6 — should be 5·6=30. Flagged in a chapter Note; corrected numbers used everywhere; gradients unaffected.
2. **Notebook's ∂L/∂ŷ = −2.037** is a rounding artifact (−1/0.491, after rounding ŷ to 0.491); exact is −1/0.4905 = −2.0387. Flagged in §42.3; chapter uses exact values.
3. **Notes PDF vs deck:** byte-distinct files, same content — checked, not assumed.
4. **The (0,1) forward pass is numerically identical to the (1,0) one** (hand-picked weights make x₁+x₂=1 on both y=1 corners). Not a bug; noted in the Problem 4 solution.
5. **Softmax+CE cancellation derivation** is the book's own (standard) derivation, not the notebook's — the notebook only covers sigmoid. Cross-checked numerically (1.17e-9 agreement) and against §31.12.

## Fixes made during review

- §42.5 Note: pairing labels corrected — the cancellation applies to pairings (i) sigmoid+BCE and (ii) softmax+CE; (iii) identity+squared has nothing to cancel (text had (ii)/(iii) vs (i) swapped).
- §42.6: "§8.5's derivative-as-limit" → §8.3 (verified against the actual section titles); the debugging-playbook reference changed from §40.4 to §40.5(i) (verified: §40.4 is the "suspiciously good" checklist, §40.5(i) is the shapes-first playbook).
- Problem 5(iii): vague "(the §41.7/NOTE case)" → "(the §42.5 Note case)".
- Problem 8 solution: qualified that post-update deltas shift slightly from Problem 4's pre-update numbers.
- Figure `assets/42-backprop-graph.png`: three layout iterations to eliminate label collisions; final render verified readable.

## Files

- `chapters/42-backpropagation.md` — the chapter (8 sections + problem set)
- `solutions/42-backpropagation.md` — full worked solutions, separate volume
- `chapters/assets/42-backprop-graph.png` — original matplotlib figure (computational graph, forward values green / gradients red), HTML comment attributes it as drawn for this chapter after the deck's slide 64
- `reviews/42-backpropagation.md` — this log

## Concerns for independent review

1. The softmax+CE cancellation derivation (§42.5(ii)) is book-original — please re-derive independently; the numerical check agrees to 1.17e-9.
2. The batch-form bias gradient presentation (`∑_rows Δ^l` with the 1/m folded into Δ) vs the notebook's "divide at the end" — both are shown equivalent; confirm the wording won't confuse.
3. Problem 4's identical-to-(1,0) forward pass — intentional (source weights), flagged; consider whether a reviewer wants a more distinct second example.
