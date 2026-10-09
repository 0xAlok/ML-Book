# Review log — Chapter 1: Sets, functions, and mathematical preliminaries

Branch: `draft/sets-functions-preliminaries`. Reviewed against sources on 2026-10-09.

## Sources used

1. Primary: `sources/course materials/Maths1_VOL1_SETS&FUNCTIONS.pdf` (66 pages; pages 4–5 were image-only table of contents — rendered and confirmed to contain nothing substantive, so nothing was lost).
   - Used: §1.1–1.3 (number sets N/Z/Q/R, irrational examples), §1.4–1.4.2 (set definition, cardinality, subsets, proper subsets, set comprehension — extraction of the worked example was garbled, see below), §1.5 (Cartesian product, binary relations, reflexive/symmetric/transitive, equivalence), §1.6 (function definition, domain/codomain/range, injective/surjective/bijective, domain of √x, range of x²), §5.1–5.2 (vertical/horizontal line tests, increasing/decreasing ⇒ one-one, |x| not one-one), §5.9 (composition, domain rules, discount example, f(x)=3x−4 & g(x)=x² example, the 3/(x−1) & 3/x domain example — garbled, see below), §5.10 (inverse: one-one condition, f⁻¹(f(x))=x identities, f⁻¹ ≠ 1/f, graph symmetry about y=x, g(x)=4x & x/4 verification, x³/x^{1/3} verification).
   - Deliberately NOT used (belongs to later chapters per OUTLINE.md): Ch. 2 straight lines/distance/section formula/SSE, Ch. 3 quadratics, Ch. 4 polynomials, §5.3–5.8 exponentials, Ch. 6 logarithms, Ch. 7 exercise set (log/exp focused).
2. Secondary: `MLF-20261009T001559Z-1-001/MLF/` Week 2 — transcript PDF `Transcripts/Week 2/1. Sets and Functions.pdf` (clean text, primary secondary source), PPT PDF (extraction garbled, used only to confirm slide topics), handwritten notes `Notes/Notes by Aarthi A/W2_MLF.pdf` (scanned, no extractable text; page 1 rendered and confirmed to mirror the transcript).
   - Used: standard sets R, R₊, Z, Z₊; closed/open interval notation with set-builder form; Rᵈ as d-fold Cartesian product; universe, union/intersection/complement/set-difference notation (backslash); De Morgan's laws with the U=[0,10], A=[2,5], B=[4,7] worked verification; quantifiers ∀/∃, ⟹/⟺; functions as mappings with domain/codomain; real-valued and Rᵈ→R functions; graph of a function G_f ⊆ R^{d+1}; Venn-diagram intuition.
   - Deliberately NOT used (belongs to later chapters): metric spaces/open & closed balls (Ch. 8/9 continuity), sequences & limit definition (Ch. 8), vector spaces/dot product/orthogonality (Ch. 4/5), contour plots & heat maps (Ch. 9).

## What was double-checked

- Every worked example was re-derived by hand, not copied: De Morgan interval arithmetic (eg 3 / problem 3) recomputed endpoint-by-endpoint, including that 4 and 5 belong to neither complement in the second-law check; composition domains (eg 9) re-derived via both domain rules; inverse verifications (eg 10, problem 10) expanded term-by-term.
- Convention mismatches resolved explicitly in text: N includes 0 (Maths 1 §1.1.1); R₊/Z₊ include 0 per the MLF transcript (non-standard vs many textbooks — flagged with a "Note" in §1.2).
- "Increasing" follows the Maths 1 source definition (x₁ ≤ x₂ ⟹ f(x₁) ≤ f(x₂)) and its theorem (increasing/decreasing ⇒ one-one).
- The empty set is never mentioned (absent from sources); the power set is never mentioned (absent); counting formulas like |A×B| = |A|·|B| are never stated as theorems (absent — problem 4 asks only to list and count).
- Cross-check anchors (mml-book, mecmath.net): consulted only to confirm standard conventions (injective/surjective/bijective, De Morgan) agree with the sources. Nothing copied from them.

## Discrepancies / garbled-source issues

1. Maths 1 §1.4.2 (set comprehension): the worked example did not extract (font-encoding corruption). Grounded set-builder notation in the MLF transcript's interval notation instead; the "generator/filter/transformer" framing from the source was not reproduced since its example was unreadable.
2. Maths 1 §5.9 composition-domain example (f(x)=3/(x−1), g(x)=3/x): the formula extraction was garbled ("32−xx x"), but the domain conclusion R\{0,3} extracted cleanly. Reconstructed the algebra as (f∘g)(x) = 3x/(3−x) and verified it satisfies both stated domain exclusions. Used as eg 9.
3. Maths 1 §5.10 inverse-verification example (f(x)=(2x−5)/(x+3) vs g(x)=(3x+5)/(1−2x)): extraction too garbled to recover the exact fractions reliably. Not reused; wrote a fresh equivalent example (f(x)=2x+3) with full verification instead.
4. MLF transcript's R₊ definition sentence is itself garbled ("set of real numbers, including 0 positive real numbers, including 0"); interpreted as non-negative reals including 0, consistent with the handwritten notes. Flagged in-chapter as a convention note.

## Omissions (marked in the chapter, not silently dropped)

- Proof techniques (direct, contrapositive, contradiction, induction): requested in the brief but absent from both source files. §1.12 states this explicitly and defers proof ideas to point of use. Do not add without a source.
- Metric spaces/balls, sequences/limits, vector spaces, contour plots, exponentials/logarithms: present in sources but assigned to later chapters by the outline; listed here so the omission is auditable.

## Uncertainties

- The two matplotlib figures are original illustrations drawn for this chapter (Venn shading for De Morgan's first law; arrow diagrams for injective/surjective/bijective). They are not from any standard resource; the HTML comments above each image say so. If the project later prefers standard-source figures with URL attribution, these can be swapped.
- §1.15 "Where this goes next" references planned chapters (2, 3, 4/5, 8–10, 23, 31) from OUTLINE.md — these are forward pointers, not source claims.
