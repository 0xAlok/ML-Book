# Review log — Chapter 46: Attention and the transformer

Branch: `draft/attention-transformers`. Self-review pass completed 2026-10-09. Not merged — awaiting independent review.

## Sources actually used

1. **GenAI Weeks 9–10 deck "Introduction to Attention and Transformer Architecture"** (Balaji Srinivasan, Ganapathy Krishnamurthi), 99 slides — `sources/course materials/GenAI-20261009T001550Z-1-001/GenAI/Slides/Week - 9 & 10/Introduction to Attention and Transformers - Lecture Slides.pdf`. Text-extractable via `pdftotext` (no `pdftoppm` needed); the 231-page PDF is the same 99 slides repeated with beamer overlays.
2. **Notes `Week 9-10.pdf`** — `.../GenAI/Notes/Week 9-10.pdf`, 20 pages, `pdftotext`-extracted to `/tmp/w910.txt` (2620 lines). Used as the working copy; content verified identical to the deck slide-for-slide (same "N / 99" numbering, same examples, same figures). Cited as "the deck" throughout.
3. **machine_translation.ipynb** — `.../Week - 9 & 10/`, 89 cells extracted via python json: Multi30k DE→EN pipeline, Bahdanau `Attention` class (energy/attention/context formulas), bidirectional-GRU encoder, GRU decoder, teacher forcing $0.6$, 20 epochs; test loss $3.445$ / PPL $31.359$ / BLEU $0.3203$ (BLEU-1 $0.6327$ → BLEU-4 $0.1672$). All torch code and numbers transcribed as [recorded] (torch not installed; nothing re-run).
4. **HuggingFace Inference.ipynb** — `.../Week - 9 & 10/`, pipeline API demos: sentiment (default + Twitter-RoBERTa), NER (dbmdz/bert-large-cased-finetuned-conll03-english), MT (google-t5/t5-base en→fr, dedicated en→de), Gemini API calls. All [recorded].
5. **Week 12 Transformers.ipynb** — `.../Slides/Week - 12/`, nanoGPT: 19-token word-level vocab, context length 12, $d_{\text{model}} = 20$, 2 layers, 2 heads, dropout $0.2$, AdamW $10^{-4}$, 10{,}000 iters, batch 64; `Head`/`MultiheadAttention`/`TransformerBlock` (pre-norm, $4\times$ FFN with ReLU) classes, causal mask with $-\infty$, top-$k$ sampling; generation samples ("i like sour so i like" → lemon 99.46%; "i like juicy so i like" → apple 61.55% / orange 37.68%; "i like spicy so i like" → orange 64.83% / chili 33.29%). All torch code and numbers [recorded].
6. **Diagram references only (not fetched)**: Vaswani et al. 2017 Fig. 1 (https://arxiv.org/abs/1706.03762) and Dosovitskiy et al. 2020 Fig. 1 (https://arxiv.org/abs/2010.11929) — both cited by the deck's own figure captions; included as HTML comments per the book's reuse rule.
7. **Book chapters as anchors**, every §-reference verified against the actual files: §40 (§40.5(i)), §42 (§42.4), §43 (§43.9, §43.12), §44 (§44.6, §44.9, §44.10), §45 (§45.1, §45.7, §45.10, §45.11, §45.13).

## Re-run / recorded ledger

| Item | Status | Evidence |
|---|---|---|
| §46.6 attention eg: scores $\begin{smallmatrix}2&1\\1&2\end{smallmatrix}$, scaled rows $[1.1547, 0.5774]$, weights $[0.6405, 0.3595]$, outputs $(1.6405, 1.3595)$ / $(1.3595, 1.6405)$ | [verified-NumPy] | exec re-run, matches to 4 decimals |
| §46.8 masked softmax: $[2.0, 1.0, -10^6] \to [0.7311, 0.2689, 0]$ | [verified-NumPy] | exec re-run |
| §46.10 PE: $PE(0) = [0,1,0,1]$; $PE(1) = [0.8415, 0.5403, 0.0100, 1.0000]$; rotation identity checked numerically | [verified-NumPy] | exec re-run |
| §46.11 layer norm: $[1,2,3,4] \to \mu = 2.5, \sigma^2 = 1.25, \hat x = [-1.3416, -0.4472, 0.4472, 1.3416]$ | [verified-NumPy] | exec re-run |
| §46.14 patches: $N = 196$, conv params $590{,}592$, 197 tokens with `[CLS]` | [verified-NumPy] | exec re-run; deck's $(224^2)^2 \approx 2.5\times10^9$ confirmed |
| §46.9 multi-head params: $8 \times 98{,}304 + 262{,}144 = 1{,}048{,}576 = 4 \cdot 512^2$ | [verified-NumPy] | exec re-run |
| §46.12 complexity: $T{=}1000, d{=}512$: $5.12\times10^8$ vs $2.62\times10^8$ | [verified-NumPy] | exec re-run |
| All solution numbers (P1 $[1.3,1.3]$; P2 $[1.0000, 0.0000]$ and $[0.9933, 0.0067]$; P4 ratio $1{:}1$; P8 $196/768/590{,}592/197$; P10 ratio $\approx 1.95$) | [verified-NumPy] | exec re-run |
| All deck equations, scaling/slide tables, all torch code, generation outputs, BLEU/test numbers | [recorded] | PDF text extraction + notebook JSON |

## Thin / contradictory points

1. **Post-norm vs pre-norm.** The deck's §46.12 layer diagrams show post-norm ("Add & Norm" after each sublayer); the nanoGPT notebook implements **pre-norm** (LayerNorm before each sublayer). Both appear in the chapter with an explicit Note saying which is which — not treated as a contradiction, since pre-norm is the modern default the notebook chose.
2. **ViT positional embeddings are learned**, not sinusoidal — deck states both explicitly ("a set of learnable positional embeddings $P_{pos}$"); the chapter keeps them separate with a Note, mirroring the nanoGPT learned-embedding note.
3. **Emergent-abilities / scaling-law claims** are the deck's empirical statements ("1 T (est.)" for GPT-4, LLaMA 3 at 126 layers) — transcribed as [recorded], not independently verifiable from here; stated as the deck's table.
4. **The 2014 Bahdanau citation was dropped** — the name is in the notebook ("Bahdanau attention mechanism") but no year appears in any source; the chapter names the mechanism without dating it.
5. **BERT's NSP** is presented as the deck taught it (pre-training objective alongside MLM). No claim made about later architectural revisions (out of source).
6. **"Karpathy-style"** was removed from §46.15 — the notebook is its own implementation; the characterization wasn't in the source.

## Fixes made during review

- §46.5: matrix-form box used $\sqrt{d}$ while the Notation paragraph promised $\sqrt{d_k}$ — box and surrounding text corrected to $\sqrt{d_k}$.
- §46.1(ii): the "$O(1)$ path length vs $O(T)$ for an RNN" claim was softened to the deck's own vocabulary ("direct interactions between any two tokens," "$O(T)$ sequential depth"); §46.16(iii) matched.
- §46.11: "Same trick as ResNet (§44.8)" was wrong — §44.8 is transfer learning; corrected to U-Net's skip connections (§44.9), which is the book's actual documented analog.
- §46.7: dropped the unsourced "(2014)" from Bahdanau.
- §46.12: "then $-\infty$'d" (coinage) → "set to $-\infty$".
- §46.6: "one-third of the other" → "about a third of the other" ($0.3595$ vs $1/3$).
- §46.15: "Karpathy-style" → plain "decoder-only model" (see thin point 6).
