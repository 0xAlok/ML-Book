# Review log — Chapter 45: RNNs — sequences and vanishing gradients

Branch: `draft/rnns`. Self-review pass completed 2026-10-09. Not merged — awaiting independent review.

## Sources actually used

1. **GenAI Weeks 7–8 "Introduction to RNN" deck** (Balaji Srinivasan, Ganapathy Krishnamurthi), 80 slides — `sources/course materials/GenAI-20261009T001550Z-1-001/GenAI/Slides/Week - 7 & 8/Introduction to RNN - Lecture Slides.pdf`. Text-extractable (no `pdftoppm` needed, unlike the CNN deck); the 231-page PDF is the same 80 slides repeated with beamer overlays. Cross-checked against the notes PDF below — identical content.
2. **Notes `Week7,8-Introduction to RNN.pdf`** — `.../GenAI/Notes/Week7,8-Introduction to RNN.pdf`, 80 pages, `pdftotext`-extracted to `/tmp/rnn-notes.txt` (1402 lines). Used as the working copy; verified slide-for-slide against the deck (same numbering "N / 80", same worked examples, same figure credits). Cited as "the deck" throughout the chapter since content is identical.
3. **Time_Series_Analysis_using_RNN.ipynb** — `.../Week - 7 & 8/`, 42 cells extracted via python json: AMZN/yfinance pipeline, `prepare_dataframe_for_lstm`, MinMaxScaler$(-1,1)$, `(samples, timesteps, features)` reshape, `RNN` class (`nn.RNN(input_size, hidden_size, num_stacked_layers, batch_first=True)` + `nn.Linear(hidden_size, 1)`, zero `h0`, head on last timestep), `nn.MSELoss`, `torch.optim.Adam(lr=0.001)`, 50 epochs. All torch code transcribed as [recorded] (torch not installed; nothing re-run).
4. **sentiment-analysis.ipynb** — `.../Week - 7 & 8/`, 53 cells extracted via python json: IMDB pipeline, $h_t = \sigma(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$ framework (§1.4), `SentimentRNN` / `SentimentLSTM` (embedding → recurrent → dropout $0.5$ → fc → sigmoid; LSTM `hidden` as `(h, c)` tuple). All torch code [recorded]. Dataset download cells not used (no download performed).
5. **d2l.ai RNN chapter index** — live-checked (https://d2l.ai/chapter_recurrent-neural-networks/index.html); Fig. 9.1 (folded vs unrolled RNN, shared parameters) cited as an HTML-comment diagram reference (CC-BY-SA), not as content source. The deck's beam-search illustration is already credited to d2l.ai in the source.
6. Deck figure credits (kept as source attributions, not fetched): BPTT illustration (DOI 10.13140/RG.2.2.34411.08483), LSTM architecture (DOI 10.1109/ACCESS.2021.3125733), GRU architecture (DOI 10.1007/s11042-023-15571-y).
7. Book chapters as anchors, every §-reference verified against the actual files: §14 (chain rule), §35 (log-loss), §36 (§§36.4, 36.6), §40 (§40.5(i)), §41 (§§41.4, 41.7, 41.14(ii)), §42 (§§42.3–42.4, 42.6, 42.8), §43 (§§43.9, 43.12), §44 (§44.7, §44.10).

## Re-run / recorded ledger

| Item | Status | Evidence |
|---|---|---|
| §45.4 forward eg: $h_t = (0.4301, 0.7259)$, logits $(0.2343, 0.6301, 2.9078, -1.0518)$, argmax 'l' | [verified-NumPy] | exec re-run; deck's 4th-logit slip ($-1.01 \to -1.4518$, final $-0.61 \to -1.0518$) corrected with Note |
| §45.6 BPTT eg: $L = 0.2216$, $\partial L/\partial w_{hh} = -0.0357$, $\partial L/\partial w_{hy} = -1.0936$, $\partial L/\partial w_{xh} = -0.2796$ | [verified-NumPy] | exec re-run; deck's $y_2$ slip ($0.744 \to 0.3717$) corrected with Note |
| §45.7 eg: $0.64^{20} \approx 1.33\times10^{-4}$, $1.5^{20} \approx 3325$; §45.8 eg: $0.95^{20} \approx 0.36$; $\tanh'(1) \approx 0.42$ | [verified-NumPy] | exec re-run |
| BLEU eg (deck's): BP $= e^{-2} \approx 0.1353$, BLEU $= 0.1353$; perplexity sanity PPL$(0.5\times4) = 2.0$; Sol 10 BP $= e^{-1} \approx 0.3679$ | [verified-NumPy] | exec re-run |
| Solution numbers: P1 $10^8/10^{12}$; P2 $h_1 = 0.7163, h_2 = -0.7235$ ($-0.6044$ w/o memory); P3 $15{,}100$ vs $153{,}000$; P8 $c_t = (1.2, 0.08)$; P9 $H_t = (0.8, 1.7)$ | [verified-NumPy] | exec re-run |
| All deck equations (RNN/LSTM/GRU/BPTT/BLEU/beam/perplexity), all torch code, d2l figure URL | [recorded] | PDF text extraction + notebook JSON |

## Thin / contradictory points

1. **Two deck arithmetic slips, both corrected with Notes** (ch42's deck-typo convention): (a) §45.4 — 4th pre-bias logit written as $-1.01$, exact $-1.4518$; (b) §45.6 — $y_2 = 0.4\cdot0.93$ written as $0.744$ (equals $0.8\cdot0.93$; the target crept in as the weight). In (b) every downstream deck number was consistent with the slipped value, so only one line needed repair; the full backward pass was re-derived from the corrected $y_2$.
2. **GRU slides switch notation** (row-vector $X_t W_{xr}$) vs LSTM slides (column $W_f\cdot[h_{t-1},x_t]$). Flagged in a §45.9 Note; comparison kept architectural, not notational.
3. **Gradient clipping has no formula in the deck** — one sentence ("If the norm of the gradient exceeds a certain threshold, it is scaled down"). The torch call is supplied by §43.12; the chapter cites it and does not invent a formula.
4. **No worked forward/BPTT example exists for LSTM/GRU in the sources.** The chapter works the vanilla RNN end-to-end and illustrates the LSTM gradient path numerically (§45.8 eg, own computation marked [verified-NumPy]) rather than inventing gate internals.
5. **The sentiment notebook's §1.4 writes $h_t = \sigma(\dots)$ with generic $\sigma$** while the deck's neuron fixes $\tanh$. Flagged in a §45.3 Note as an activation swap, same cell.
6. **LSTM/GRU deck figures are DOI-credited** (researchgate/IEEE/Springer) and were not fetched; the chapter reuses no figure from them — only the d2l.ai unrolled-RNN figure (verified live URL) as an HTML-comment reference, and the deck's d2l.ai credit for beam search is carried in the Sources footnote.
7. **GRU appears in the deck's theory only** — neither notebook builds one (time-series uses `nn.RNN`, sentiment builds RNN and LSTM). §45.9 states this explicitly.

## Fixes made during review

- §45.1: "(§14's rule, reused)" removed — no book chapter labels the probability chain rule; the deck is the source, now cited as such.
- §45.7: §41.7 quote corrected to the tanh row's exact wording ("same vanishing-gradient flattening — 'difficult to use in very deep networks'"); previously mixed the sigmoid row's "flat at both ends" with the tanh row.
- §45.11: "log-loss of §35's vocabulary" → "negative log-likelihood of §35.6's cross-entropy vocabulary" (ch35's actual terms).
- Sources footnote: dropped ch14 and §36.6 (neither used in the chapter body); now lists exactly the referenced sections.
- §45.6: $\partial L/\partial h_1$ rounding slip $-0.1656 \to -0.1657$ (exact $-0.165654$; $-0.1563 + -0.0094$ also sums to $-0.1657$); same fix in the following $\left.\partial L/\partial w_{hh}\right|_{t=1}$ line.
- Full-number audit: every figure in chapter + solutions re-run against numpy at the printed precision; all other roundings check out (see ledger).

## Files

- `chapters/45-rnns-sequences-vanishing-gradients.md` — the chapter (13 sections + problem set), ~10–14 pages
- `solutions/45-rnns-sequences-vanishing-gradients.md` — full worked solutions, separate volume
- `reviews/45-rnns-sequences-vanishing-gradients.md` — this log

## Concerns for independent review

1. **§45.6's corrected backward pass**: confirm the re-derived numbers (especially $\partial L/\partial h_1 = -0.1657$ with future component $-0.0094$) read correctly against the deck's original intent, and that flagging the $y_2$ slip in a Note (rather than silently using corrected numbers) matches the book's convention.
2. **§45.8's gradient-path eg** ($0.95^{20}$ vs $0.64^{20}$) is a book-computed illustration of the deck's "additive interaction" claim, not a source example — confirm the framing ("the gates learn where to sit") doesn't overclaim beyond the deck.
3. **§45.10's teacher-forcing mismatch** is stated but named as left open; confirm not naming exposure bias (absent from sources) is the right call.
4. **§45.11's BLEU $\min(0, 1 - \cdot)$ rendering**: confirm the brevity-penalty formula transcription matches the deck's typesetting.
5. **Problem 10(iii)** asks whether beam search fixes the teacher-forcing mismatch — the accepted answer is "no." Confirm this isn't too subtle for the chapter's stated scope.
