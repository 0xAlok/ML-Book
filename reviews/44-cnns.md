# Review log — Chapter 44: CNNs — convolution, pooling, transfer learning

Branch: `draft/cnns`. Self-review pass completed 2026-10-09. Not merged — awaiting independent review.

## Sources actually used

1. **GenAI Weeks 3–4 "Introduction to CNN" deck** (Balaji Srinivasan, Ganapathy Krishnamurthi), 91 slides — `sources/course materials/GenAI-20261009T001550Z-1-001/GenAI/Slides/Week - 3 & 4/Introduction to CNN - Lecture Slides.pdf`. Image-only PDF, so all 234 beamer-overlay pages rendered with `pdftoppm -png -r 100` and read off PNGs. Used: naive-MLP-flatten slide (deck slide 12), Principle 1 (translation invariance), "A Note on Terminology: Convolution" slide (deck slide 23 — DL uses cross-correlation, "Does it matter?"), edge-detection slide (slide 27), "Learning a Convolutional Kernel" slide (slide 30 — $W \leftarrow W - \eta\,\partial\mathcal{L}/\partial W$), kernel-size note (odd sizes, "provides a center pixel"), stride illustration (slide 40 — $5 \times 5$, $3 \times 3$, stride 2 → $2 \times 2$), padding illustration (slide 43 — "valid" $4 \times 4$, $3 \times 3$ → $2 \times 2$, "Output shrinks"), pooling definition + purposes (slide 47), **batch-normalization walkthrough (slide 54)** — $\hat x_i = (x_i-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$, $y_i = \gamma\hat x_i+\beta$, data augmentation, ImageNet challenge, ResNet degradation, transfer-learning slides (slides 88, 90 — "don't reinvent the wheel", ResNet50/MobileNet backbone, ImageNet "1.4 million images, 1000 classes").
2. **CNN_Fundamentals.ipynb** — `.../Week - 3 & 4/CNN_Fundamentals.ipynb`, 35 cells extracted via python json: MLP limitations (§44.1's four counts; 150,528 px, 15,052,800 weights for 100 neurons), convolution definition $(I*K)(i,j)=\sum_m\sum_n I(i+m,j+n)K(m,n)$ + hyperparameters, weight sharing (64 filters: CNN 640 vs MLP 50,176; biological inspiration), equivariance vs invariance, pooling types + worked $4 \times 4$→$2 \times 2$ max-pool ASCII example, complete MNIST CNN architecture ($26{\times}26{\times}32 \to 13{\times}13{\times}32 \to 11{\times}11{\times}64 \to 5{\times}5{\times}64 \to 3{\times}3{\times}128 \to 1152$ → FC), feature-map/filter visualizations. All torch code transcribed as [recorded] (torch not installed; training outputs not re-run).
3. **Transfer_Learning.ipynb** — `.../Week - 3 & 4/Transfer_Learning.ipynb`, 52 cells extracted via python json: transfer objective $L_{\text{transfer}} = L_{\text{target}} + \lambda R(\varphi)$, three-strategy table (feature extraction / fine-tuning / progressive), freeze mechanics (`requires_grad=False`, "untrainable"), EfficientNet-B0 anatomy (features/avgpool/classifier, `out_features=1000`), ImageNet preprocessing (mean $[0.485,0.456,0.406]$, std $[0.229,0.224,0.225]$, 224×224 resize), FoodVision Mini recipe (pizza/steak/sushi), Adam $\eta=0.001$ for the head. All torch code [recorded].
4. **Image_Segmentation_UNET.ipynb** — 69 cells, header + framing cells read only: U-Net = encoder (conv+pool) + decoder (upsampling) + skip connections, leaf-disease segmentation case study, Dice coefficient. Used for §44.9's one-paragraph note only; architecture detail not worked.
5. **Notes `Week3-CNN.pdf` (45 pp), `Week4-CNN,Transfer Learning.pdf` (46 pp)** — rendered and contact-sheeted; verified to be slide-copies of the deck (same content, beamer formatting), no unique material; not cited beyond this note.
6. Book chapters as anchors, every §-reference verified against the actual files: §24.5 (PCA), §36.4 (axis semantics), §40.5(i) (debugging playbook), §41.4 (two equations), §41.7 (activations), §41.14(ii) (39,760 params, dropout), §42.4 (layer-wise rules), §42.8(ii) (vanishing-gradient reading), Chapter 43 (layer-agnostic optimizers; §43.9's BN gap; §43.10 init; §43.12 AdamW defaults).
7. **d2l.ai §7.2.1 (Convolutions for Images)** — live-checked (https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html); confirms the "strictly speaking cross-correlation" point and the 3×3-input/2×2-kernel worked example. Cited only as an HTML-comment diagram reference (CC-BY-SA), not as content source.

## Re-run / recorded ledger

| Item | Status | Evidence |
|---|---|---|
| eg 3: $4{\times}4$ conv with $K=\begin{bmatrix}1&0\\-1&1\end{bmatrix}$ → `[[2,3,4],[6,7,8],[10,11,12]]` | [verified-NumPy] | exec re-run (hand-computed entries match numpy exactly) |
| Output-size formula $\lfloor(W-K+2P)/S\rfloor+1$ on 8 configs incl. deck's $(4,3,0,1)\to2$, $(5,3,0,2)\to2$, notebook's $(28,3,0,1)\to26$, $(26,2,0,2)\to13$, $(13,3,0,1)\to11$ | [verified-NumPy] | exec re-run |
| eg 5: notebook's $4{\times}4$ max-pool → `[[9,8],[4,7]]` | [verified-NumPy] | exec re-run (matches notebook's ASCII diagram) |
| Param counts: dense $224{\times}224{\times}3{\to}100 = 15{,}052{,}900$; conv $64{\times}(3{\cdot}3{\cdot}3{+}1) = 1{,}792$; ratio $8{,}400\times$; $784{\times}64 = 50{,}176$; $64{\times}10 = 640$ | [verified-NumPy] | exec re-run |
| Solution counts: Prob 1(iii) $15{,}052{,}900/39{,}760 = 378.6$; Sol 5(iii) $128{\times}289 = 36{,}992$; Sol 10(ii) $448+4{,}640+9{,}248 = 14{,}336$; dense $64{\times}64{\times}3{\to}1152 = 14{,}156{,}928$ (caught 14,159,616 typo in review) | [verified-NumPy] | exec re-run |
| All deck/notebook numbers (15,052,800; 640 vs 50,176; BN formulas; ImageNet stats; transfer tables; torch recipes), d2l cross-check | [recorded] | PNG renders + notebook JSON extraction |

## Thin / contradictory points

1. **The notebook's $3{\times}3 = 9$ weights claim is per input channel.** It says "for a single $3 \times 3$ convolutional filter: only 9 weights (plus 1 bias)" — true for grayscale; on 3-channel input it's 27. §44.4 states this nuance explicitly rather than transcribing the claim raw.
2. **§43.9's flagged BN gap is now partly closed.** The ch43 log said batch norm was "used but never explained" in the *Week 5* materials; the Weeks 3–4 deck *does* give the two-line walkthrough (normalize + scale/shift, slide 54). §44.7 cites it and keeps the full treatment (why $\gamma,\beta$; train/test behaviour) deferred. §43.9's statement about Week 5 stands — no retro-edit needed, but the ch43 review log's "later pass with Chapter 44" proposal is now resolved as §44.7's note.
3. **U-Net coverage is intentionally thin.** The notebook is a full 69-cell worked tutorial, but the chapter title and Part VI sequencing only ask for a pointer. §44.9 is one paragraph: encoder/decoder/skip connections, per-pixel classification, Dice — architecture detail flagged for a later pass.
4. **The deck's famous-architectures section is thin.** Slides name ImageNet, ResNet (degradation problem), MobileNet; no parameter tables or derivations in the deck were cited. §44.8 names them as backbones only and claims nothing about their internals.
5. **Fashion-MNIST training numbers exist in the notebook but were not re-run** (torch absent). §44.6 cites the architecture and visualization claims, not the loss/accuracy curves.
6. **Notes PDFs = deck.** Verified by contact-sheet comparison, not assumed (45+46 pp rendered at r80).

## Fixes made during review

- §44.1(i): "a model a tenth of that size" was wrong — §41.14(ii)'s 39,760 is ~1/380 of 15,052,900, not 1/10; reworded to "almost four hundred times smaller."
- §44.6: removed a stray "§27-style interpretability" cross-reference (Ch 27 is kernel methods — wrong anchor).
- Problem 2(iii): "Add a bias $b = 2$" read as if the bias value were given; reworded to "Add one learned bias $b$" so the answer (5 learned params) is unambiguous.
- Problem 9(ii): "the two PyTorch calls" mislabeled attribute assignment as a call; reworded to "the model-construction call and the freeze mechanism."
- Solution 10(ii): dense-comparison weight count was 14,159,616 — recomputed as 14,156,928 ($64\cdot64\cdot3\cdot1152 + 1152$) and fixed.
- §44.7: $8{,}400\times$ ratio re-verified (15,052,900/1,792 = 8400.06).

## Files

- `chapters/44-cnns.md` — the chapter (10 sections + problem set), ~8–12 pages
- `solutions/44-cnns.md` — full worked solutions, separate volume
- `reviews/44-cnns.md` — this log

## Concerns for independent review

1. **Cross-correlation honesty.** §44.2 states DL "convolution" = cross-correlation and that it doesn't matter because kernels are learned. The deck says the same. Confirm the framing ("flipped copy of a learned kernel is just another learnable kernel") reads correctly for a student, not hand-wavy.
2. **Solution 8(ii)'s claim** that deltas still propagate *through* frozen layers to earlier trainable ones while $\partial\mathcal{L}/\partial W$ isn't computed — technically true for `requires_grad=False` parameters, but please confirm the wording ("freezing stops $\partial\mathcal{L}/\partial W$ from being computed, not $\partial\mathcal{L}/\partial x$ from propagating") is exact for the efficientnet_b0 features block where *everything* is frozen.
3. **§44.7's BN note**: confirm citing the slide-54 formulas as "partly closing §43.9's gap" is the right call vs. leaving BN entirely alone — it introduces two new symbols ($\gamma, \beta$) in a chapter not otherwise about normalization.
4. **§44.9 U-Net brevity**: confirm the one-paragraph pointer is sufficient for the chapter title, or whether the 69-cell notebook deserves a worked eg.
5. **d2l HTML comment**: the diagram is referenced (CC-BY-SA) with a verified live URL but not hotlinked — confirm this matches how the book wants diagram attribution (ch43 drew its own figure instead).
6. **Problem 9(iii)** asks for "the risk" with Adam at $\eta = 0.001$ on the unfrozen backbone — the accepted answer is "destroying the pretrained features." Confirm this isn't overcautious given Chapter 43 lists fine-tuning with AdamW as standard practice.
