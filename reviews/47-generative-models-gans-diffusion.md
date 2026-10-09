# Review log — Chapter 47: Generative models: GANs and diffusion

Branch: `draft/generative-models-gans-diffusion`. First draft written 2026-10-09. Not merged — awaiting independent review. Final chapter of the 47-chapter book.

## Sources actually used

1. **GenAI Week 6 deck "Introduction to Generative Adversarial Networks (GANs)"** (Balaji Srinivasan, Ganapathy Krishnamurthi), 65 slides — `sources/course materials/GenAI-20261009T001550Z-1-001/GenAI/Slides/Week - 6/Generative Models and GAN - Lecture Slides.pdf`. The 225-page PDF is the same 65 slides repeated with beamer overlays (verified via the "N / 65" page numbering). Notes PDF `.../GenAI/Notes/Week6-Generative Models and GAN .pdf` is identical content — used as the working cross-check copy (extracted to `/tmp/ch47_src/gan_notes.txt`). Cited as "the GAN deck" / "the deck."
2. **GenAI Week 11 deck "Introduction to Diffusion Models"** (same authors), 94 slides — `.../Slides/Week - 11/Week11_GenAI_Diffusion20251127.pdf` (339 pages with beamer overlays; text-extractable via `pdftotext`, extracted to `/tmp/ch47_src/diffusion.txt`, 20466 words). Cited as "the diffusion deck."
3. **GAN_from_Scratch.ipynb** — `.../GenAI/Slides/Week - 6/GAN_from_Scratch.ipynb`, 32 cells extracted via python JSON. All torch code and logged numbers transcribed as [recorded] (torch is not installed on this machine; nothing re-run). See thin point 1.
4. **Diagram references only (not fetched)**: Goodfellow et al. 2014 GAN (https://arxiv.org/abs/1406.2661), Ho et al. 2020 DDPM (https://arxiv.org/abs/2006.11239), Song et al. 2020 DDIM (https://arxiv.org/abs/2010.08092), Rombach et al. 2021 LDM (https://arxiv.org/abs/2112.10752) — cited in HTML comments per the book's reuse rule. The diffusion deck itself credits its forward/reverse figure to "the DDPM paper"; the others are the standard references for the named papers the decks cite.
5. **Book chapters as anchors**, every §-reference verified against the actual files: ch 24 (PCA), ch 30 (Naive Bayes), ch 40 (§40.5(i)), ch 42 (§42.4), ch 43 (§43.12), ch 44 (§§44.6, 44.9, 44.10(iii)), ch 45 (§§45.1, 45.10, 45.11, 45.13(iii)), ch 46 (§§46.2, 46.5, 46.7, 46.10, 46.12, 46.16(i)).

## Re-run / recorded ledger

| Item | Status | Evidence |
|---|---|---|
| §47.2 marginal $P(x_{11}=1) = 0.51$ from the deck's 8 numbers | [verified-NumPy] | exec re-run, exact |
| §47.6 BCE eg: $L_D = 0.4338$, $L_G$ minimax $= 0.6365$, $L_G$ non-sat $= 0.8574$ | [verified-NumPy] | exec re-run |
| §47.6 Note: $-\log(0.05) \approx 2.9957$ | [verified-NumPy] | exec re-run |
| §47.7 param counts: 283,920 / 267,009; 16,911 difference | [verified-NumPy] | exec re-run; matches notebook's printed totals |
| §47.7: 100×600 = 60,000 steps; ratio $1.5290/0.8781 = 1.7413$ | [verified-NumPy] | exec re-run |
| §47.12 forward toy: $\bar\alpha_2 = 0.6$, $x_2 = 1.2173$ | [verified-NumPy] | exec re-run |
| §47.15 time embedding: emb(0) $= [0,1,0,1]$; emb(1) $= [0.8415, 0.9950, 0.0001, 1.0000]$ | [verified-NumPy] | exec re-run, deck formula |
| §47.17 DDIM step: $\hat x_0 = 1.0142$, $x_{t-1} = 1.0386$ | [verified-NumPy] | exec re-run |
| §47.19: $786{,}432 / 16{,}384 = 48$ compression | [verified-NumPy] | exec re-run |
| §47.20 FID toy $= 1.0$ | [verified-NumPy] | exec re-run |
| Solutions P2: $0.2521$ / $0.3670$ / $1.2629$ | [verified-NumPy] | exec re-run |
| Solutions P4: Nash $V = -1.3863$; P5: $\bar\alpha_2 = 0.72$, $\mathcal{N}(0.8485\,x_0, 0.28\,I)$ | [verified-NumPy] | exec re-run |
| Solutions P6: $\beta_{500} \approx 0.0100$, $\sqrt{\bar\alpha_{500}} \approx 0.2803$, $\sqrt{\bar\alpha_{1000}} \approx 0.0064$ | [verified-NumPy] | exec re-run |
| Solutions P7: $\hat x_0 = 0.7940$, $x_{t-1} = 0.5376$ | [verified-NumPy] | exec re-run |
| Solutions P9: $(2.0, 1.1)$; P10: IS $= 1.2835$ (KLs $0.2059/0.0781/0.4648$) | [verified-NumPy] | exec re-run |
| All deck equations (minimax, $q(x_t\mid x_{t-1})$, reparam formula, $\tilde\mu_t$, $L_{\text{simple}}$, Alg 1–2, DDIM update, guidance formulas, FID/IS/CLIP, trilemma + architecture + versions tables), all notebook code/numbers (losses, Adam $0.0002$, epochs, outputs) | [recorded] | PDF text extraction + notebook JSON |

## Thin / contradictory points

1. **Notebook dataset naming.** The hyperparameter cell says "FashionMNIST" (and calls them "handwritten digits" — FashionMNIST is clothing, not digits), while the final comparison panel is titled "Real MNIST Images vs GAN-Generated Images." The chapter describes the data as MNIST-family 28×28 grayscale images and notes the naming mix here rather than in the chapter. Reviewer: decide whether to pin one name.
2. **Notebook betas.** Cell 21's comment names Adam betas $(0.5, 0.999)$ but the code line passes only `lr=0.0002`. The chapter reports Adam at $0.0002$ and does not claim the betas were used. Fine as-is.
3. **Diffusion training/inference numbers** (20 s vs 1 s, "48x less", LAION-5B, SD version table, CLIP 400M pairs) are the diffusion deck's claims, transcribed as [recorded] — not independently verifiable from here.
4. **VAE is not a chapter topic** — it appears only in §47.1(iii)'s trilemma and the taxonomy (§47.2), both faithful to the decks' own VAE-vs-GAN slides (incl. the ELBO loss, which the chapter deliberately omits to stay in scope). Reviewer: confirm this scope cut is acceptable.
5. **"The deck" convention** — two decks feed this chapter; both are called "the deck" with a disambiguation note in the intro paragraph and explicit "GAN deck"/"diffusion deck" labels where needed.
6. **DDIM inversion formula** is given in the deck but the chapter summarizes it in prose (formula deemed too heavy for the chapter's length budget); the numeric DDIM eg covers the update mechanics.
7. **Post-norm vs pre-norm** (cf. ch46 thin point 1): the diffusion deck's DDPM table specifies GroupNorm + SiLU; no conflict with the ch46 ResNet/transformer note since these are different architectures.

## Fixes made during review (self-review pass)

- §47.15: the numeric label $\cos(5.6 \times 10^{-5})$ for the $i = 3$ embedding component was wrong — for $d = 4$, $(2i-1)/d = 1.25$, so the argument is $1/10000^{1.25} = 1 \times 10^{-5}$ (the numpy code always used the deck's exact formula; the displayed value $1.0000$ was unaffected). Corrected to $\cos(1 \times 10^{-5})$.
- §47.6: aligned-equation line break used a single `\` — corrected to `\\`.
- §47.6 Note: "the non-saturating trick is the standard patch" was an editorial claim; rephrased to attribute the practice to the notebook ("the one the notebook actually trains with").
- §47.18 Basically: "trains the model twice" was misleading for classifier-free guidance; rephrased to "trains one model on both regimes (condition dropped 10% of the time)."
- §47.8: DCGAN described as "the first architecture that made GANs work on real images" — an editorial superlative; replaced with the deck's own wording ("a major step forward, successfully using deep convolutional nets for larger images").
- §47.1(ii): "the GAN's discriminator *is* a CNN classifier" overclaimed for the vanilla (dense) notebook GAN; qualified to "becomes a CNN classifier at DCGAN scale."
- §47.21(iv): dropped the named unsourced techniques (score-based SDEs, flow matching) from the beyond-the-course paragraph; now points only to the GOAL.md supplementary shelf.
- §47.2: "Take a \"image\"" typo → "Take an \"image\"".
- §-cross-references: verified §§44.6, 44.9, 44.10(iii), 45.1, 45.10, 45.11, 45.13(iii), 46.2, 46.5, 46.7, 46.10, 46.12, 46.16(i), 24 (eigenfaces), 30 (Naive Bayes), 40.5(i), 42.4, 43.12 against the actual chapter files; eigenfaces attribution to §24 confirmed via grep.
- Full numpy re-run of every [verified-NumPy] number in the chapter and all 11 problem solutions after the fixes: all PASS (see ledger above).
