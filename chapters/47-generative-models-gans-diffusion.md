# Chapter 47: Generative models: GANs and diffusion

Chapter 46 closed with the observation that the decoder of §46.12 *is* a generator: it scores sequences (the §45.1 modelling target) and samples new ones; §45.13(iii) said the same of the §45.10 decoder, and §44.10(iii) promised that "GANs and diffusion models are built on conv stacks like this chapter's — the discriminator is a CNN classifier, and the generator is the §44.9 decoder run in reverse." This chapter cashes all three. The spine is the GenAI Week 6 deck ("Introduction to Generative Adversarial Networks," Balaji Srinivasan and Ganapathy Krishnamurthi, 65 slides — generative models, latent space, the GAN game, DCGAN, StyleGAN, training challenges) plus its companion notebook *GAN_from_Scratch.ipynb* (a vanilla GAN on 28×28 FashionMNIST images), and the GenAI Week 11 deck ("Introduction to Diffusion Models," same authors, 94 slides — taxonomy, FID/IS/CLIP, DDPM forward/reverse, reparameterization, schedulers, DDIM, classifier-free guidance, latent diffusion/Stable Diffusion). Every numpy number below was re-run here and is marked **[verified-NumPy]**; deck, notebook, and torch numbers are transcriptions, marked **[recorded]**. "The deck" below means the GAN deck when GANs are discussed and the diffusion deck when diffusion is — they agree on everything they share.

**Notation.** $p_{\text{data}}$ is the real data distribution; $p_{\text{model}}$ (sometimes $p_\theta$) the model's. $z \sim p(z)$ is a latent/noise vector, usually $\mathcal{N}(0, I)$. The generator is $G(z; \theta_g)$, the discriminator $D(x) \in [0, 1]$. In diffusion, $x_t$ is the image at noise level $t$, $\beta_t$ the noise schedule, $\alpha_t = 1 - \beta_t$, $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$, and $\epsilon_\theta(x_t, t)$ the network's predicted noise.

## 47.1 Two promises this chapter cashes

**i) The §46.16(i) promise.** The transformer decoder *generates* by sampling next tokens autoregressively. This chapter adds two more generator families: the GAN (learn the mapping noise → image in *one* shot, trained adversarially) and the diffusion model (learn to undo noise in *many* small steps). Three ways to turn noise into data — decoder, adversary, denoiser — closing the arc that began at §45.13(iii).

**ii) The §44.10(iii) promise.** Both new families run on Chapter 44's conv stacks. The GAN's discriminator becomes a CNN classifier at DCGAN scale (§44.6's stack with a sigmoid head); its generator is the §44.9 U-Net decoder run in reverse (noise up to full image). The diffusion model's noise predictor is a U-Net with skip connections (§44.9) — the same Lego, a new game.

**iii) The trilemma that motivated diffusion.** The diffusion deck's table (condensed) [recorded]: VAEs give **stable training + good diversity** but **blurry samples** (the MSE loss "averages out fine details"); GANs give **sharp samples** but **poor diversity** (mode collapse) and **unstable training**. Diffusion asks: can we keep the stability and coverage of VAEs *with* GAN-level sharpness? Its answer is to replace one hard generation step with many easy denoising steps.

**Basically, ...** "Three ways to make new data from noise: the transformer writes it word by word (§46.12), the GAN forges it in one shot while a critic judges (§47.4), and diffusion sculpts it by repeatedly removing a little noise (§47.13). All three reuse the machinery of Chapters 41–46 — conv stacks, skip connections, attention, Adam."

## 47.2 What a generative model is

**Def (the deck's).** "A generative model is a type of statistical model that is capable of learning the underlying distribution of a dataset in order to generate new, synthetic data samples that are similar to the original data." Formally: the real data comes from $p_{\text{data}}(x)$; the model learns a distribution $p_{\text{model}}(x)$ and training tries to make $p_{\text{model}}$ as close to $p_{\text{data}}$ as possible.

**The generative vs. discriminative split.** A discriminative model learns $P(y \mid x)$ — "given this image, cat or dog?" (logistic regression, SVMs, ordinary classifiers). A generative model learns $P(x)$ — "what makes an image look like a cat?" — and can then *sample* from it (VAEs, GANs, diffusion).

**The deck's taxonomy (condensed)** [recorded]:
- **i) Explicit density, tractable.** Define and compute $p(x)$ exactly. PixelCNN/RNN, Glow, NADE/MADE — autoregressive pixel-by-pixel models live here.
- **ii) Explicit density, approximate.** $p(x)$ is intractable; optimize a proxy. VAEs maximize a lower bound on likelihood (ELBO); diffusion models are classified here too ("systematically add noise to data, then learn to reverse the process").
- **iii) Implicit density.** Never define $p(x)$ at all — just learn to *sample* from it. GANs are "the dominant family in this category."

**Why they're useful (deck's list)** [recorded]: data generation, understanding latent structure, missing-data imputation (inpainting), anomaly detection, representation learning. The deck also notes the generative-classifier trick: with one generative model $P(x \mid y = c)$ per class, Bayes' rule picks the class maximizing $P(x \mid y = c)P(y = c)$ — Naive Bayes is the textbook instance (§30's territory).

**The deck's toy (2×2 grid, worked).** Take an "image" as a 2×2 grid, each pixel black (0) or white (1): $2^{2 \times 2} = 2^4 = 16$ possible images. A generative model learns $P(x_{11}, x_{12}, x_{21}, x_{22})$ — for this toy, just count occurrences and normalize [recorded].

**eg (the deck's marginalization numbers — [verified-NumPy]).** The deck's hypothetical joint gives eight configs with $x_{11} = 1$: $0.05, 0.10, 0.15, 0.03, 0.02, 0.08, 0.07, 0.01$. Marginalizing means summing over the other pixels:
$$P(x_{11} = 1) = 0.05 + 0.10 + 0.15 + 0.03 + 0.02 + 0.08 + 0.07 + 0.01 = 0.51.$$
Same idea, one step harder: **inpainting.** If $x_{11} = 1$ is known and the rest missing, the best fill is $\arg\max_{x_{\text{mis}}} P(x_{\text{mis}}, x_{\text{obs}})$; the deck's table max is $P(1, 0, 1, 0) = 0.15$, so the completion is $(x_{12}, x_{21}, x_{22}) = (0, 1, 0)$ [recorded].

**Note.** The toy works because $16$ is tiny. A $32 \times 32$ grayscale image has $2^{1024}$ possible binary configurations — explicit enumeration is "computationally impossible," the deck's motivating fact for everything that follows. Modern models never store the joint table; they learn a *function* that approximates it or samples from it.

**Basically, ...** "A generative model learns 'what kind of data this is' — the probability pattern — then rolls new dice weighted by that pattern. On a 2×2 toy you can write the whole probability table by hand; on real images the table is astronomically huge, so you need smarter tricks: GANs (§47.4) or diffusion (§47.10)."

## 47.3 The latent space: from enumeration to noise

For real data, the deck's pivot is: don't try to sample from $p(x)$ in pixel space — learn a **mapping** from a simple noise distribution to the data:

**Def.** Sample $z \sim p(z)$ (usually $\mathcal{N}(0, I)$ — easy). A generator $G(z; \theta_g)$ maps the latent space $\mathcal{Z}$ to data space $\mathcal{X}$: $x = G(z)$. Training finds $\theta_g^*$ so the distribution of $G(z)$ matches $p_{\text{data}}$.

**Why it works (the deck's intuition).** **i)** Most pixel configurations are meaningless noise; real images sit on a thin "manifold" inside pixel space. **ii)** You cannot "pick random pixels and get a cat," but you can pick a random point in a small "idea space" and let a trained function grow the idea into an image. **iii)** Each $z$ is a different seed/recipe — the *noise is the source of variety*.

**The paradigm shift (deck's table, condensed)** [recorded]: pre-deep-learning latent spaces (PCA's §24 eigenfaces, factor analysis, LDA topics, SVD matrix factorization) went **data → latent** and asked "what are the hidden factors in my data?" Modern models flip the arrow — **noise → data** — and ask "how can I create new data that looks like my data?" The latent space got an *imposed* structure (Gaussian) instead of a discovered one, so any random point in it decodes to something sensible.

**Basically, ...** "Old latent space: take a real face, squeeze it into a short code (analysis). New latent space: pick a random short code from a nice distribution, grow a face from it (synthesis). Same word, opposite arrow — and the arrow reversal is what makes generation possible."

## 47.4 The GAN: a two-player game

Proposed by Goodfellow et al. (2014) [recorded]. The deck's analogy: an **aspiring artist** (the generator) forges paintings in a master's style without ever seeing the originals, while an **art critic** (the discriminator) studies real masterpieces and forgeries to learn to tell them apart. The game ends when the forgeries fool the critic half the time.

<!-- Diagram: the GAN architecture — generator fed by noise, discriminator judging real vs. generated, feedback loop; the deck's figure (source credited to Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow). Standard reference: Goodfellow et al., "Generative Adversarial Networks," https://arxiv.org/abs/1406.2661 -->

**i) The discriminator $D$: the critic.** Wants to classify correctly — real data as real, fakes as fake. Its objective to maximize:
$$\max_D \; \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] \quad \text{[recorded]}.$$
First term: push $D(x) \to 1$ on real data. Second term: push $D(G(z)) \to 0$ on fakes.

**ii) The generator $G$: the forger.** Wants to fool $D$ — minimize the second term above:
$$\min_G \; \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] \quad \text{[recorded]}.$$

**The combined minimax objective** [recorded]:
$$\boxed{\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]}$$
"The Generator tries to minimize the value function that the Discriminator tries to maximize" — a zero-sum game called **adversarial learning**. The ideal end state is a **Nash equilibrium**: $G$ produces perfectly realistic data and $D$ "outputs a 50% probability for all inputs" — the critic is fooled exactly half the time [recorded].

**Note (the deck vs. the notebook).** The deck teaches the minimax form above. In practice — including this chapter's notebook (§47.7) — the generator is instead trained to **maximize $\log D(G(z))$** (label fakes as real), which the deck flags as equivalent ("pushing the Discriminator's output for fake images towards 1") but with livelier gradients early on. §47.6 works both.

**Basically, ...** "Two networks, one lie detector. The forger makes fakes, the critic calls them out; both get better. The score to beat: when the critic is right exactly half the time — no better than a coin flip — the forgeries are perfect."

## 47.5 Training: alternate, freeze, repeat

The deck's training algorithm has two phases, run in turns [recorded]:

**Phase 1 — train $D$ (freeze $G$).** Sample real $x \sim p_{\text{data}}$ and noise $z \sim p_z$, make fakes $G(z)$. Gradient **ascent** on the discriminator's objective (it's a max):
$$\theta_d \leftarrow \theta_d + \alpha \nabla_{\theta_d} \frac{1}{m}\sum_{i=1}^{m} \big[\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))\big].$$

**Phase 2 — train $G$ (freeze $D$).** Sample fresh noise $z \sim p_z$. Gradient **descent** on the generator's loss — gradients flow back *through the frozen discriminator* to reach $G$:
$$\theta_g \leftarrow \theta_g - \alpha \nabla_{\theta_g} \frac{1}{m}\sum_{i=1}^{m} \log(1 - D(G(z^{(i)}))).$$

Repeat both phases until the Nash point (or exhaustion). The freezing matters: each network learns against a fixed opponent for one step, so the "moving target" only moves between steps.

**Basically, ...** "Coach the critic while the forger holds still; then coach the forger while the critic holds still. Two optimizers, two losses, one loop — and each half updates while the other is frozen, so the target stands still for a step."

## 47.6 The losses, worked by hand [verified-NumPy]

**eg (own).** Mini-batch of 2 real, 2 fake. Suppose $D$ outputs $D(x) = [0.9,\ 0.7]$ on real images and $D(G(z)) = [0.3,\ 0.6]$ on fakes.

i) **Discriminator loss (BCE).** Each sample's binary cross-entropy is $-\log D(x)$ (real) or $-\log(1 - D(G(z)))$ (fake):
$$\begin{aligned}
L_D &= -\tfrac{1}{4}\big[\log 0.9 + \log(1-0.3) + \log 0.7 + \log(1-0.6)\big] \\
&= -\tfrac{1}{4}\big[-0.1054 - 0.3567 - 0.3567 - 0.9163\big] = 0.4338.
\end{aligned}$$

ii) **Generator loss, minimax version.** $L_G = -\tfrac{1}{2}[\log(1-0.3) + \log(1-0.6)] = -\tfrac{1}{2}[-0.3567 - 0.9163] = 0.6365$.

iii) **Generator loss, non-saturating version** (the one the notebook actually uses). Label fakes as real: $L_G = -\tfrac{1}{2}[\log 0.3 + \log 0.6] = 0.8574$.

**Note.** When $D$ is strong — say it gives a fake only $0.05$ — the non-saturating loss is $-\log(0.05) \approx 2.9957$, still steep, while the minimax $\log(1-0.05) \approx -0.0513$ barely moves. That is the deck's **vanishing-gradient** failure (§47.9): a critic that wins too early starves the forger of gradients; the non-saturating loss — the one the notebook actually trains with — is the practical patch.

**Basically, ...** "The critic's loss is plain binary cross-entropy — real images should score 1, fakes 0. The forger's smart loss is the same BCE with flipped labels: it pays the critic to score its fakes as 1. And when the critic gets too good, the forger needs the flipped-label version, because the textbook one goes flat."

## 47.7 The notebook: a GAN on FashionMNIST, from scratch [recorded — torch]

The *GAN_from_Scratch.ipynb* builds exactly the game above — plain dense networks, no convolutions yet — on 60,000 grayscale $28 \times 28$ images. **Note (notebook naming slip).** The code loads `datasets.FashionMNIST`, but the comments call the images "handwritten digits" (wrong — FashionMNIST is clothing items, not digits) and the final comparison panel titles them "MNIST". The code is authoritative: the dataset is FashionMNIST.

**The two networks** [recorded]:
- **Generator.** $\text{Linear}(64 \to 256) \to \text{LeakyReLU}(0.2) \to \text{Linear}(256 \to 256) \to \text{LeakyReLU}(0.2) \to \text{Linear}(256 \to 784) \to \tanh$. Latent vector $z \in \mathbb{R}^{64}$, $\tanh$ output matches the $[-1, 1]$ normalized images.
- **Discriminator.** $\text{Linear}(784 \to 256) \to \text{LeakyReLU}(0.2) \to \text{Linear}(256 \to 256) \to \text{LeakyReLU}(0.2) \to \text{Linear}(256 \to 1) \to \text{sigmoid}$ — a binary classifier (§44.6's stack without the convolutions).

**Parameter counts [verified-NumPy].** Generator: $(64\cdot256+256) + (256\cdot256+256) + (256\cdot784+784) = 16{,}640 + 65{,}792 + 201{,}488 = 283{,}920$. Discriminator: $(784\cdot256+256) + (256\cdot256+256) + (256\cdot1+1) = 200{,}960 + 65{,}792 + 257 = 267{,}009$.

**The training recipe** [recorded]: pixels normalized to $[-1, 1]$ to match the $\tanh$ head (denormalize with $(x+1)/2$ for display); **BCE loss**; **Adam, $\eta = 0.0002$** with separate optimizers for $D$ and $G$; batch $100$; $100$ epochs $\times$ $600$ steps $=$ $60{,}000$ updates. The $D$ step calls `fake_images.detach()` — no gradient flows into $G$ while the critic trains (§47.5's freeze, implemented as a detach). The $G$ step is the non-saturating form: BCE against `real_labels` (fakes labelled $1$).

**Results** [recorded]: average $L_D = 0.7651$, average $L_G = 2.0232$; final epoch $L_D = 0.8781$, $L_G = 1.5290$ (ratio $G/D = 1.7413$). Early on $D(x) \approx 0.95$, $D(G(z)) \approx 0.05$ — the critic dominates, exactly the regime §47.6's note describes — and the generator closes the gap over the 100 epochs. The final panel shows generated digits next to real ones.

**Basically, ...** "Sixty-four random numbers in, a $28 \times 28$ digit out — two small MLPs, a coin-flip game, Adam at $0.0002$. The critic learns fast ($0.95$ on real images by epoch 1), the forger catches up slowly, and after 60,000 alternating steps the fake digits look like digits."

## 47.8 Scaling up: DCGAN and StyleGAN [recorded]

**DCGAN (Deep Convolutional GAN).** "A major step forward, successfully using deep convolutional nets for larger images" — and §44.10(iii) in the flesh: a CNN discriminator plus a conv-transpose generator. The deck's stability guidelines [recorded]:
- **i)** Replace pooling: **strided convolutions** in $D$ (downsampling) and **transposed convolutions** in $G$ (upsampling).
- **ii)** **Batch normalization** in both nets — but *not* on $G$'s output layer or $D$'s input layer.
- **iii)** Remove fully connected hidden layers (for deeper architectures).
- **iv)** Activations: **ReLU** in $G$ everywhere except the output ($\tanh$); **LeakyReLU** in $D$ everywhere.

**StyleGAN (Nvidia).** The deck's state-of-the-art step [recorded]. Two nets inside the generator: **1)** a **mapping network** (an MLP) that sends the latent code $z \in \mathcal{Z}$ to an *intermediate* latent space $w \in \mathcal{W}$ — "disentangles the latent space and reduces correlations"; affine transforms of $w$ then produce style vectors that control different visual features (hair colour, face shape) at different scales. **2)** A **synthesis network** that starts from a learned $4 \times 4$ constant and upsamples: **AdaIN** (adaptive instance normalization) applies each style vector as scale-and-bias at every conv layer, and explicit **noise injection** at each layer models stochastic detail (freckles, hair placement) "more efficiently than encoding this in the latent code."

**The deck's payoff claim** [recorded]: GANs learn "rich, meaningful latent representations that allow for semantic operations, such as vector arithmetic on faces" — the same embedding arithmetic as §46.2's king − man + woman, now over images.

**Basically, ...** "DCGAN = the GAN idea with Chapter 44's parts: stride instead of pooling, transposed conv to grow the image, batch norm, the right activations. StyleGAN = a two-stage generator: first warp the noise into a better-behaved latent space, then grow the image from a constant while dialling 'style' knobs (and sprinkling noise) at every layer."

## 47.9 When GANs break [recorded]

"Training GANs can be notoriously difficult and unstable." The deck's four failure modes:

- **i) Mode collapse** — "the biggest difficulty." $G$ finds one output that fools $D$ (the deck's example: images of shoes) and produces only that, "forgetting how to generate other classes" — or cycles through a few modes, never learning the full distribution. §47.1(iii)'s missing diversity, in one failure.
- **ii) Non-convergence.** Parameters "oscillate, become unstable, and never converge" — the two networks chase each other forever.
- **iii) Vanishing gradients.** "If the discriminator gets too good too quickly, the generator's gradients can vanish" — §47.6's note, now as a failure mode.
- **iv) Hyperparameter sensitivity.** GANs "require significant fine-tuning" — §47.7's Adam-at-$0.0002$ recipe is typical of how narrow the workable settings are.

**Plausible fixes (the deck's)** [recorded]: **WGAN** (Wasserstein GAN) — measures the Wasserstein distance between the real and generated *distributions* and minimizes it, "strongly incentivizing the generator to produce a wide variety of outputs to match the diversity of the real data, thus significantly reducing mode collapse." **Minibatch discrimination** — lets $D$ see whole batches and measure sample similarity, so a batch of identical fakes "can be easily rejected." **Progressive growing** — start at $4 \times 4$, then add layers to grow $8 \times 8$, $16 \times 16$, $\dots$; "simplifies the learning task at each stage."

**Basically, ...** "GAN training is a balancing act on a knife edge: the forger either collapses to one trick (mode collapse), the critic gets too strong and kills the signal (vanishing gradients), or the two just orbit forever. Fixes: measure distribution distance instead of a classifier score (WGAN), punish boring batches, or grow the image gradually from tiny to full size."

## 47.10 The diffusion pitch: many easy steps

The diffusion deck replays the trilemma and then changes the question: instead of *one* hard mapping noise → image (GAN) or a *lossy* compression (VAE), do the generation in **many easy steps** — learn to fix a *slightly* noisy image, then iterate.

**The origin story (Sohl-Dickstein et al., 2015)** [recorded]: ink diffusing in water. **Forward** (nature): the drop "naturally diffuses until it creates a uniform, chaotic distribution" — easy, irreversible. **Reverse** (learned): undo the diffusion, one small step at a time — the hard part a neural network is asked to learn.

<!-- Diagram: the forward process (image → noise) beside the reverse process (noise → image), as in the DDPM paper's figure (Ho et al., 2020, https://arxiv.org/abs/2006.11239) — the diffusion deck credits this figure to "the DDPM paper." -->

- **i) The forward process $q$** — "the systematic destruction of data." Slowly add Gaussian noise to a real image over $T$ steps. **Fixed: no learning here**, just follow a schedule. Ends at pure noise $x_T \sim \mathcal{N}(0, I)$.
- **ii) The reverse process $p$** — "the learned restoration." Start from pure noise; the network denoises "one step at a time." **This is the trainable part.**

**The deck's conclusion** [recorded]: "We moved from 'One Hard Step' (GANs) to 'Many Easy Steps' (Diffusion). The trade-off: we gained training stability and mode coverage at the cost of inference speed" — and the fundamental insight: "Don't learn to create an image from scratch; just learn to fix a slightly noisy one."

**Basically, ...** "GANs bet everything on one perfect forgery; diffusion takes a thousand tiny safe bets. Destroying an image with noise needs no learning — it's a fixed recipe — so all the learning goes into the one easy skill: remove a little noise. That simplicity is why diffusion trains stably where GANs wobble."

## 47.11 The forward process: the math

**The Markov chain.** Each noisy step depends only on the previous one [recorded]:
$$q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1}).$$

**One step of noise.** At step $t$, draw from a Gaussian centred on the previous image, scaled slightly down [recorded]:
$$\boxed{q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\; \sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\right)}.$$
$\beta_t \in (0, 1)$ is the **variance schedule** — typically increasing, $\beta_1 < \beta_2 < \cdots < \beta_T$ ("noise increases over time").

**The closed-form shortcut (the reparameterization trick).** Reaching $x_{500}$ step by step costs 500 sequential ops — too slow for training. Since sums of Gaussians are Gaussian, there is a one-jump formula. Define $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$; then [recorded]:
$$\boxed{x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)},$$
i.e. $q(x_t \mid x_0) = \mathcal{N}(x_t;\, \sqrt{\bar\alpha_t}\,x_0,\, (1-\bar\alpha_t)I)$. Signal weight $\sqrt{\bar\alpha_t}$ fades while noise weight $\sqrt{1-\bar\alpha_t}$ grows — the same Gaussian-sum algebra as the deck's two-step derivation (two independent Gaussians $a\epsilon_1 + b\epsilon_2$ merge into $\sqrt{a^2+b^2}\,\epsilon$).

**The schedule.** The standard linear schedule: $T = 1000$ steps, $\beta$ rising linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$. As $\beta_t$ grows, $\bar\alpha_t \to 0$, and by step $T$, $x_T \approx \mathcal{N}(0, I)$ — pure noise [recorded].

**Basically, ...** "Each forward step keeps most of the old image and mixes in a little Gaussian noise ($\beta_t$ controls how much). Because Gaussians add nicely, you can skip the whole chain: jump straight from the clean image to step $t$ in one formula. Crank $\beta$ up over 1000 steps and the image melts into pure static."

## 47.12 eg: two steps of noise, one jump [verified-NumPy]

Own toy: $\beta_1 = 0.2$, $\beta_2 = 0.25$; take a scalar "image" $x_0 = 1.0$ and sampled noise $\epsilon = 0.7$.

i) $\alpha_1 = 0.8$, $\alpha_2 = 0.75$, so $\bar\alpha_2 = 0.8 \times 0.75 = 0.6$.

ii) Jump directly: $x_2 = \sqrt{0.6}\cdot 1.0 + \sqrt{1-0.6}\cdot 0.7 = 0.7746 + 0.6325 \times 0.7 = 0.7746 + 0.4427 = 1.2173$.

No 2-step chain needed — the formula carries both noise injections at once. At $t = 2$ the signal still outweighs the noise ($\bar\alpha_2 = 0.6$); at $t = 1000$ of the real schedule the signal weight is essentially zero.

**Basically, ...** "The $\bar\alpha$ bookkeeping says: after two steps, 60% of the variance is still signal, 40% is noise — and the one-line formula lands you at $x_2 = 1.2173$ without ever computing $x_1$."

## 47.13 The reverse process: learning to denoise

**The model.** If the forward steps are small, the reverse step is (approximately) Gaussian too. Learn it with a network [recorded]:
$$\boxed{p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\; \mu_\theta(x_t, t),\; \Sigma_\theta(t)\right)},$$
input: noisy image $x_t$ and timestep $t$; output: the mean of the slightly cleaner $x_{t-1}$. The variance $\Sigma_\theta$ is usually fixed to a constant (e.g. $\beta_t$) — the deck keeps it simple and predicts only the mean.

**The training target (the clever workaround).** The true reverse $q(x_{t-1} \mid x_t)$ is intractable — it would need the distribution of *all* images. But during training we *know* $x_0$, so the posterior $q(x_{t-1} \mid x_t, x_0)$ is tractable by Bayes' rule [recorded]:
$$q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1})\,q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)} = \mathcal{N}(x_{t-1};\, \tilde\mu_t(x_t, x_0),\, \tilde\beta_t I).$$
A product of Gaussians is a Gaussian; grinding the algebra gives the target mean [recorded]:
$$\tilde\mu_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right).$$
**Key insight** (the deck's): $x_t$ is known to the network; the *only unknown* in that formula is $\epsilon$ — the noise that was added. "To predict the previous image mean $\tilde\mu_t$, the neural network only needs to predict the noise $\epsilon$."

**The loss collapses to MSE.** KL divergence between the ground-truth Gaussian and $p_\theta$, with fixed variances, reduces to matching the means; reparameterizing the mean through the predicted noise $\epsilon_\theta$ cancels the constants [recorded]:
$$\boxed{L_{\text{simple}} = \big\|\epsilon - \epsilon_\theta\!\left(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\; t\right)\big\|^2}.$$
"Take a clean image, add noise, ask the U-Net 'what noise was added?', MSE between actual and predicted noise. We are literally teaching the model to separate signal from noise."

**Training (Algorithm 1)** [recorded]: repeat — sample $x_0$ from the data, sample $t$ uniformly from $\{1, \dots, T\}$, sample $\epsilon \sim \mathcal{N}(0, I)$, gradient-descent step on $L_{\text{simple}}$. Note the efficiency: training never unrolls the full chain — one random timestep per step.

**Sampling (Algorithm 2)** [recorded]: start $x_T \sim \mathcal{N}(0, I)$; for $t = T, \dots, 1$: compute $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t)\right)$, sample $z \sim \mathcal{N}(0, I)$ (but $z = 0$ at the final step $t = 1$), set $x_{t-1} = \mu_\theta + \sigma_t z$. Return $x_0$. The re-injected noise $z$ is "Langevin dynamics" — it keeps samples diverse instead of collapsing to one average image.

**Basically, ...** "The network never predicts the clean image directly — it predicts the *noise* that was added, and subtracting the predicted noise is the denoising step. Training is one tidy loop: pick a random moment in the melting process and fix it. Generation runs the film backwards, adding a whisper of fresh noise each frame so every sample comes out different."

## 47.14 The denoiser: a U-Net that knows what time it is

The noise predictor $\epsilon_\theta$ is a **modified U-Net** — §44.9's encoder-decoder with skip connections, now predicting noise instead of segment masks [recorded]:

- **Structure.** Encoder downsamples ($64 \times 64 \to 8 \times 8$ in the deck's figure), bottleneck, decoder upsamples; **skip connections** carry fine detail from encoder to decoder (the §44.9 justification, verbatim).
- **Details (the deck's table, condensed)** [recorded]: **GroupNorm** (better than BatchNorm for small batches), **SiLU/Swish** activation ("smoother gradients than ReLU"), self-attention at $16 \times 16$ resolution ("captures long-range dependencies"), and **noise $\epsilon$ as the prediction target** ("more stable than predicting $x_0$").

**The time problem.** One network must denoise at *every* noise level — but "timestep is just a number," and raw $t = 500$ vs $501$ "look too different" while they "should be similar." The fix: **sinusoidal time embeddings**, the §46.10 trick applied to timesteps [recorded]:
$$\text{emb}(t)_i = \begin{cases}
\sin\!\left(\dfrac{t}{10000^{2i/d}}\right) & i\ \text{even},\\[6pt]
\cos\!\left(\dfrac{t}{10000^{(2i-1)/d}}\right) & i\ \text{odd}.
\end{cases}$$
Low dimensions oscillate fast (fine time differences), high dimensions slowly (coarse), all values bounded in $[-1, 1]$, deterministic — no learning needed.

**The pipeline** [recorded]: scalar $t \to$ sinusoidal vector $\mathbb{R}^{128} \to$ 2-layer MLP (Linear + SiLU) $\to$ $\mathbb{R}^{512}$ embedding $\text{temb}$. The embedding is broadcast and **added to the feature maps in every residual block**, "modulating the network's behaviour" — the same embedding at all spatial locations. The MLP "learns a task-specific representation" on top of the fixed sinusoids.

**Basically, ...** "The denoiser is a U-Net (§44.9's shape, new job) that also gets told the clock: a sine-wave fingerprint of the timestep, refined by a small MLP, mixed into every layer — so one network can behave like 1000 different denoisers, one per noise level."

## 47.15 eg: a time embedding, by hand [verified-NumPy]

Use the deck's formula with $d = 4$: $i = 0$ gives $\sin(t)$, $i = 1$ gives $\cos(t/10)$, $i = 2$ gives $\sin(t/10000)$, $i = 3$ gives $\cos(t/10000^{1.25})$.

i) $t = 0$: $[0,\ 1,\ 0,\ 1]$.

ii) $t = 1$: $[\sin 1,\ \cos 0.1,\ \sin 0.0001,\ \cos(1 \times 10^{-5})] = [0.8415,\ 0.9950,\ 0.0001,\ 1.0000]$.

Two adjacent timesteps get nearly identical embeddings in the slow dimensions and visibly different ones in the fast dimension — exactly the "nearby timesteps should be similar" property §47.14 demanded.

**Basically, ...** "Same sine-wave trick as §46.10's positional encoding, pointed at time instead of position: $t = 0$ and $t = 1$ are neighbours, and their fingerprints say so."

## 47.16 DDIM: sampling 20–100× faster

**The bottleneck.** Sampling walks $1000 \to 999 \to \cdots \to 0$ — strictly sequential ("to calculate step 999, we must have the result of step 1000"), one heavy U-Net call per step. "Generating one image requires running the heavy neural network 1000 times" [recorded].

**The insight (Song et al., 2020)** [recorded]: "Does the forward process HAVE to be random?" Redefine the schedule so the noise→image mapping is **deterministic** — no randomness, yet the *same training objective* holds. The trajectory becomes smooth and predictable, "we don't need to take baby steps. We can take huge strides." **Key benefit: use the exact same pre-trained DDPM model — no retraining required.**

**The update rule** [recorded]:
$$\boxed{x_{t-1} = \underbrace{\sqrt{\bar\alpha_{t-1}}\;\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}}_{\text{predicted } \hat x_0} + \underbrace{\sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta(x_t, t)}_{\text{direction pointing to } x_t}}$$
Read it as a weighted mix: $\hat x_0$ (the clean image the network thinks it sees) scaled by $\sqrt{\bar\alpha_{t-1}}$, plus the predicted noise scaled by $\sqrt{1-\bar\alpha_{t-1}}$ — "we mix clean image and noise according to the noise schedule at $t-1$."

**Skipping steps.** Pick a subsequence $\tau = [1000, 950, 900, \dots, 0]$ — 20 steps instead of 1000; the deck's timing table [recorded]: DDPM 1000 steps ≈ 20 s vs DDIM 20–50 steps ≈ 1 s. "DDIM allows us to trade a tiny bit of quality for a massive gain in speed."

**DDIM inversion: image → noise.** Determinism buys reversibility: run the update *backwards* and a real image maps to the specific noise $x_T$ that would have generated it [recorded]. That noise "encodes the original image" — composition, structure, layout — so you can edit: invert a cat photo, change the prompt to "dog," denoise, and get "a dog in the same pose/composition as the original cat." DDPM cannot do this ("due to randomness"). Pros/cons [recorded]: fast, deterministic, invertible, enables editing — but less diversity, small inversion errors accumulate, and a quality-vs-speed tradeoff.

**Basically, ...** "DDPM rolls dice at every step, so it must creep forward. DDIM removes the dice — same trained model, same loss — and the path becomes a smooth, predictable road you can drive in 20 strides instead of 1000 baby steps. And because nothing is random, you can drive it backwards: photo in, meaningful noise out, edit, drive forward again."

## 47.17 eg: one DDIM step, by hand [verified-NumPy]

Own numbers: $\bar\alpha_t = 0.5$, $\bar\alpha_{t-1} = 0.6$, $x_t = 1.0$, $\epsilon_\theta(x_t, t) = 0.4$.

i) Predicted clean image: $\hat x_0 = \dfrac{1.0 - \sqrt{1-0.5}\times 0.4}{\sqrt{0.5}} = \dfrac{1.0 - 0.2828}{0.7071} = 1.0142$.

ii) DDIM step: $x_{t-1} = \sqrt{0.6}\times 1.0142 + \sqrt{1-0.6}\times 0.4 = 0.7746 \times 1.0142 + 0.6325 \times 0.4 = 0.7856 + 0.2530 = 1.0386$.

No randomness anywhere — rerun this and you get $1.0386$ again. That determinism is the whole point.

**Basically, ...** "Predict the clean image (1.0142), then re-mix it with the predicted noise at the *previous* step's schedule weights — landing at 1.0386. Same inputs, same output, every time: the anti-DDPM."

## 47.18 Guidance: steering the denoiser

So far the model generates *anything*. Guidance steers it: "generate only dogs," "a dog wearing sunglasses."

**Interlude: the score function** [recorded]. The score is the gradient of the log-probability: $\text{score}(x_t) = \nabla_{x_t} \log p(x_t)$ — "which direction increases the probability of $x_t$?" Noise prediction and score prediction are the same thing in different clothes [recorded]:
$$\nabla_{x_t} \log p(x_t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}}.$$
Score space makes guidance math cleaner: conditioning = adding a gradient that points uphill toward the desired class.

**Classifier guidance** [recorded]: train a plain unconditional diffusion model, then at each denoising step ask a separate classifier "does this look like class $y$?" and push along its gradient. By Bayes:
$$\boxed{\nabla_{x_t} \log p(x_t \mid y) = \underbrace{\nabla_{x_t} \log p(x_t)}_{\text{unconditional score}} + \lambda \cdot \underbrace{\nabla_{x_t} \log p(y \mid x_t)}_{\text{classifier gradient}}}$$
$\lambda$ is the guidance scale. Pros: simple, works with existing models. Cons: needs a *separate* classifier that must work on *noisy* images, limited to the classifier's fixed classes — "can't use text prompts."

**Classifier-free guidance (CFG)** [recorded] — the current standard (DALL·E 2, Imagen, Stable Diffusion): train *one* model to do both conditional and unconditional generation by randomly **dropping the condition 10% of the time** (null condition $\emptyset$). Rewrite the classifier-gradient term via Bayes — $\nabla_{x_t}\log p(y \mid x_t) = \nabla_{x_t}\log p(x_t \mid y) - \nabla_{x_t}\log p(x_t)$ — and substitute:
$$\nabla_{x_t}\log p(x_t \mid y) = (1-\lambda)\nabla_{x_t}\log p(x_t) + \lambda\,\nabla_{x_t}\log p(x_t \mid y),$$
in noise space [recorded]:
$$\boxed{\epsilon_{\text{guided}} = \epsilon_{\text{uncond}} + \lambda \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})}.$$
Read it as a vector: find the direction from unconditional to conditional, amplify by $\lambda$ — "higher $\lambda$ → stronger adherence to condition." The conditioning vector $c$ (text embedding, class label) is injected into the U-Net "just like the time embedding in DDPM" (§47.14).

**Classifier vs classifier-free (the deck's table, condensed)** [recorded]: classifier guidance — needs an external model, fixed classes, one forward pass + classifier cost, "less common" today. CFG — single model, "any condition (text, image, etc.)," two forward passes per step, typically better quality, "the dominant approach in modern text-to-image models."

**How the text gets in: cross-attention.** At each U-Net layer [recorded]: $Q = W_Q z_t$ (from the image latent), $K = W_K c$, $V = W_V c$ (from the text embedding), then §46.5's formula $\text{Attention} = \text{softmax}(QK^\top/\sqrt{d})\,V$. "Image asks: 'What features should I have here?' Text provides: 'A cat should have whiskers, fur texture...'"

**Basically, ...** "Unguided diffusion is a slot machine. Classifier guidance bolts on a teacher that nudges each step toward 'dog' — but the teacher only knows its ten classes. Classifier-free guidance trains one model on both regimes (condition dropped 10% of the time) and steers by the *difference*: push from 'whatever' toward 'what the prompt says,' scaled by $\lambda$. Text reaches the image through cross-attention — §46.7's mechanism, pointed at words instead of encoder states."

## 47.19 Latent diffusion: Stable Diffusion

**The problem.** DDPM diffuses in pixel space: a $512 \times 512 \times 3$ image is $786{,}432$ dimensions, processed *every* denoising step — "computational cost... memory intensive... slow training... slow generation" [recorded]. Key observation: "not all pixels are equally important — many pixels are redundant (e.g., clear blue sky)."

**The fix.** Compress first, diffuse the compression, decompress last. An autoencoder (encoder $\mathcal{E}$, decoder $\mathcal{D}$, trained with reconstruction + regularization) squeezes $512 \times 512 \times 3$ to a $64 \times 64 \times 4$ latent [recorded]: $786{,}432$ dims → $16{,}384$ dims — a **48× reduction** [verified-NumPy]. Latent space is "like JPEG compression... even more compressed, captures 'essence.'"

<!-- Diagram: the latent diffusion architecture — image → VAE encoder → latent diffusion (U-Net + cross-attention with text) → VAE decoder → image; the standard figure from Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," https://arxiv.org/abs/2112.10752, the architecture behind Stable Diffusion. -->

**Training (two stages)** [recorded]: **1)** train the autoencoder (VAE) on a large image dataset until $\mathcal{D}(\mathcal{E}(x)) \approx x$ with small latents, then **freeze** it; **2)** encode images ($z = \mathcal{E}(x)$), add noise in latent space ($z_t = \sqrt{\bar\alpha_t}\,z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$), and train the U-Net to predict the noise: $L = \|\epsilon - \epsilon_\theta(z_t, t, c)\|^2$, with condition $c$ = text embeddings, class labels, etc.

**Generation** [recorded]: sample $z_T \sim \mathcal{N}(0, I)$ at $64 \times 64 \times 4$ (not $512 \times 512 \times 3$!), take a prompt $c$ ("a cat on a sofa"), denoise $z_T \to z_0$ with the conditioned U-Net, decode $x = \mathcal{D}(z_0)$. Full pipeline: $\text{Text} \xrightarrow{\text{CLIP}} c \to \text{U-Net}(z_t, t, c) \to z_0 \xrightarrow{\text{Decoder}} \text{Image}$, training objective $\mathbb{E}_{z,\epsilon,t,c}\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2$ [recorded].

**Stable Diffusion in practice** [recorded]: "Latent Diffusion Model released by Stability AI" — VAE compression factor $f = 8$ ($512 \times 512 \to 64 \times 64$), 4 latent channels, CLIP ViT-L/14 text encoder (SD 1.x/2.x), cross-attention at multiple U-Net resolutions, trained on LAION-5B; "first high-quality, open-source text-to-image model that can run on consumer GPUs (RTX 3090, even RTX 3060)." Versions (deck's table, condensed): SD 1.4 (Aug 2022, first public, $512^2$) → 1.5 → 2.0 (OpenCLIP encoder) → 2.1 → SDXL 1.0 (Jul 2023, $1024^2$, dual encoders, refiner) → SD 3.0 (2024, diffusion-transformer architecture) — "all based on Latent Diffusion Models (LDM) architecture."

**Basically, ...** "Denoising a full-resolution photo 50 times is expensive, so cheat: shrink the photo 48× with a VAE, do all the noisy work on the tiny version, then blow it back up. Same math, same training loop (§47.13), one frozen compressor — and suddenly a consumer GPU can paint 'a cat on a sofa.'"

## 47.20 Judging the forgeries: FID, IS, CLIP

"Evaluating generative models is like evaluating art" — no single correct output, so the field settled on statistics [recorded].

**i) FID — Fréchet Inception Distance** [recorded]. Pass real and generated images through Inception-v3, fit a Gaussian to each feature set, and measure the Fréchet distance between the Gaussians:
$$\boxed{\text{FID} = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\right)}.$$
$\mu_r, \Sigma_r$ = real-image feature statistics; $\mu_g, \Sigma_g$ = generated. **Lower is better** ($0$ = identical distributions); sensitive to *both* quality and diversity; "most widely used metric in practice"; good FID $< 10$ (domain-dependent).

**eg (own — [verified-NumPy]).** $\mu_r = (0, 0)$, $\mu_g = (1, 0)$, $\Sigma_r = \Sigma_g = I_2$: $(\Sigma_r\Sigma_g)^{1/2} = I$, so the trace term is $\mathrm{Tr}(I + I - 2I) = 0$ and $\text{FID} = \|(-1, 0)\|^2 = 1.0$. Identical spreads, shifted means — the distance is just the squared shift.

**ii) Inception Score (IS)** [recorded]. Uses a pretrained Inception-v3 classifier — no real images needed. Two desiderata: each generated image should be *clearly classifiable* (low-entropy $p(y \mid x)$, quality) and the set should span *many classes* (high-entropy marginal $p(y)$, diversity):
$$\boxed{\text{IS} = \exp\!\left(\mathbb{E}_x[\mathrm{KL}(p(y \mid x)\,\|\,p(y))]\right)}.$$
**Higher is better** (typical range $2$–$10$). Limitation: "biased toward ImageNet classes, doesn't compare to real data."

**iii) CLIP score** [recorded]. For text-to-image: embed the image and the caption with CLIP ("contrastive language-image pre-training," trained on 400M image-text pairs, shared image-text embedding space) and take cosine similarity:
$$\boxed{\text{CLIPScore} = \max\!\big(100 \cdot \cos(E_I(x), E_T(c)),\, 0\big)}.$$
Measures image–text *alignment*, not realism alone.

**Which metric when (the deck's table, condensed)** [recorded]: FID — distribution similarity to real images — best for overall unconditional quality+diversity (needs large samples; implementation-sensitive). IS — quality + diversity — best for class-conditional generation (ignores real data; ImageNet-biased). CLIP — image-text alignment — best for text-to-image (doesn't measure realism alone). In practice: combine them (FID + IS unconditional; FID + CLIP text-conditional) — and "human evaluation still crucial."

**Basically, ...** "FID asks: do the generated photos have the same *statistics* as real photos? (Lower wins.) Inception Score asks: is each image confidently one thing, and are the images many different things? (Higher wins.) CLIP score asks: does the image match its caption? (Higher wins.) No single number tells the truth — use them together, and trust human eyes last."

## 47.21 Where this goes next: closing Part VI (and the book)

i) **The generator census.** Four ways to generate, four training signals: **1)** the autoregressive decoder (§46.12, via §45.13(iii)) — trained by next-token likelihood, sampling one token at a time; **2)** the GAN generator (§47.4) — trained by a learned critic, one shot from noise; **3)** the diffusion denoiser (§47.13) — trained by noise-prediction MSE, a thousand small steps; **4)** the fast variants — DDIM's deterministic skips (§47.16) and the latent compressor (§47.19). All four are differentiable modules (§42.4's machinery); none needed new optimization beyond §43.12's Adam family (§47.7's GAN used Adam at $0.0002$).

ii) **The Part VI arc, complete.** Chapter 41 built the neuron; 42 derived backprop; 43 tuned the optimizer; 44 shared weights across space; 45 across time; 46 let any position talk to any other; 47 turns the whole machine around and *generates* — noise in, data out. The book's one-stop promise lands here: from a single dense layer to Stable Diffusion's text-to-image pipeline, every step traced to a committed source and every number re-checked.

iii) **The debugging playbook's last word.** §40.5(i) — "shapes first" — closes the loop with this chapter: a GAN that dies at step 1 is almost always a 784-vs-28×28 reshape or a tanh-vs-sigmoid range mismatch (§47.7); a diffusion model that explodes is a schedule or time-embedding bug (§§47.11, 47.14). Same discipline, new architectures.

iv) **Beyond the course.** The decks stop at the fundamentals; the live field keeps moving — through the supplementary shelf in GOAL.md for the follow-up reading. The foundation in this chapter — the forward/reverse split, the noise-prediction loss, guidance as a vector in score space — is the vocabulary those readings are written in.

## Problem set

1. **The 2×2 marginal.** The deck's eight configurations with $x_{11} = 1$ have probabilities $0.05, 0.10, 0.15, 0.03, 0.02, 0.08, 0.07, 0.01$. (i) Compute $P(x_{11} = 1)$ by marginalization. (ii) Given $x_{11} = 1$ and the rest missing, which completion $(x_{12}, x_{21}, x_{22})$ maximizes $P(x_{\text{mis}}, x_{\text{obs}})$? (iii) In one line, say why the same computation is impossible for a $32 \times 32$ binary image.
2. **GAN losses by hand.** Real outputs $D(x) = [0.8,\ 0.95]$, fake outputs $D(G(z)) = [0.4,\ 0.2]$. (i) Compute the discriminator's BCE loss. (ii) Compute the generator's loss in the minimax form. (iii) Compute the generator's non-saturating loss and explain in one line why it is larger here.
3. **The notebook's parameter counts.** The generator is $\text{Linear}(64 \to 256)$, $\text{Linear}(256 \to 256)$, $\text{Linear}(256 \to 784)$ with biases; the discriminator is $\text{Linear}(784 \to 256)$, $\text{Linear}(256 \to 256)$, $\text{Linear}(256 \to 1)$ with biases. (i) Compute both totals. (ii) Which network is larger, and by how many parameters? (iii) In one line, say why activations and batch-norm parameters (none here) don't change the count.
4. **The Nash point.** (i) At Nash, $D$ outputs $0.5$ on every input: evaluate the minimax value $V(D, G) = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$. (ii) Early in training $D(G(z)) = 0.05$: compute the generator's non-saturating loss $-\log D(G(z))$. (iii) In two lines, connect (ii) to the §47.6 Note — which failure mode does the non-saturating loss patch?
5. **Forward diffusion numbers.** $\beta_1 = 0.1$, $\beta_2 = 0.2$. (i) Compute $\bar\alpha_2$. (ii) Write $q(x_2 \mid x_0)$ as $\mathcal{N}(\cdot\, x_0,\, \cdot\, I)$ with numbers. (iii) How much signal variance remains at $t = 2$ vs noise variance?
6. **The linear schedule's endpoints.** $\beta_1 = 10^{-4}$, $\beta_{1000} = 0.02$, linear in $t$. (i) Compute $\beta_{500}$. (ii) The signal weight is $\sqrt{\bar\alpha_t}$: compute it at $t = 500$ and at $t = 1000$ (values: $\bar\alpha_{500} \approx 0.0786$, $\bar\alpha_{1000} \approx 4.04 \times 10^{-5}$). (iii) In one line, state what $x_{1000}$ is distributed as.
7. **One DDIM step.** $\bar\alpha_t = 0.7$, $\bar\alpha_{t-1} = 0.75$, $x_t = 0.5$, $\epsilon_\theta(x_t, t) = -0.3$. (i) Compute the predicted clean image $\hat x_0$. (ii) Compute $x_{t-1}$ via the DDIM update. (iii) In one line, say what changes in the answer if the network predicts $\epsilon_\theta = 0$.
8. **FID by hand.** $\mu_r = (0, 0)$, $\mu_g = (2, 0)$, $\Sigma_r = \Sigma_g = I_2$. (i) Compute the FID. (ii) Recompute with $\mu_g = (0, 0)$. (iii) In two lines, say which of the deck's desiderata — quality or diversity — a perfect FID of $0$ can still hide.
9. **Classifier-free guidance numbers.** $\epsilon_{\text{uncond}} = (0.2, -0.1)$, $\epsilon_{\text{cond}} = (0.8, 0.3)$, guidance scale $\lambda = 3$. (i) Compute $\epsilon_{\text{guided}}$. (ii) Recompute with $\lambda = 1$ and $\lambda = 0$. (iii) In one line, say what $\lambda = 0$ generation corresponds to.
10. **Inception Score by hand.** Three generated images with $p(y \mid x)$: $(0.9, 0.1)$, $(0.8, 0.2)$, $(0.15, 0.85)$. (i) Compute the marginal $p(y)$. (ii) Compute the three KL divergences. (iii) Compute the IS. (iv) In one line, say whether this model's weakness is quality or diversity, and why.
11. **The book-ender.** In one paragraph, name the four generator families the book built (§46.12's decoder, the GAN generator, the diffusion denoiser, the latent-diffusion pipeline) and for each say what the generator *is* and what training signal teaches it.

---

*Sources: GenAI Week 6 "Generative Models and GAN" deck (Balaji Srinivasan, Ganapathy Krishnamurthi), 65 slides — `sources/course materials/GenAI-20261009T001550Z-1-001/GenAI/Slides/Week - 6/Generative Models and GAN - Lecture Slides.pdf` (the 225-page PDF is the same 65 slides with beamer overlays) and the notes PDF `.../GenAI/Notes/Week6-Generative Models and GAN .pdf` (identical content, used for cross-check) — generative-model definition and goal, 2×2 toy grid (marginalization numbers $0.05$–$0.01$, inpainting argmax $P(1,0,1,0) = 0.15$), computational bottleneck ($2^{1024}$), generative classifier (Naive Bayes), latent-space paradigm shift (PCA/LDA/SVD examples), explicit/implicit taxonomy, minimax objective and D/G objectives, non-saturating equivalence, Nash equilibrium, two-phase training algorithm, DCGAN guidelines, StyleGAN (mapping/synthesis networks, AdaIN, noise injection, latent arithmetic), training challenges (mode collapse, non-convergence, vanishing gradients, hyperparameter sensitivity) and fixes (WGAN, minibatch discrimination, progressive growing), learnopencv.com figure credits; `GAN_from_Scratch.ipynb` — vanilla GAN on 60{,}000 grayscale $28\times28$ images: latent $64$, hidden $256$, image $784$; generator (LeakyReLU $0.2$, $\tanh$; 283{,}920 params) and discriminator (LeakyReLU $0.2$, sigmoid; 267{,}009 params) architectures; BCE loss; Adam $\eta = 0.0002$ with separate optimizers; $[-1,1]$ normalization + $(x+1)/2$ denormalization; `detach()` on D's fake pass; non-saturating G via BCE against real labels; 100 epochs $\times$ 600 steps = 60{,}000 updates; avg $L_D = 0.7651$, avg $L_G = 2.0232$, final $0.8781$ / $1.5290$, ratio $1.7413$; early-epoch $D(x) \approx 0.95$, $D(G(z)) \approx 0.05$ (all torch code and numbers [recorded]); GenAI Week 11 "Introduction to Diffusion Models" deck (same authors), 94 slides — `.../Slides/Week - 11/Week11_GenAI_Diffusion20251127.pdf` (339-page PDF with beamer overlays, text-extractable) — generative trilemma (VAE vs GAN table), Sohl-Dickstein 2015 origin, forward/reverse definitions, Markov chain, $q(x_t\mid x_{t-1})$ formula, $\alpha_t$/$\bar\alpha_t$ notation, reparameterization derivation, linear schedule ($\beta_1 = 10^{-4} \to \beta_T = 0.02$, $T = 1000$), reverse model $p_\theta$, Bayes posterior and $\tilde\mu_t$ formula, KL→MSE, $L_{\text{simple}}$, Algorithms 1–2 (incl. $z = 0$ at $t = 1$, Langevin noise), U-Net architecture table (GroupNorm, SiLU, $16\times16$ attention, skip connections), sinusoidal time-embedding formula and MLP pipeline, DDIM (Song et al. 2020; deterministic update, step-skipping, 20–50 steps $\approx$ 1 s vs 1000 $\approx$ 20 s, inversion, cat→dog editing example, pros/cons), score function and noise–score link, classifier guidance formula + pros/cons, classifier-free guidance (10% condition dropout, $\epsilon_{\text{guided}}$ formula, comparison table), cross-attention $Q = W_Q z_t$, $K = W_K c$, $V = W_V c$, latent diffusion (48× claim: $786{,}432 \to 16{,}384$; two-stage training; generation pipeline; training objective), Stable Diffusion specs (f $= 8$, 4 latent channels, CLIP ViT-L/14, LAION-5B) and versions table, FID/IS/CLIP formulas and comparison table, evaluation-metrics guidance (all [recorded]); diagram references — Goodfellow et al. 2014 (https://arxiv.org/abs/1406.2661), Ho et al. 2020 DDPM (https://arxiv.org/abs/2006.11239), Song et al. 2020 DDIM (https://arxiv.org/abs/2010.08092), Rombach et al. 2021 LDM (https://arxiv.org/abs/2112.10752); book chapters 24 (PCA), 30 (Naive Bayes), 40 (§40.5(i)), 41–43 (§§42.4, 43.12), 44 (§§44.6, 44.9, 44.10), 45 (§§45.1, 45.10, 45.11, 45.13), 46 (§§46.2, 46.5, 46.7, 46.10, 46.12, 46.16).*
