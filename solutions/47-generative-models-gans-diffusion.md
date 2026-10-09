# Solutions — Chapter 47: Generative models: GANs and diffusion

Full worked solutions. All numbers re-run in numpy [verified-NumPy]; deck/notebook transcriptions marked [recorded].

## 1. The 2×2 marginal

(i) $P(x_{11} = 1)$ sums the eight configurations with $x_{11} = 1$: $0.05 + 0.10 + 0.15 + 0.03 + 0.02 + 0.08 + 0.07 + 0.01 = 0.51$.

(ii) The denominator $P(x_{\text{obs}})$ is constant across candidates, so maximize the joint: the deck's table gives $P(1, 0, 1, 0) = 0.15$ as the largest of the eight, hence $(x_{12}, x_{21}, x_{22}) = (0, 1, 0)$.

(iii) A $32 \times 32$ binary image has $2^{1024}$ configurations — half the pixels missing means summing over $2^{512}$ candidates for the denominator, "computationally impossible" (§47.2).

## 2. GAN losses by hand

(i) $L_D = -\tfrac{1}{4}[\log 0.8 + \log(1-0.4) + \log 0.95 + \log(1-0.2)] = -\tfrac{1}{4}[-0.2231 - 0.5108 - 0.0513 - 0.2231] = 0.2521$.

(ii) Minimax: $L_G = -\tfrac{1}{2}[\log(1-0.4) + \log(1-0.2)] = -\tfrac{1}{2}[-0.5108 - 0.2231] = 0.3670$.

(iii) Non-saturating: $L_G = -\tfrac{1}{2}[\log 0.4 + \log 0.2] = -\tfrac{1}{2}[-0.9163 - 1.6094] = 1.2629$. It is larger because $D$ is confident the fakes are fake ($0.4, 0.2 \ll 0.5$), so labeling them "real" incurs a steep penalty — the steep slope is exactly the usable gradient the minimax form lacks here.

## 3. The notebook's parameter counts

(i) Generator: $(64\cdot256 + 256) + (256\cdot256 + 256) + (256\cdot784 + 784) = 16{,}640 + 65{,}792 + 201{,}488 = 283{,}920$. Discriminator: $(784\cdot256 + 256) + (256\cdot256 + 256) + (256\cdot1 + 1) = 200{,}960 + 65{,}792 + 257 = 267{,}009$.

(ii) The generator is larger, by $283{,}920 - 267{,}009 = 16{,}911$ parameters.

(iii) Activations (LeakyReLU, $\tanh$, sigmoid) have no learnable parameters; biases were included in each layer's count ($+256$, $+784$, $+1$).

## 4. The Nash point

(i) $V = \mathbb{E}[\log 0.5] + \mathbb{E}[\log(1-0.5)] = 2 \log 0.5 = -1.3863$ (natural log).

(ii) $-\log 0.05 \approx 2.9957$.

(iii) At (ii) the minimax generator term would be $\log(1-0.05) \approx -0.0513$ — nearly flat, the §47.6 Note's vanishing-gradient failure (critic too strong, forger gets no signal). The non-saturating loss instead stays steep ($\approx 2.9957$), patching that failure mode.

## 5. Forward diffusion numbers

(i) $\alpha_1 = 0.9$, $\alpha_2 = 0.8$, $\bar\alpha_2 = 0.9 \times 0.8 = 0.72$.

(ii) $q(x_2 \mid x_0) = \mathcal{N}(x_2;\, \sqrt{0.72}\,x_0,\, (1-0.72)I) = \mathcal{N}(0.8485\,x_0,\, 0.28\,I)$.

(iii) Signal variance $0.72$ vs noise variance $0.28$ — $72\%$ of the variance is still the original signal at $t = 2$.

## 6. The linear schedule's endpoints

(i) $\beta_t = 10^{-4} + (0.02 - 10^{-4})\frac{t-1}{999}$: $\beta_{500} = 10^{-4} + 0.0199 \times \frac{499}{999} \approx 0.0100$.

(ii) $\sqrt{\bar\alpha_{500}} = \sqrt{0.0786} \approx 0.2803$ — about $28\%$ of the signal's scale survives at the midpoint. At $t = 1000$: $\sqrt{\bar\alpha_{1000}} = \sqrt{4.04 \times 10^{-5}} \approx 0.0064$ — effectively gone.

(iii) $x_{1000} \sim \mathcal{N}(0, I)$: pure noise, the §47.11 claim verified numerically.

## 7. One DDIM step

(i) $\hat x_0 = \dfrac{0.5 - \sqrt{1-0.7}\times(-0.3)}{\sqrt{0.7}} = \dfrac{0.5 + 0.1643}{0.8367} = \dfrac{0.6643}{0.8367} = 0.7940$.

(ii) $x_{t-1} = \sqrt{0.75}\times 0.7940 + \sqrt{1-0.75}\times(-0.3) = 0.8660 \times 0.7940 - 0.5 \times 0.3 = 0.6876 - 0.15 = 0.5376$.

(iii) If $\epsilon_\theta = 0$, then $\hat x_0 = x_t/\sqrt{\bar\alpha_t}$ (the network claims the image is already clean, just rescaled) and the step is $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0$ — pure rescaling, no noise mixing.

## 8. FID by hand

(i) $(\Sigma_r\Sigma_g)^{1/2} = I_2$, so the trace term is $\mathrm{Tr}(I + I - 2I) = 0$; $\text{FID} = \|(0,0)-(2,0)\|^2 = 4.0$.

(ii) With $\mu_g = (0,0)$: $\text{FID} = 0$ — identical Gaussians.

(iii) FID measures distribution *statistics*: a model that memorizes and regurgitates a small set of real images can score near $0$ while having no true diversity (the deck's warning that FID is "sensitive to both quality and diversity" cuts both ways — here, memorization games the diversity side).

## 9. Classifier-free guidance numbers

(i) $\epsilon_{\text{guided}} = (0.2, -0.1) + 3\times\big((0.8, 0.3) - (0.2, -0.1)\big) = (0.2, -0.1) + 3\times(0.6, 0.4) = (2.0,\ 1.1)$.

(ii) $\lambda = 1$: $(0.2, -0.1) + (0.6, 0.4) = (0.8, 0.3) = \epsilon_{\text{cond}}$ — pure conditional. $\lambda = 0$: $(0.2, -0.1) = \epsilon_{\text{uncond}}$ — pure unconditional.

(iii) $\lambda = 0$ ignores the prompt entirely: unguided generation (§47.18's slot machine).

## 10. Inception Score by hand

(i) $p(y) = \frac{1}{3}\big[(0.9, 0.1) + (0.8, 0.2) + (0.15, 0.85)\big] = (0.6167,\ 0.3833)$.

(ii) $\mathrm{KL}_1 = 0.9\log\frac{0.9}{0.6167} + 0.1\log\frac{0.1}{0.3833} = 0.9(0.3788) + 0.1(-1.3445) = 0.2059$; $\mathrm{KL}_2 = 0.8\log\frac{0.8}{0.6167} + 0.2\log\frac{0.2}{0.3833} = 0.8(0.2595) + 0.2(-0.6505) = 0.0781$; $\mathrm{KL}_3 = 0.15\log\frac{0.15}{0.6167} + 0.85\log\frac{0.85}{0.3833} = 0.15(-1.4145) + 0.85(0.7962) = 0.4648$.

(iii) Mean KL $= (0.2059 + 0.0781 + 0.4648)/3 = 0.2496$; $\text{IS} = \exp(0.2496) = 1.2835$.

(iv) **Diversity** is the weakness: each image is confidently one class (good quality), but two of three images fall in the same class — the marginal $(0.6167, 0.3833)$ is skewed, so the model is near a mode-collapse regime (§47.9).

## 11. The book-ender

Accept any well-argued paragraph along these lines: **1)** the §46.12 transformer decoder — a next-token distribution trained by likelihood on sequences, generating autoregressively; **2)** the GAN generator — a noise→image map trained adversarially by a discriminator's binary signal, generating in one shot; **3)** the diffusion denoiser — a U-Net trained by the noise-prediction MSE (§47.13's $L_{\text{simple}}$), generating by iterating the reverse process from pure noise; **4)** the latent-diffusion pipeline — same denoiser operating on VAE-compressed latents (§47.19), optionally steered by classifier-free guidance and cross-attention (§47.18). All four rest on the same Chapters 41–46 machinery: MLPs, backprop, Adam, conv stacks, skip connections, attention.
