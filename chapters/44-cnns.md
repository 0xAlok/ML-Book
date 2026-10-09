# Chapter 44: CNNs — convolution, pooling, transfer learning

Chapter 41 built the neuron and the MLP; Chapter 42 derived the gradients; Chapter 43 picked the optimizers. This chapter changes what a *layer* computes. Nothing else changes: the optimizer is layer-agnostic (Chapter 43's point), and backprop is the same machinery with new local gradients (§42.4). The spine is the GenAI Weeks 3–4 deck ("Introduction to CNN," Balaji Srinivasan and Ganapathy Krishnamurthi, 91 slides — vision modelling, convolution vs cross-correlation, stride/padding illustrations, pooling, batch normalization walkthrough, famous architectures, transfer learning) plus its three companion notebooks: *CNN_Fundamentals* (why MLPs fail on images, the convolution definition, pooling, a complete MNIST/Fashion-MNIST CNN in PyTorch), *Transfer_Learning* (feature extraction vs fine-tuning on FoodVision Mini with EfficientNet-B0), and *Image_Segmentation_UNET* (U-Net, used briefly in §44.9). Every numpy number in this chapter was re-run here and is marked **[verified-NumPy]**; deck, notebook, and torch numbers are transcriptions, marked **[recorded]**.

**Notation.** $I$ = input image (or feature map), $K$ = convolution kernel (filter), $(I * K)$ = output feature map. Image/feature-map shape is written $H \times W \times C$ (height, width, channels), the notebook's convention — §36.4's axis semantics apply to every shape question below.

## 44.1 Why dense layers fail on images

The *CNN_Fundamentals* notebook's case against feeding images to MLPs has four counts:

i) **Parameter explosion.** "A single $224 \times 224 \times 3$ color image has 150,528 pixels. If the first hidden layer has just 100 neurons, we'd need 15,052,800 parameters (weights) just for the first layer!" [recorded]. Set it next to §41.14(ii)'s $784 \to 50 \to 10$ MNIST net, where "just 1 hidden layer" already had "nearly 40,000 parameters" — §41.14(ii)'s overfitting warning was about a model almost four hundred times smaller than this dense vision layer.

ii) **Spatial information loss.** "MLPs flatten images into 1D vectors, destroying spatial relationships. The fact that adjacent pixels are related in an image is completely ignored" [recorded]. The deck's version: the naive approach is "flatten the image... feed into MLP" — a $28 \times 28$ grid becomes a $784 \times 1$ vector and the network never learns that pixel $(i,j)$ sits next to pixel $(i, j{+}1)$.

iii) **Translation variance.** "MLPs don't handle translations well — an object shifted slightly in an image appears as a completely different input. A dog in the left corner vs. center would require learning separate representations" [recorded]. Every position needs its own copy of every pattern.

iv) **Data inefficiency.** With no built-in inductive bias for images, the MLP "require[s] enormous datasets" and is "prone to overfitting with limited training data" [recorded].

The CNN fixes all four at once: a small *learnable filter*, slid across the whole image, with the *same weights reused at every position*.

**Basically, ...** "An MLP sees a photo as a flat list of 150,528 numbers — no neighbours, no 'left of'. One hidden layer of 100 neurons already costs 15 million weights, and a dog in the corner must be learned separately from a dog in the center. CNNs instead learn small pattern-detectors and slide them everywhere, so one edge detector finds edges at every position."

## 44.2 Convolution as local pattern matching

**Def (the notebook's).** "The convolution operation involves sliding a small filter (or kernel) across an input image and computing element-wise multiplications followed by summation." For a 2D input $I$ and kernel $K$:
$$\boxed{(I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m, n)} \quad \text{[recorded]}.$$
Each output pixel is the dot product of the kernel with one small patch of the input; move the window one step, repeat. The output is called a **feature map** (or activation map): at each position it says *how strongly the kernel's pattern matched there*.

**A terminology note the deck insists on.** Deep learning's "convolution" is, mathematically, **cross-correlation**: a true convolution flips the kernel first ($I(i-m, j-n)$); DL implementations never flip ($I(i+m, j+n)$). The deck's slide is blunt: "What we use is technically called cross-correlation," and then asks "Does it matter?" The honest answer: **no**, because the kernel weights are *learned*. A flipped copy of a learned kernel is just another learnable kernel — the network finds the same patterns either way. From here on this chapter says "convolution" in the deep-learning sense and means cross-correlation.

Three ideas travel with the operation:

i) **Local receptive field.** "Each neuron in a convolutional layer connects only to a small region of the input" [recorded]. A neuron in a $3 \times 3$ layer sees 9 pixels, not 150,528 — the locality the MLP threw away is back.

ii) **Learnable filters.** "The values in each filter are learnable parameters updated during training" [recorded]. The deck's learning rule is the same gradient-descent step as ever: $W \leftarrow W - \eta\,\partial\mathcal{L}/\partial W$, "repeated for every image and for multiple iterations till the loss converges" [recorded]. The filters "evolve to detect features (edges, colors, textures, patterns) that are most useful for the specific task" [recorded] — nobody hand-designs the edge detector; §42.4's backprop machinery computes $\partial\mathcal{L}/\partial W$ for a conv layer exactly as it did for a dense one.

iii) **Weight sharing.** "The same set of weights is used regardless of where in the image a feature appears" [recorded] — this is the engine of everything else: parameters (§44.7), translation behaviour (§44.5), and data efficiency all follow from one filter being applied everywhere.

**Basically, ...** "Put a small stencil over the top-left of the image, multiply its numbers with the pixels underneath, add up — that's one output pixel. Slide the stencil right, repeat; then down, repeat. The full output says, at every position, 'how much does my stencil match here?' The stencil's numbers are learned like any weights. (Pedant corner: this is cross-correlation, not a true convolution — but since the stencil is learned, the missing flip changes nothing.)"

## 44.3 eg: a 2D convolution, worked by hand [verified-NumPy]

Input $I$ ($4 \times 4$), kernel $K$ ($2 \times 2$), stride $1$, no padding:
$$I = \begin{bmatrix} 1 & 2 & 3 & 4 \\\\ 5 & 6 & 7 & 8 \\\\ 9 & 10 & 11 & 12 \\\\ 13 & 14 & 15 & 16 \end{bmatrix}, \qquad K = \begin{bmatrix} 1 & 0 \\\\ -1 & 1 \end{bmatrix}.$$

Position $(0,0)$: the top-left $2 \times 2$ patch is $\begin{bmatrix} 1 & 2 \\\\ 5 & 6 \end{bmatrix}$, element-wise with $K$:
$$y_{0,0} = 1\cdot 1 + 2\cdot 0 + 5\cdot(-1) + 6\cdot 1 = 1 - 5 + 6 = 2.$$
Position $(0,1)$: patch $\begin{bmatrix} 2 & 3 \\\\ 6 & 7 \end{bmatrix}$, $y_{0,1} = 2 - 6 + 7 = 3$. Position $(0,2)$: $y_{0,2} = 3 - 7 + 8 = 4$. Row 1: $y_{1,0} = 5 - 9 + 10 = 6$, $y_{1,1} = 6 - 10 + 11 = 7$, $y_{1,2} = 7 - 11 + 12 = 8$. Row 2: $y_{2,0} = 9 - 13 + 14 = 10$, $y_{2,1} = 10 - 14 + 15 = 11$, $y_{2,2} = 11 - 15 + 16 = 12$. Re-run in numpy:
```
[[ 2.  3.  4.]
 [ 6.  7.  8.]
 [10. 11. 12.]]
```
The output is $3 \times 3$ — smaller than the $4 \times 4$ input, because the $2 \times 2$ window only fits in 3 positions per axis. The next section names that rule.

<!-- Diagram reference (CC-BY-SA): d2l.ai §7.2.1 shows the same operation with a 3x3 input and 2x2 kernel, with the sliding window shaded — the canonical figure for this eg. Source URL: https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html -->

## 44.4 Channels, stride, padding: the shape bookkeeping

Every shape question about a conv layer is §36.4's axis semantics applied to $H \times W \times C$.

**Channels.** A filter is as deep as its input. On a $28 \times 28 \times 1$ grayscale image a $3 \times 3$ filter has $3 \times 3 \times 1 = 9$ weights; on a $224 \times 224 \times 3$ color image it has $3 \times 3 \times 3 = 27$. The kernel multiplies all three channel-planes at each position and sums to one number — so each *filter* produces one 2D feature map, and $F$ filters produce an $H' \times W' \times F$ output. The notebook's numbers: "for a single $3 \times 3$ convolutional filter: only 9 weights (plus 1 bias)" [recorded] — that is per input channel; the $+1$ bias is per filter, added to every position of its feature map.

**Stride.** How many pixels the window moves per step. The deck's illustration: $5 \times 5$ input, $3 \times 3$ kernel, stride $2$ → $2 \times 2$ output [recorded].

**Padding.** Zeros added around the border so the kernel can cover the edge pixels. The deck's illustration: $4 \times 4$ input, $3 \times 3$ kernel, no padding ("valid") → $2 \times 2$ output — "(Output shrinks)" [recorded].

**Theorem (output size, per axis).** With input width $W$, kernel $K$, padding $P$ per side, stride $S$:
$$\boxed{W_{\text{out}} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1} \quad \text{[verified-NumPy]}.$$
Why: the first window starts at pixel $0$, the last at pixel $W - K + 2P$ (counting padding), in steps of $S$ — $(W-K+2P)/S + 1$ positions, floored. **"Same" padding** picks $P$ so the output keeps the input size: for stride $1$ and odd $K$, $P = (K-1)/2$ (e.g. $3 \times 3$ with $P = 1$ on $28 \times 28$ → $28 \times 28$). **"Valid"** is $P = 0$: the size shrinks by $K-1$.

**eg 4 (the formula against the course's own diagrams).**
- Deck's padding slide: $W = 4$, $K = 3$, $P = 0$, $S = 1$ → $\lfloor(4-3)/1\rfloor + 1 = 2$. Output $2 \times 2$ — the slide's number.
- Deck's stride slide: $W = 5$, $K = 3$, $P = 0$, $S = 2$ → $\lfloor(5-3)/2\rfloor + 1 = 2$. Output $2 \times 2$ — the slide's number.
- Notebook's MNIST net: Conv1 ($3 \times 3$, stride $1$, valid) on $28 \times 28$ → $26 \times 26$; 32 filters → $26 \times 26 \times 32$. Same-padding version would stay $28 \times 28$.

**Note.** Filter sizes are "typically odd numbers like $3 \times 3, 5 \times 5, 7 \times 7$" because "an odd size provides a center pixel" [recorded] — symmetric padding around a center is what makes "same" padding exact.

**Basically, ...** "The kernel is as deep as the input ($3 \times 3 \times 3$ on color), and one filter = one output sheet, so $F$ filters give $F$ sheets. The output shrinks whenever the kernel is bigger than a pixel and has no padding: $(W-K+2P)/S + 1$ positions per side. Want to keep the size? Pad one ring of zeros ($P{=}1$ for a $3 \times 3$). Want to shrink fast? Stride $2$."

## 44.5 Pooling: downsampling with a purpose

**Def (the deck's).** "A pooling layer performs non-linear downsampling to reduce the spatial dimensions (width and height) of the feature maps." Its two key purposes: "Reduce Computational Cost" and "Increase Receptive Field — allows subsequent convolutional layers to see a larger area of the original input" [recorded].

The notebook's three types:

i) **Max pooling.** "Takes the maximum value from a local region. Most commonly used in practice. Helps extract the most prominent features" [recorded]. Has no learned parameters.

ii) **Average pooling.** "Takes the average value from a local region. Preserves more background information" [recorded].

iii) **Global pooling.** "Reduces each feature map to a single value (max or average)" [recorded] — a $H \times W \times F$ stack becomes one $F$-vector, often right before the dense layers.

Typical settings: $2 \times 2$ window, stride $2$ — "halves both dimensions" [recorded]; padding is "rarely used in pooling layers" [recorded].

**eg 5 (the notebook's pooling diagram, [verified-NumPy]).** $2 \times 2$ max pooling, stride $2$, on
$$\begin{bmatrix} 1 & 3 & 2 & 5 \\\\ 9 & 7 & 8 & 6 \\\\ 4 & 2 & 7 & 3 \\\\ 1 & 0 & 5 & 2 \end{bmatrix} \longrightarrow \begin{bmatrix} 9 & 8 \\\\ 4 & 7 \end{bmatrix}:$$
top-left window $\{1,3,9,7\}$ → $9$; top-right $\{2,5,8,6\}$ → $8$; bottom-left $\{4,2,1,0\}$ → $4$; bottom-right $\{7,3,5,2\}$ → $7$. Pooling asks "what was the strongest response in this patch?" and throws away *exactly where* — that is the mechanism behind "invariance to small translations" [recorded].

**What pooling buys, in three lines.**

i) **Translation robustness.** Convolution is *equivariant* — "if a feature moves in the input, its representation moves correspondingly in the feature maps" [recorded]. Pooling then summarizes each patch, which *enhances translational invariance* [recorded]: the exact position matters less after a max.

ii) **Downsampling.** Halving $H$ and $W$ quarters the computation of every later layer, and (§44.7's point) keeps the parameter count in check.

iii) **Bigger receptive fields downstream.** After pooling, the next conv layer's $3 \times 3$ window covers twice as much of the original image — deep layers see larger patterns built from the small ones.

**Note (pooling vs PCA, §24's contrast).** Both shrink representations, but by opposite means. PCA (§24) *learns* a global linear projection from the data's covariance — optimal reconstruction, but it flattens everything (§24.5) and knows nothing about 2D layout. Pooling is a *fixed*, *local*, *nonlinear* summary that preserves the spatial grid. Pooling throws away position-within-patch; PCA throws away low-variance directions. Different losses, different keepers.

**Basically, ...** "Pooling is the network deliberately squinting: take each $2 \times 2$ patch, keep only its loudest pixel (max) or its average. It shrinks the map (cheap!), makes exact positions matter less (robust!), and lets deeper layers see bigger picture chunks through the same small window. Compared to Chapter 24's PCA: PCA learns one global linear squish from the data; pooling is a fixed local squish that keeps the 2D layout."

## 44.6 The CNN stack: feature hierarchy, then a dense head

The notebook's "standard layer sequence," with its own MNIST example traced end to end:

i) **Input:** $H \times W \times C$ raw pixels ($28 \times 28 \times 1$ grayscale, $224 \times 224 \times 3$ color).

ii) **Conv + activation:** filters extract features, "typically ReLU" [recorded] for the non-linearity (§41.7's activation logic applies unchanged).

iii) **Pool:** downsample.

iv) **Repeat** — conv→pool blocks stacked. The design pattern: "increase filters, decrease spatial dimensions. As we go deeper, number of filters increases while spatial dimensions decrease" [recorded].

v) **Flatten → dense → output:** "converts 2D feature maps to 1D feature vector" [recorded], then "traditional neural network layers" for the final prediction — §41.4's two equations ($Z^l = A^{l-1}W^l + b^l$, $A^l = g(Z^l)$) are back, with dropout ($p = 0.5$ in the example, §41.14(ii)'s dropout) between dense layers.

**eg 6 (the notebook's MNIST architecture, shapes traced — [verified-NumPy]).**
```
Input: 28x28x1
Conv1: 3x3, 32 filters, stride 1, valid -> 26x26x32
Pool1: 2x2, stride 2                    -> 13x13x32
Conv2: 3x3, 64 filters, stride 1, valid -> 11x11x64
Pool2: 2x2, stride 2                    -> 5x5x64
Conv3: 3x3, 128 filters, stride 1, valid -> 3x3x128
Flatten -> 1152
FC1: 128, ReLU; Dropout 0.5; FC2: 10, softmax
```
Every arrow is §44.4's formula: $(28-3)/1+1 = 26$, $(26-2)/2+1 = 13$, $(13-3)/1+1 = 11$, $(11-2)/2+1 = 5$, $(5-3)/1+1 = 3$, $3 \times 3 \times 128 = 1152$.

What each level *learns* is the hierarchy the notebook states plainly: "early layers: detect simple features (edges, corners); middle layers: textures, patterns; later layers: complex objects, parts" [recorded]. The notebook backs it with feature-map and filter visualizations — free from the architecture: first-layer filters are readable directly because they act on raw pixels.

**Basically, ...** "Stack conv+ReLU+pool blocks: each block shrinks the map and doubles down on features, early blocks finding edges, late blocks finding parts of objects. Then flatten and hand the $F$-vector to an ordinary dense network (§41.4's equations, dropout included) for the final call. The traced MNIST stack: $28{\times}28 \to 26{\times}26{\times}32 \to 13{\times}13{\times}32 \to 11{\times}11{\times}64 \to 5{\times}5{\times}64 \to 3{\times}3{\times}128 \to 1152 \to 128 \to 10$."

## 44.7 The parameter-sharing payoff

The notebook does the count twice, and both are worth keeping:

i) **64 filters on $28 \times 28$.** "For 64 filters: CNN = 640 parameters vs MLP = 50,176 parameters" [recorded]: CNN $= 64 \times (3 \times 3 + 1) = 640$ (the $+1$ is the bias per filter); MLP $= 784 \times 64 = 50{,}176$ (one weight per input–neuron pair). Roughly an $80\times$ saving — and the CNN's count does not grow with image size.

ii) **The big one, re-run here.** A $224 \times 224 \times 3$ image into 100 hidden neurons costs the dense layer $224 \cdot 224 \cdot 3 \cdot 100 + 100 = 15{,}052{,}900$ parameters [verified-NumPy]. The same first layer as a 64-filter $3 \times 3$ conv on 3 channels: $64 \cdot (3 \cdot 3 \cdot 3 + 1) = 1{,}792$ parameters [verified-NumPy]. A $8{,}400\times$ reduction — that is the whole of §44.1(i) answered by one subtraction.

**Why sharing is legal.** A filter that detects a vertical edge in the top-left detects the same vertical edge in the bottom-right — "the same feature detector applied across entire image" [recorded]. §41.14(ii) counted parameters to warn about overfitting; convolution is the structural answer to that warning: fewer parameters, and they mean the same thing everywhere.

**What does not change.** The deck's transfer-learning slides train these layers with the machinery of Chapter 43 — the optimizer is layer-agnostic (Chapter 43's closing point). And §43.9's flagged gap is now partly closed: the Weeks 3–4 deck *does* walk through batch normalization — normalize each activation, $\hat x_i = (x_i - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$, then scale-and-shift $y_i = \gamma \hat x_i + \beta$ with two learned parameters [recorded; deck slide 54]. The full BN treatment (why $\gamma, \beta$ restore representational power, train/test behaviour) stays a candidate for its own note.

**Basically, ...** "Same patterns, fewer numbers. A $3 \times 3$ filter has 9 weights no matter how big the image is, and one edge detector works at every position — so a conv layer that replaces a 15-million-parameter dense layer needs under two thousand. That is §41.14(ii)'s overfitting warning being answered by the architecture itself. (Bonus: the CNN deck's slide 54 fills in the batch-norm formula §43.9 said was missing.)"

## 44.8 Transfer learning: don't reinvent the wheel

**Def (the deck's).** "Transfer Learning is the process of taking a model pre-trained on a large dataset and fine-tuning it for a new, specific task." The motivation: "Training a large CNN from scratch requires massive datasets (like ImageNet) and huge computational resources" [recorded]. The notebook's framing: pre-trained weights are "superior initialization points for optimization" (§43.10's initialization problem, solved by borrowing someone else's answer).

**When it works.** Early conv layers learn generic features — edges, colors, textures — that are useful for almost any vision task; only the late layers are task-specific. So the bottom of the network transfers across domains, and you only teach the top what your classes look like. The standard backbone: "a network (like ResNet50 or MobileNet) that has been pre-trained on the massive ImageNet dataset (1.4 million images, 1000 classes)" [recorded].

**The three strategies (notebook's table, [recorded]).**

| Strategy | Method | Use case |
|---|---|---|
| **Feature extraction** | Freeze pre-trained layers, train only classifier | Small target datasets |
| **Fine-tuning** | Update all parameters with reduced learning rates | Moderate target datasets |
| **Progressive fine-tuning** | Gradual unfreezing of network layers | Large target datasets |

"To *freeze* layers means to keep them how they are during training" — in PyTorch, `requires_grad=False`, "so PyTorch doesn't track gradient updates and these parameters won't be changed by our optimizer" [recorded].

**eg 7 (the notebook's recipe, [recorded] torch).** The *Transfer_Learning* notebook classifies pizza/steak/sushi (FoodVision Mini) with an ImageNet-pretrained **EfficientNet-B0**:

1. `weights = EfficientNet_B0_Weights.DEFAULT`, `model = efficientnet_b0(weights=weights)`.
2. The model has three parts: `features` ("a collection of convolutional layers... to learn a base representation of vision data"), `avgpool` ("takes the average of the output of the `features` layer(s) and turns it into a feature vector"), `classifier` (turns the feature vector into class scores — `out_features=1000` for ImageNet).
3. Freeze `features` (`requires_grad=False`) — the feature-extraction strategy.
4. Replace `classifier` with a new head for 3 classes (the old head is exactly §44.6(v)'s dense head, rebuilt for the new task).
5. Feed it ImageNet-preprocessed inputs: $224 \times 224$ resize, normalize with ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$ [recorded] — "the custom data going into the model should be prepared in the same way as the original training data."
6. Train the head with Adam ($\eta = 0.001$, Chapter 43's defaults) on the small dataset.

**The notebook's one-line rule.** The "custom data" warning is the whole discipline: a pre-trained backbone only helps if its input distribution matches training — wrong normalization silently degrades the transferred features.

**Basically, ...** "Don't train from scratch on a mountain of images when someone already did it on ImageNet. Take their conv stack (it already knows edges and textures), freeze it (set `requires_grad=False`), bolt on your own small classifier for your classes, and train just that head — on your tiny dataset, with Adam at the usual settings. When you have more data, unfreeze and fine-tune the whole thing with a small learning rate. Rule: preprocess your images exactly like ImageNet's, or the borrowed features go stale."

## 44.9 A second head for CNNs: segmentation (the U-Net note)

The *Image_Segmentation_UNET* notebook applies this chapter's layers to a different output shape: **segmentation** assigns a class to *every pixel* (the notebook's case study is leaf-disease segmentation), not one label to the whole image. Its **U-Net** is an encoder (the §44.6 conv+pool stack, shrinking) followed by a decoder (upsampling back to full resolution), with **skip connections** joining matching resolutions — early fine-grained maps guide the upsampling. Evaluate with the Dice coefficient, not accuracy [recorded]. The architecture detail is kept for a later pass; the takeaway now: conv and pooling are Lego for more than classification — reshape the head, change the loss, and the same stack segments.

## 44.10 Where this goes next

i) **RNNs (Chapter 45).** The same idea, one new dimension: weight sharing across *time* instead of *space*. Chapter 44's filter is reused at every pixel; Chapter 45's cell is reused at every timestep — and the price of sharing (vanishing gradients, §42.8's other half) arrives with it.

ii) **Attention (Chapter 46).** The hierarchy of §44.6 — local patterns building up to global ones — gets a shortcut: attention lets any position talk to any other directly, instead of waiting for pooling to bring them together.

iii) **Generative models (Chapter 47).** GANs and diffusion models are built on conv stacks like this chapter's — the discriminator is a CNN classifier, and the generator is the §44.9 decoder run in reverse.

iv) **The debugging playbook rides along.** §40.5(i): "shapes first" — for a CNN that means running every layer through §44.4's formula before training, exactly as eg 6 did. Wrong flatten size is the transpose bug's (§41.4) convolutional cousin.

## Problem set

1. **Parameter explosion, quantified.** A $224 \times 224 \times 3$ image feeds a dense layer with 100 neurons. (i) Count the weights and biases. (ii) Replace it with a conv layer of 64 $3 \times 3$ filters: count weights and biases. (iii) Compare with §41.14(ii)'s $39{,}760$ and say in one line what this implies about overfitting risk.
2. **The convolution, by hand.** $I = \begin{bmatrix} 1 & 0 & 2 \\\\ 3 & 1 & 1 \\\\ 0 & 2 & 1 \end{bmatrix}$, $K = \begin{bmatrix} 1 & 1 \\\\ 0 & -1 \end{bmatrix}$, stride $1$, no padding. (i) Compute the $2 \times 2$ output entry by entry. (ii) Which input position contributes to all four output entries, and why does that matter for "local pattern matching"? (iii) Add one learned bias $b$ (same for all four output positions): how many learned parameters does this layer have?
3. **Name the operation.** (i) Write the true-convolution formula (flipped kernel) and the deep-learning formula side by side. (ii) In two lines, explain why the missing flip does not hurt a network whose kernels are learned. (iii) The deck asks "Does it matter?" — give your one-line verdict.
4. **Stride and padding.** (i) $W = 7$, $K = 3$, $P = 0$, $S = 2$: compute $W_{\text{out}}$. (ii) Same $W, K, S$ with $P = 1$: compute $W_{\text{out}}$. (iii) A $32 \times 32$ image passes through conv ($3 \times 3$, stride 1, "same") then $2 \times 2$ max-pool (stride 2): give the feature-map size after each step.
5. **Trace the MNIST stack.** (i) Reproduce eg 6's $28 \to 26 \to 13 \to 11 \to 5 \to 3$ shape chain from the output-size formula. (ii) Why does MaxPool2 give $5$ and not $5.5$? (iii) If Conv2 used $128$ filters instead of $64$, how would the flatten size change, and how many weights would Conv2 then have (per $3 \times 3 \times C_{\text{in}} + 1$)?
6. **Pooling vs PCA.** (i) Pooling a $4 \times 4$ map with $2 \times 2$ max-pool gives $2 \times 2$: how many numbers are discarded, and what information exactly is lost? (ii) In two lines, contrast with §24's PCA: what does PCA keep that pooling throws away, and vice versa? (iii) Name one task where you'd pick PCA over pooling, and one where you'd pick pooling over PCA.
7. **Equivariance, then invariance.** (i) In one line each, define translation *equivariance* and translation *invariance*. (ii) Say which one a conv layer has and which one pooling adds, in the notebook's terms. (iii) A cat photo is shifted right by 10 pixels: trace what happens to the conv feature map and then to the pooled map, in words.
8. **Freeze vs fine-tune.** (i) In the notebook's three-row table: which strategy for 200 training images, and why? (ii) What does `requires_grad=False` on `features` do to the §42.4 backward pass, mechanically? (iii) Why does the notebook insist on ImageNet normalization of the *new* dataset?
9. **The transfer-learning head.** EfficientNet-B0's `classifier` outputs 1000 scores. (i) For FoodVision Mini (3 classes), what shape must the replacement head have? (ii) Name the model-construction call and the freeze mechanism used in eg 7. (iii) If you unfreeze everything with the same $\eta = 0.001$, name the risk (Chapter 43's vocabulary) and the fix (fine-tuning row of the table).
10. **Design a small CNN.** Input $64 \times 64 \times 3$, classes $= 10$. (i) Propose conv/pool blocks ending with a flatten size $\le 2000$, showing the shape chain. (ii) Count the conv parameters (assume $3 \times 3$ filters, $+1$ bias each). (iii) Name the §41.14(ii) guardrail you would add before the output layer, and why.

---

*Sources: GenAI Weeks 3–4 "Introduction to CNN" deck (Balaji Srinivasan, Ganapathy Krishnamurthi), 91 slides — naive-MLP-flatten slide, translation-invariance principle, convolution-vs-cross-correlation terminology slide, edge-detection slide, kernel-learning slide ($W \leftarrow W - \eta\,\partial\mathcal{L}/\partial W$), kernel-size note (odd sizes), stride illustration ($5 \times 5$, $3 \times 3$, $S = 2$ → $2 \times 2$), padding illustration ("valid" $4 \times 4$, $3 \times 3$ → $2 \times 2$, "output shrinks"), pooling definition + purposes, batch-normalization walkthrough (slide 54), data augmentation, ImageNet challenge, ResNet degradation, transfer-learning slides ("don't reinvent the wheel", ResNet50/MobileNet backbone, ImageNet 1.4M/1000 classes); CNN_Fundamentals.ipynb — MLP limitations (§44.1's four counts, 15,052,800 parameters), convolution definition + hyperparameters, weight sharing (64 filters: 640 vs 50,176), equivariance vs invariance, pooling types + worked $4 \times 4$→$2 \times 2$ example, MNIST CNN architecture and shape chain, Fashion-MNIST training, feature-map/filter visualizations; Transfer_Learning.ipynb — transfer objective framing, three-strategy table, freeze mechanics (`requires_grad=False`), EfficientNet-B0 anatomy (features/avgpool/classifier), ImageNet preprocessing (means, stds), FoodVision Mini recipe (all torch code [recorded]); Image_Segmentation_UNET.ipynb — U-Net encoder/decoder + skip connections, leaf-disease case study, Dice coefficient (brief §44.9); Week3-CNN.pdf and Week4-CNN notes — mirror of the deck; book chapters 24 (§24.5), 36 (§36.3–§36.4), 40 (§40.5(i)), 41 (§§41.4, 41.7, 41.14(ii)), 42 (§42.4), 43 (layer-agnostic optimizers, §43.9–§43.10, §43.12 defaults).*
