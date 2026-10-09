# Foundations to Deep Learning

*From foundations to deep learning*

47 chapters taking you from mathematical foundations (linear algebra, calculus,
probability) through classical machine learning to neural networks and
transformers, everything needed before the core deep learning and AI courses.

**Read it:** [web book](https://0xAlok.github.io/ML-Book/) · [PDF](./docs/foundations-to-deep-learning.pdf)


## Contents

- **Part I — Mathematical Foundations** (chs 1–13): sets, vectors and matrices,
  linear systems, vector spaces, orthogonality, eigenvalues, SVD, single and
  multivariable calculus, optimization, convexity, duality.
- **Part II — Probability and Statistics** (chs 14–21): probability basics,
  discrete and continuous random variables, joint distributions, random vectors,
  MLE/MAP estimation, inequalities and the CLT.
- **Part III — ML Foundations** (chs 22–25): what ML is, regression, PCA,
  GMMs and EM.
- **Part IV — ML Techniques** (chs 26–35): k-means, kernels, regularization,
  trees, naive Bayes, perceptron and logistic regression, SVMs, ensembles,
  loss functions.
- **Part V — ML in Practice** (chs 36–40): NumPy/Pandas, sklearn workflows,
  evaluation and tuning, end-to-end debugging.
- **Part VI — Bridge to Deep Learning** (chs 41–47): MLPs from scratch,
  backpropagation, neural-net optimization, CNNs, RNNs, attention and
  transformers, GANs and diffusion.

## Repo layout

- `chapters/` — the book, one markdown file per chapter
- `solutions/` — full worked solutions, one file per chapter
- `reviews/` — per-chapter review logs from the QA passes
- `errata.md` — every correction: what was wrong → what it became
- `OUTLINE.md` — the original 47-chapter plan
- `STYLE.md` — the writing style guide the book follows
- `build.py` — one-command rebuild of the web book and PDF (needs Quarto)
- `docs/` — built static site, served by GitHub Pages

## License

Creative Commons Attribution 4.0 International (CC BY 4.0) —
https://creativecommons.org/licenses/by/4.0/ — you are free to share and
adapt with attribution.
