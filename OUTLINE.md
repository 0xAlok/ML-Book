# Book Outline — APPROVED by Alok as-is, 2026-10-09 (47 chapters; GenAI Part VI in scope; course sequencing kept)

Derived from the source dump inventory (`sources/`) and the 6-phase spine in
GOAL.md. Each chapter = one topic, with worked `eg` blocks, a problem set in
`chapters/`, and full worked solutions in `solutions/`. Standard explanation
first, then a "Basically, ..." ELI10/15 simplification for complex topics.
Diagrams reused from standard resources with attribution comments.

## Part I — Mathematical Foundations (Maths 1, Maths 2, MLF W2–W9)

1. Sets, functions, and mathematical preliminaries — Maths1 VOL1; MLF W2
2. Vectors and matrices — Maths1 slides W1–W2; Maths2 LA PDF
3. Linear systems, determinants, inverses — Maths1 slides W1–W2
4. Vector spaces and the four fundamental subspaces — MLF W3
5. Orthogonality, projections, least squares — MLF W3
6. Eigenvalues, eigenvectors, diagonalization — MLF W4
7. SVD and positive definite matrices — MLF W5–W6
8. Single-variable calculus: continuity, differentiability, derivatives — Maths1 VOL2; MLF W2
9. Multivariable calculus: partial derivatives, gradients, Taylor series — MLF W2, W7
10. Unconstrained optimization and gradient descent — MLF W7
11. Constrained optimization: Lagrange multipliers, projected GD — MLF W8
12. Convex sets and convex functions — MLF W8–W9
13. Duality and KKT conditions — MLF W9–W10

## Part II — Probability and Statistics (Stats 2, MLF W11–W12)

14. Probability basics: experiments, sample spaces, Bayes — Stats2 W0
15. Discrete random variables and key distributions — Stats2 W0 (+ distribution-explorer links)
16. Continuous random variables — Stats2 W1; MLF W11
17. Joint distributions: two random variables — Stats2 VOL1; MLF W11
18. Joint continuous distributions — Stats2 VOL2
19. Random vectors and the multivariate normal — MLF W12
20. Estimation: MLE and Bayesian/MAP — MLF W12; MLT W4
21. Inequalities and the central limit theorem — MLF W12

## Part III — Machine Learning Foundations (MLF W1, W4, W6)

22. What is ML: data, models, tasks; supervised vs unsupervised — MLF W1
23. Linear and polynomial regression — MLF W4
24. PCA: variance maximization, eigendecomposition, kernel PCA — MLF W6; MLT W1–W2
25. GMMs and the EM algorithm — MLF W12; MLT W4

## Part IV — Machine Learning Techniques (MLT W1–W12)

26. K-means clustering — MLT W3
27. Kernel methods and kernel regression — MLT W2, W5
28. Regularization: ridge and lasso — MLT W3 (Ashish Tendulkar slides; actual location — the shared materials' Week 6 is Naive Bayes)
29. KNN and decision trees — MLT W7, W9
30. Naive Bayes: generative vs discriminative models — MLT W8
31. Perceptron and logistic regression — MLT W4 slides (perceptron; actual location — outline said W9); MITx 6.036 notes in course materials (logistic regression)
32. Hard-margin SVM — MLT W10
33. Soft-margin SVM and the kernel trick — MLT W11
34. Ensembles: bagging, boosting, AdaBoost — MLT W11
35. Loss functions for classification — MLT W12

## Part V — ML in Practice, code-first (MLP)

36. NumPy and Pandas for ML — MLP W1; MLT TA colab notes
37. scikit-learn: regression workflows — MLP
38. scikit-learn: classification workflows — MLP W5
39. Model evaluation, cross-validation, hyperparameter tuning — MLP
40. End-to-end model training and debugging — MLP

## Part VI — Bridge to Deep Learning (GenAI, MLT W12)

41. The artificial neuron and MLPs from scratch — GenAI W1–2; MLT W12
42. Backpropagation, worked end-to-end — GenAI W1–2
43. Optimization for neural nets: SGD variants, init, normalization — GenAI W5
44. CNNs: convolution, pooling, transfer learning — GenAI W3–4
45. RNNs: sequences and vanishing gradients — GenAI W7–8
46. Attention and the transformer — GenAI W9–10
47. Generative models: GANs and diffusion — GenAI W6, W11

## Notes / open decisions for approval

- ~47 chapters. At 8–15 pages each this is a ~500–700 page book: thorough but
  not a 2000-page monument. Trim/merge candidates: ch 2+3, ch 11+13, ch 32+33.
- Stats1.zip in the dump was a mislabeled duplicate of Maths1 (verified
  byte-identical, removed). Stats 2's Week 0 covers the Stats 1 basics, so no
  gap in practice — but flagging it.
- GenAI material was in the shared dump and is the spine of Part VI. Included
  on the assumption that "everything he shared" is in scope.
- MLP week-11 slides are missing from the dump; practice solutions cover W11.
  Not blocking.
- Cross-check anchors per chapter: mml-book (Parts I–II), d2l.ai math appendix
  + main chapters (Part VI), distribution-explorer (Part II), IITM community
  notes (Parts III–IV), mecmath.net (Part I calculus).
