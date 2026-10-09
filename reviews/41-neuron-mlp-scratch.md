# Review log — Chapter 41: The artificial neuron and MLPs from scratch

Branch: `draft/neuron-mlp-scratch`. Reviewed against the committed sources; self-review pass done 2026-10-09.

## Sources actually used

1. `GenAI/.../Slides/Week - 1 & 2/Introduction to ANN - Lecture Slides.pdf` (93 slides, text-extractable, authors Balaji Srinivasan, Ganapathy Krishnamurthi) — the spine: neuron two-step, M-P→perceptron→neuron lineage, vectorized two-core equations, the 2→3 ReLU worked example, MLP definition, stacking-linear collapse, activations + selection guide, forward pass, MSE/cross-entropy loss slides.
2. `GenAI/.../Slides/Week - 1 & 2/XOR Problem.ipynb` (28 cells) — XOR proof, the 2-2-1 class skeleton (sigmoid/BCE, `dL/dz2 = ŷ−y` trick), the hand-worked forward pass (§41.9's eg 4), the 1969 Minsky–Papert history line.
3. `GenAI/.../Slides/Week - 1 & 2/Pytorch Fundamentals.ipynb` (149 cells) — tensor vocabulary transcribed [recorded]: creation, `*` vs `@` ("the difference between element-wise multiplication and matrix multiplication is the addition of values" — verified verbatim), `reshape`/`view`/`squeeze`/`unsqueeze`/`permute`, indexing, `torch.from_numpy`.
4. `GenAI/.../Slides/Week - 1 & 2/Pytorch Workflow.ipynb` (84 cells) — the five-step loop transcribed verbatim [recorded]; train/eval separation ("prevents information leakage"); `state_dict()` save/load.
5. `GenAI/.../Slides/Week - 1 & 2/Fashion MNIST Classification.ipynb` (24 cells) — `FashionMLP` 784→128→64→10 transcribed [recorded]; `nn.CrossEntropyLoss()` + Adam (`lr=0.001`, `weight_decay=1e-4`), 10 epochs [recorded].
6. `MLT/.../Slides - Ashish Tendulkar/Week 12/MLT Week 12 Slides ANN.pdf` (82 pages, text-extractable) — neuron equations, feedforward, $S_l$ notation, $W^l$ size $S_{l-1}\times S_l$, batch form $Z^l = A^{l-1}W^l + b^l$, output-layer activation pairings, the $\tfrac12$-scaled squared-error with the verbatim numpy line, categorical CE matrix form with its numpy line, GD init $\theta\sim\mathcal{N}(0,1)$, zero-init symmetry argument, MNIST 784-50-10 parameter count (39,760), dropout vs L1/L2 distinction.
7. `MLT/.../PPT/Week 12/slide2.pdf` p.2 (rendered via pdftoppm) — the hand-annotated net: sigmoid → $p(y\mid x)$ → "Cross-Entropy Loss" (the §35.13 forward pointer landing), "converges to local minima", "typically works very well in practice, especially for unstructured data."
8. `MLP/.../Practice and Graded Assignment Solutions/Week 12/MLP_W12_PGQ.ipynb` — `MLPClassifier` diabetes run (train 0.7915309446254072, test 0.7662337662337663) and `MLPRegressor` run (train 0.999992602958679, test 0.9999920781245708) transcribed [recorded]; sklearn not installed here.
9. `GenAI/.../Notes/Week-1,2-ANN.pdf` — byte-distinct PDF of the same 93-slide deck; no unique material, not used.

## Re-run / recorded ledger

- [verified-NumPy] Deck's 2→3 ReLU worked example: `z = [0.9, 0.1, -0.35]`, `a = [0.9, 0.1, 0.]` — matches slide.
- [verified-NumPy] Notebook's hand-worked forward pass (x=(1,0)): `z1=[1.,0.], h1=[0.7311,0.5], z3=-0.0379, ŷ=0.4905, loss=0.7123` — matches notebook's 0.491/0.712.
- [verified-NumPy] Book's own 2-2-1 training run (seed 42, lr 5.0, 1000 epochs, full-batch): 50%→75%→100% by epoch 200; final preds `[0.0021, 0.9981, 0.9971, 0.0018]`, loss 0.002178. Script kept at /tmp/xor_experiment.py (ephemeral; trace printed in chapter).
- [verified-NumPy] All 10 problem-set numbers re-computed (P1 z=-0.2, σ=0.4502; P5 z=[1.5,-2.5,-2.5], a=[1.5,0,0]; P6 losses 1.6094/0.2231/0.3567/0.75; P7 chain (−2.0387)(0.24991)=−0.5095 = g−y; P2 93 params, MNIST 39,760).
- [recorded] PyTorch snippets and the two MLPClassifier/MLPRegressor scores — transcribed, never executed (torch/sklearn absent).

## Thin / contradictory points

- The deck's per-example notation ($W^{[l]}a^{[l-1]}$, $W$ is $S_l\times S_{l-1}$) and the MLT slides' batch notation ($A^{l-1}W^l$, $W$ is $S_{l-1}\times S_l$) are transposes of each other; even the index-order convention on $W_{ij}$ differs between the two decks. Handled in §41.4's Note + problem 3 rather than reconciled silently.
- The MLPRegressor experiment's dataset (target `AH`, test n=1872) is not identified by name in the notebook; the chapter describes it minimally ("an sklearn regression demo") rather than guessing.
- The backprop math (`backward`) is presented as the chapter's machine but not derived — Chapter 42's job, stated explicitly at §41.10, §41.13, §41.15(i).

## Fixes made in review

- Deck's guide typo "Parameteric ReLU" silently corrected to "Parametric ReLU" in §41.7 (this log).
- §41.10(iii) clarified that §31.10's weights-inflate warning is what the shrinking-to-never-zero loss demonstrates.
- ReLU range written as $[0,\infty)$ in the table; ReLU called "smooth enough" (corner at 0) in §41.3 rather than smooth.
