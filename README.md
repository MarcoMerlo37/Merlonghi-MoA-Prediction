# MoA Prediction — Multi-Model Ensemble

Author: Marco Merlonghi — Project Work in Machine Learning and Data Mining

## Overview

This project addresses the Kaggle **Mechanisms of Action (MoA) Prediction** challenge:
given gene-expression and cell-viability profiles of human cells treated with a
chemical compound, predict which of 206 possible biological mechanisms of action
(MoAs) the compound triggers. It is a **multi-label classification** problem (a
sample can be positive for zero, one, or several of the 206 targets simultaneously)
evaluated with the mean binary log loss over all samples and targets [1].

- Competition page: <https://www.kaggle.com/competitions/lish-moa>
- Data: <https://www.kaggle.com/competitions/lish-moa/data>

### Data in brief

- About 23.8k training rows, about 22k after removing control rows
  (`cp_type == ctl_vehicle`), which have no MoA by definition.
- 772 gene-expression features (`g-`), 100 cell-viability features (`c-`), plus the
  categorical `cp_time` and `cp_dose`.
- Positives are very rare: the mean predicted probability on treated test rows is
  about 0.0038.

## Objectives

- Build a robust cross-validated pipeline for a large-scale, highly imbalanced
  multi-label classification problem.
- Compare several modelling paradigms on the same folds: a residual tabular network
  (TabularResNet), an attention-based FT-Transformer [2], and a gradient-boosted tree
  model (CatBoost) [3], each fed a preprocessing pipeline suited to it.
- Reduce variance through multi-seed bagging and 5-fold multilabel-stratified cross
  validation [4, 5], and combine the base models with an ensemble whose weights are
  learned and re-validated through nested cross-validation.
- Produce a clean, reproducible, well-documented deliverable (see `software/`).

## Approach summary

| Stage | Choice |
|---|---|
| Cross-validation | 5-fold `MultilabelStratifiedKFold`, `random_state=42` |
| Model selection (neural nets) | Inner stratified split (10% of each training fold) for early stopping, LR scheduling and best-epoch choice; the outer fold is scored once |
| Preprocessing (ResNet) | `QuantileTransformer` (RankGauss) → PCA (50 components on gene features, 15 on cell-viability features), concatenated to the transformed features |
| Preprocessing (FT-Transformer, CatBoost) | `QuantileTransformer` (RankGauss), no PCA |
| Base models | TabularResNet, FT-Transformer (3-seed bagging each), CatBoost (`MultiLogloss`, fixed 250 iterations) |
| Loss | `BCEWithLogitsLoss` (optionally focal loss [8]), label smoothing 0.001 [9] |
| Optimisation | AdamW [6], `ReduceLROnPlateau`, mixed precision [7] |
| Regularisation | Early stopping (patience 8), dropout, batch norm, weight decay |
| Ensembling | Softmax-reparameterised weight search (Nelder–Mead [10]) on out-of-fold predictions, re-validated with nested CV |
| Post-processing | Control-vehicle test rows set to 0 |

Architectural and implementation details are documented inline in the notebook —
see [`software/moa_prediction.ipynb`](software/moa_prediction.ipynb). The
FT-Transformer is a simplified variant of the one in [2] (post-LayerNorm encoder,
GELU feed-forward, no ReGLU).

## Results

Out-of-fold (OOF) mean log loss over the 5 folds (lower is better). Network scores
use predictions averaged over 3 seeds.

| Model | OOF log loss |
|---|---|
| TabularResNet | **0.01632** |
| FT-Transformer | 0.01643 |
| CatBoost | 0.01700 |
| **Ensemble (nested, honest estimate)** | **0.01606 ± 0.00019** |

The naive OOF blend score (weights and score computed on the same data) is also
0.01606; the two are reported separately in the notebook because they are not the
same quantity in general. With only three weights to fit, their agreement is
expected and does not rule out leakage from the CV design (see Limitations).

Final blend weights (averaged across the nested-CV folds):

| Model | Weight |
|---|---|
| TabularResNet | 0.447 |
| FT-Transformer | 0.375 |
| CatBoost | 0.178 |

(Exact values: 0.4474, 0.3750, 0.1776.)

The ensemble is about 1.6% better than the best single model (ResNet).
TabularResNet and FT-Transformer carry about 82% of the weight. CatBoost, despite
the weakest standalone score, still earns 18% of it; a plausible reason is that
tree-based errors differ from the networks', but this was **not measured** here.

Per-fold scores in the notebook output are the mean of the per-seed losses; they
are higher than the OOF scores, which use seed-averaged predictions (bagging gain of
about 0.0005 per network).

A full run (5 folds × 2 networks × 3 seeds, plus the CatBoost CV) took about
129 minutes on a Kaggle notebook with 2× NVIDIA T4 (96 min neural-network CV, 30 min
CatBoost CV, under 1 min test-set inference). Wall-clock time is noisy on the
platform: the other configurations below, which train more jobs, finished in 106–113
minutes.

### Ablation: is an extra network worth it?

Three configurations with a fourth model were also run on the same folds. The base
is always ResNet + FT-Transformer + CatBoost.

| Configuration | Extra net OOF | Ensemble (nested) | Extra net weight | Total run |
|---|---|---|---|---|
| + MLP (PCA features) | 0.01649 | 0.01604 | 0.015 | 113 min |
| + 1D-CNN (PCA features) | 0.01661 | 0.01605 | 0.082 | 113 min |
| + 1D-CNN (raw features) | 0.01661 | 0.01606 | 0.070 | 106 min |
| **No extra network (final)** | – | **0.01606** | – | 129 min |

The MLP was redundant with the ResNet (same features, same model family) and was
replaced by a 1D-CNN with a different inductive bias. The CNN earns 7–8% of the
weight, but no configuration improves the ensemble: all four scores lie within
0.00002, below the fold standard deviation (0.00018) and the run-to-run GPU noise
(about 1e-5). Giving the CNN raw instead of PCA features did not change its own OOF
score (0.01661). The simpler 3-model blend was therefore kept as the final
solution: fewer components to train, explain and maintain, for the same score.

### Protocol note

An earlier version of the code selected the best epoch on the same fold that
produced the OOF predictions, giving an ensemble score of 0.01595. The final code
uses an inner validation split instead; the score moved to about 0.0161, so roughly
0.0001 of the earlier figure was selection optimism.

## Limitations

- **No grouping by compound.** The competition data has no drug identifier, so the
  same compound can appear in the training and validation split of a fold. Scores
  can be optimistic relative to the leaderboard.
- **Leaderboard score not measured.** No late submission was made; only
  cross-validated scores are reported.
- **Non-determinism.** Repeated GPU runs differ by about 1e-5 in OOF log loss.
- No ablation of individual design choices (PCA vs. raw features, focal loss, label
  smoothing) was run.

## Repository structure

```
merlonghi-moa-prediction/
├── README.md                          <- this file
├── software/
│     ├── README.md                    <- setup / how-to-run instructions
│     ├── moa_prediction.ipynb
│     ├── moa_notebook_landscape.pdf
│     └── requirements.txt             <- python packages to install
└── doc/
      ├── moa_presentation.pdf
      └── moa_presentation.tex         <- latex beamer (20 slides)
```

To rebuild the slides, compile `doc/moa_presentation.tex` with XeLaTeX or LuaLaTeX
(the Metropolis theme uses system fonts; pdfLaTeX works through fallback fonts).

## References

1. Kaggle, *Mechanisms of Action (MoA) Prediction*, 2020.
   <https://www.kaggle.com/competitions/lish-moa>
2. Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021).
   *Revisiting Deep Learning Models for Tabular Data.* NeurIPS 2021. (FT-Transformer)
3. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A. (2018).
   *CatBoost: unbiased boosting with categorical features.* NeurIPS 2018.
4. Sechidis, K., Tsoumakas, G., Vlahavas, I. (2011). *On the Stratification of
   Multi-Label Data.* ECML PKDD 2011.
5. Bradberry, T. *iterative-stratification* (Python package implementing [4]).
   <https://github.com/trent-b/iterative-stratification>
6. Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.*
   ICLR 2019. (AdamW)
7. Micikevicius, P. et al. (2018). *Mixed Precision Training.* ICLR 2018.
8. Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017). *Focal Loss for
   Dense Object Detection.* ICCV 2017.
9. Szegedy, C. et al. (2016). *Rethinking the Inception Architecture for Computer
   Vision.* CVPR 2016. (label smoothing)
10. Nelder, J.A., Mead, R. (1965). *A simplex method for function minimization.*
    The Computer Journal 7(4).
11. Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep
    Learning Library.* NeurIPS 2019.
12. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
    (`QuantileTransformer`, `PCA`)
