# MoA Prediction — Multi-Model Ensemble

## Overview

This project addresses the Kaggle **Mechanisms of Action (MoA) Prediction** challenge:
given gene-expression and cell-viability profiles of human cells treated with a
chemical compound, predict which of 206 possible biological mechanisms of action
(MoAs) the compound triggers. It is a **multi-label classification** problem (a
sample can be positive for zero, one, or several of the 206 targets simultaneously)
and is evaluated with the mean per-target binary log loss.

- Competition page: <https://www.kaggle.com/competitions/lish-moa>
- Data: <https://www.kaggle.com/competitions/lish-moa/data>

## Objectives

- Build a robust cross-validated pipeline for a large-scale, highly imbalanced
  multi-label classification problem (~206 targets, many with very few positives).
- Compare several modelling paradigms on the same folds: a plain MLP, a residual
  tabular network (TabularResNet), an attention-based FT-Transformer, and a gradient
  boosted tree model (CatBoost), each fed a preprocessing pipeline suited to it
  (RankGauss + PCA for the dense nets, RankGauss without PCA for the tree model and
  the transformer).
- Reduce variance through multi-seed bagging and 5-fold multilabel-stratified cross
  validation, and combine the four base models with an ensemble whose weights are
  learned and *honestly* re-validated through a nested cross-validation scheme.
- Produce a clean, reproducible, well-documented deliverable (see `software/`).

## Approach summary

| Stage | Choice |
|---|---|
| Cross-validation | 5-fold `MultilabelStratifiedKFold`, `random_state=42` |
| Preprocessing (dense nets) | `QuantileTransformer` (RankGauss) → PCA (50 comps on gene features, 15 on cell-viability features) |
| Preprocessing (tree model / transformer) | `QuantileTransformer` (RankGauss), no PCA — preserves per-feature meaning |
| Base models | MLP, TabularResNet, FT-Transformer (3-seed bagging each), CatBoost (`MultiLogloss`) |
| Loss | `BCEWithLogitsLoss` (optionally focal loss), label smoothing 0.001 |
| Regularisation | Early stopping (patience 8), dropout, batch norm, weight decay |
| Ensembling | Softmax-reparameterised weight search (Nelder-Mead) on out-of-fold predictions, honesty-checked with nested CV |

Full architectural and implementation details, including the reasoning behind each
design choice, are documented inline in the notebook — see
[`software/moa_prediction.ipynb`](software/moa_prediction.ipynb).

## Results

Out-of-fold (OOF) mean log loss per model, over the 5 folds:

| Model | OOF log loss |
|---|---|
| MLP | 0.01635 |
| TabularResNet | 0.01620 |
| FT-Transformer | 0.01630 |
| CatBoost | 0.01700 |

**Ensemble (nested, honest estimate): 0.01595 ± 0.00020**
(naive OOF blend score, i.e. weights and score computed on the same data, was
identical to three decimal places here, but the two are reported separately in the
notebook because they are not the same quantity in general.)

Final blend weights (averaged across the nested-CV folds):

| Model | Weight |
|---|---|
| MLP | 0.073 |
| TabularResNet | 0.437 |
| FT-Transformer | 0.359 |
| CatBoost | 0.132 |

TabularResNet and the FT-Transformer carry most of the ensemble's weight; CatBoost,
despite the weakest standalone score, still earns a non-trivial weight because its
tree-based errors are the least correlated with the neural networks' errors.

A full training run (5 folds × 3 models × 3 seeds for the neural nets, plus the
CatBoost CV) took approximately 2 hours on a 2× NVIDIA T4 Kaggle notebook instance.

## Repository structure

```
merlonghi-moa-prediction/
├── software/
│     ├── README.md                    <- setup / how-to-run instructions
│     ├── moa_prediction.ipynb
│     ├── moa_notebook_landscape.pdf
│     └── requirements.txt             <- text file containing python packages to install
└── doc/
      ├── moa_presentation.pdf
      └── moa_presentation.tex      <- latex beamer 
```

## 6. References

1. Kaggle, *Mechanisms of Action (MoA) Prediction*, 2020.
   <https://www.kaggle.com/competitions/lish-moa>
2. Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021).
   *Revisiting Deep Learning Models for Tabular Data.* NeurIPS 2021. (FT-Transformer)
3. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A. (2018).
   *CatBoost: unbiased boosting with categorical features.* NeurIPS 2018.
4. Sechidis, K., Tsoumakas, G., Vlahavas, I. (2011). *On the Stratification of
   Multi-Label Data.* ECML PKDD 2011. (basis for `iterative-stratification`)
5. Szymański, P., Kajdanowicz, T. (2017). *A scikit-based Python environment for
   performing multi-label classification.* — `iterative-stratification` package.
6. Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep
   Learning Library.* NeurIPS 2019.
7. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.

