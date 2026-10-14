# MoA Prediction — Multi-Model Ensemble

Author: Marco Merlonghi — Project Work in Machine Learning and Data Mining

## Overview

This project addresses the Kaggle **Mechanisms of Action (MoA) Prediction** challenge: given gene-expression and cell-viability profiles of human cells treated with a chemical compound, predict which of 206 possible biological mechanisms of action (MoAs) the compound triggers. It is a **multi-label classification** problem (a sample can be positive for zero, one, or several of the 206 targets simultaneously) evaluated with the mean binary log loss over all samples and targets [1].

- Competition page: <https://www.kaggle.com/competitions/lish-moa>
- Data: <https://www.kaggle.com/competitions/lish-moa/data>

The solution is delivered as two notebooks (see [`software/`](software/)): a **training** notebook, which cross-validates the models and saves them, and an **inference** notebook, which loads the saved models and produces the Kaggle submission. The final model was submitted once to the Kaggle test set as a late submission (see [Results](#results)).

### Data in brief

Figures from the exploratory analysis in the training notebook (Step 2.3):
- 23,814 training rows: 21,948 treated and 1,866 control rows (7.8%, `cp_type == ctl_vehicle`). Controls have no MoA by definition: none of them has a positive label. They are excluded from training and set to 0 at submission.
- 772 gene-expression features (`g-`), 100 cell-viability features (`c-`), plus the categorical `cp_time` (24, 48, 72 h) and `cp_dose` (D1, D2).
- Positives are very rare: 0.373% of the (treated row, target) cells are positive. 34.2% of the treated rows have no label, 57.1% have one, 8.7% have two or more (at most 7).
- The 206 targets are highly imbalanced: positives per target range from 1 to 832 (median 38), and 53 targets have fewer than 20 positives.

## Objectives

- Build a robust cross-validated pipeline for a large-scale, highly imbalanced multi-label classification problem.
- Compare several modelling paradigms on the same folds: a residual tabular network (TabularResNet), an attention-based FT-Transformer [2], and a gradient-boosted tree model (CatBoost) [3], each fed a preprocessing pipeline suited to it.
- Reduce variance through multi-seed bagging and 5-fold multilabel-stratified cross validation [4, 5], and combine the base models with an ensemble whose weights are learned and re-validated through nested cross-validation.
- Check the cross-validated estimate against an independent measure: the Kaggle test set, used once and never for tuning.
- Produce a clean, reproducible, well-documented deliverable (see `software/`).

## Approach summary

| Stage | Choice |
|---|---|
| Cross-validation | 5-fold `MultilabelStratifiedKFold`, `random_state=42`, on treated rows only |
| Model selection (neural nets) | Inner stratified split (10% of each training fold) for early stopping, LR scheduling and best-epoch choice; the outer fold is scored once |
| Preprocessing (ResNet) | `QuantileTransformer` (RankGauss) → PCA (50 components on gene features, 15 on cell-viability features), concatenated to the transformed features [11] |
| Preprocessing (FT-Transformer, CatBoost) | `QuantileTransformer` (RankGauss), no PCA [11] |
| Base models | TabularResNet, FT-Transformer (3-seed bagging each), CatBoost (`MultiLogloss`, fixed 250 iterations) |
| Constant targets (CatBoost) | A target with no positive in a training fold is excluded from the fit and predicted with a smoothed prior (Jeffreys, 0.5 pseudo-counts) instead of 0 |
| Loss | `BCEWithLogitsLoss` [10], label smoothing 0.001 [8] |
| Optimisation | AdamW [6], `ReduceLROnPlateau`, mixed precision [7] |
| Regularisation | Early stopping (patience 8), dropout, batch norm, weight decay |
| Ensembling | Softmax-reparameterised weight search (Nelder–Mead [9]) on out-of-fold predictions, re-validated with nested CV |
| Post-processing | Control-vehicle test rows set to 0 |
| Inference | Separate notebook: loads the 30 network checkpoints, 5 CatBoost models, 10 fold pipelines and the blend weights; no training |

Architectural and implementation details are documented inline in the notebooks — see [`software/moa_prediction.ipynb`](software/moa_prediction.ipynb) (training) and [`software/moa_inference.ipynb`](software/moa_inference.ipynb) (inference). The FT-Transformer is a simplified variant of the one in [2] (post-LayerNorm encoder, GELU feed-forward, no ReGLU). Setup and run instructions are in [`software/README.md`](software/README.md); the slides are in [`doc/`](doc/).

## Results

### Cross-validation

Out-of-fold (OOF) mean log loss over the 5 folds, treated rows only (lower is better). Network scores use predictions averaged over 3 seeds.

| Model | OOF log loss |
|---|---|
| TabularResNet | **0.01632** |
| FT-Transformer | 0.01642 |
| CatBoost | 0.01699 |
| **Ensemble (nested, honest estimate)** | **0.01605 ± 0.00019** |

Two blend scores are computed. The *nested* score fits the weights on 4 folds and scores them on the fifth, for each fold in turn. The *in-sample* score fits the weights on all OOF rows and scores them on the same rows. They agree to five decimals (0.01605); the difference is 9.4e-7, which measures the optimism of the blending step. With only three weights to fit it is negligible. This agreement says nothing about leakage from the CV design itself (see Limitations).

Final blend weights (averaged across the nested-CV folds):

| Model | Weight |
|---|---|
| TabularResNet | 0.436 |
| FT-Transformer | 0.387 |
| CatBoost | 0.177 |

(Exact values: 0.4356, 0.3870, 0.1774.)

The ensemble is about 1.7% better than the best single model (ResNet). TabularResNet and FT-Transformer carry about 82% of the weight. CatBoost, despite the weakest standalone score, still earns 18% of it; a plausible reason is that tree-based errors differ from the networks', but this was **not measured** here.

Per-fold scores in the notebook output are the mean of the per-seed losses; they are higher than the OOF scores, which use seed-averaged predictions (bagging gain of about 0.0005 per network).

A full training run (5 folds × 2 networks × 3 seeds, plus the CatBoost CV) took about 131 minutes on a Kaggle notebook with 2× NVIDIA T4 (98.5 min neural-network CV, 29.1 min CatBoost CV, under 1 min test-set inference). Wall-clock time is noisy on the platform: the ablation configurations below, which train more jobs, finished in 106–113 minutes.

### Independent check on the Kaggle test set

The final model was submitted **once**, as a late submission, through the inference notebook. The Kaggle test set was never used for tuning.

| Estimate | Log loss |
|---|---|
| Nested-CV blend, treated rows only | 0.01605 ± 0.00019 |
| Nested-CV blend, rescaled to all test rows | 0.01461 |
| Kaggle private score | **0.01634** |
| Kaggle public score | 0.01864 |

The cross-validation scores are computed on treated rows only, whereas the Kaggle score averages over all test rows. Control rows are predicted as 0 with true value 0 and add almost no loss, so the all-rows score is approximately the treated-only score multiplied by the share of treated rows. That share is 0.910 in the public test set, which gives the rescaled estimate 0.01605 × 0.910 = 0.01461. The share in the hidden part of the test set may differ slightly, so the rescaling is approximate.

The private score is 11.8% above the rescaled CV estimate and the public score is 27.6% above it. The private score, computed on more rows, is the more reliable of the two. The cross-validated estimate is therefore optimistic. A likely cause is the lack of grouping by compound (see Limitations), but this explanation was **not tested**. The consistency of the inference notebook was checked on the public test set: its predictions differ from those of the training notebook by at most 1.1e-16.

### Ablation: is an extra network worth it?

Three configurations with a fourth model were also run on the same folds. The base is always ResNet + FT-Transformer + CatBoost. The ablation was run on an earlier revision of the code, before the final fixes (among them the smoothed prior for constant CatBoost targets), so its absolute values differ from the final run in the last digit.

| Configuration | Extra net OOF | Ensemble (nested) | Extra net weight | Total run |
|---|---|---|---|---|
| + MLP (PCA features) | 0.01649 | 0.01604 | 0.015 | 113 min |
| + 1D-CNN (PCA features) | 0.01661 | 0.01605 | 0.082 | 113 min |
| + 1D-CNN (raw features) | 0.01661 | 0.01606 | 0.070 | 106 min |
| No extra network (baseline at the time of the ablation) | – | 0.01606 | – | 129 min |
| **No extra network (final rerun)** | – | **0.01605** | – | 131 min |

The MLP was redundant with the ResNet (same features, same model family) and was replaced by a 1D-CNN with a different inductive bias. The CNN earns 7–8% of the weight, but no configuration improves the ensemble: all scores lie within 0.00002, below the fold standard deviation (0.00019) and the run-to-run GPU noise (about 1e-5). Giving the CNN raw instead of PCA features did not change its own OOF score (0.01661). The simpler 3-model blend was therefore kept as the final solution: fewer components to train, explain and maintain, for the same score. The code of the MLP and 1D-CNN is not part of this repository, so this ablation cannot be reproduced from it.

### Protocol note

An earlier version of the code selected the best epoch on the same fold that produced the OOF predictions, giving an ensemble score of 0.01595. The final code uses an inner validation split instead, and the score moved to about 0.0161. The difference of about 0.0001 is consistent with selection optimism, but it also includes the effect of training on 10% less data per fold and the run-to-run noise (about 1e-5), so it is not an exact measurement of the optimism.

## Limitations

- **No grouping by compound.** The competition data has no drug identifier, so the same compound can appear in the training and validation split of a fold, while the test set contains different compounds. The Kaggle private score (0.01634) is above the rescaled CV estimate (0.01461), which is consistent with this limitation.
- **Selection bias from tuning.** Hyperparameters, architectures and the final 3-model choice were selected by looking at OOF scores on the same 5 folds that are reported. No hold-out was reserved before tuning, and nested cross-validation of the whole procedure was too costly (estimated at tens of hours on the available GPUs). The CV scores are therefore optimistic by an unknown amount. The differences between the configurations that were compared are below the fold standard deviation.
- **Preprocessing fitted on the whole training fold.** The RankGauss and PCA transforms are fitted on the entire training fold, including the 10% inner validation rows. The transforms are unsupervised, so no labels leak and the outer fold is never seen; the only possible effect is a slightly optimistic early-stopping choice.
- **CV on treated rows only.** CV scores are comparable with leaderboard scores only after rescaling by the treated share (see above).
- **One submission, noisy test subsets.** The Kaggle score comes from a single submission. Public and private scores differ by 0.0023, more than ten times the fold standard deviation, so the fold standard deviation understates the uncertainty on unseen data. Late submissions do not appear on the leaderboard.
- **Non-determinism.** Repeated GPU runs differ by about 1e-5 in OOF log loss.
- No ablation of individual design choices (PCA vs. raw features, label smoothing) was run.
- **Inference depends on saved artifacts.** The inference notebook needs the Kaggle Dataset of trained models, which is not stored in this repository; it is produced by the training notebook (see `software/README.md`).

## Repository structure

```
merlonghi-moa-prediction/
├── README.md                          <- this file
├── software/
│     ├── README.md                    <- setup / how-to-run instructions
│     ├── moa_prediction.ipynb         <- training and cross-validation
│     ├── moa_notebook_landscape.pdf   <- landscape export of the training notebook
│     ├── moa_inference.ipynb          <- inference and Kaggle submission
│     ├── moa_inference_landscape.pdf  <- landscape export of the inference notebook
│     └── requirements.txt             <- python packages to install
└── doc/
      ├── moa_presentation.pdf
      └── moa_presentation.tex         <- latex beamer
```

## References

1. Kaggle, *Mechanisms of Action (MoA) Prediction*, 2020. <https://www.kaggle.com/competitions/lish-moa>
2. Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data.* NeurIPS 2021. (FT-Transformer)
3. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A. (2018). *CatBoost: unbiased boosting with categorical features.* NeurIPS 2018.
4. Sechidis, K., Tsoumakas, G., Vlahavas, I. (2011). *On the Stratification of Multi-Label Data.* ECML PKDD 2011.
5. Bradberry, T. *iterative-stratification* (Python package implementing [4]). <https://github.com/trent-b/iterative-stratification>
6. Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019. (AdamW)
7. Micikevicius, P. et al. (2018). *Mixed Precision Training.* ICLR 2018.
8. Szegedy, C. et al. (2016). *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016. (label smoothing)
9. Nelder, J.A., Mead, R. (1965). *A simplex method for function minimization.* The Computer Journal 7(4).
10. Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
11. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12. (`QuantileTransformer`, `PCA`)