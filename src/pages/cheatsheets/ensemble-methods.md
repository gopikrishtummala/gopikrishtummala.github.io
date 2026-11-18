---
layout: ../../layouts/CheatsheetLayout.astro
title: "Trees & Ensembles"
description: "Decision trees, bagging, random forests, boosting, and XGBoost, LightGBM and CatBoost."
---

## Decision trees in one minute

- Greedy recursive splits that reduce impurity (Gini / entropy for classification, variance for regression).
- Stop or prune with: max depth, min samples per leaf, min impurity decrease, cost-complexity pruning ($\alpha$).
- One tree: interpretable, high variance. Ensembles fix the variance.

## Why ensembles work

- Averaging many models with partly independent errors reduces variance.
- Boosting sequentially fixes what previous models got wrong, reducing bias.
- Ensembles of trees are the default winners on tabular data.

## Bagging and random forests

- **Bagging:** train each tree on a bootstrap sample (sample with replacement); average or vote.
- **Random forest:** bagging plus a random subset of features at each split (typically $\sqrt{d}$ for classification, $d/3$ for regression), which decorrelates the trees.
- **Out-of-bag (OOB) error:** each tree skips ~37% of rows; evaluate on those for free validation.
- More trees never overfits by itself; it only stops improving.
- **Extra Trees:** random split thresholds too; faster, more randomness.

## Boosting

- **AdaBoost:** reweights misclassified examples; combines weak learners with weights.
- **Gradient boosting:** each new tree fits the negative gradient of the loss (residuals for MSE); prediction = sum of trees × learning rate.
- Learning rate (shrinkage) × number of trees trade off; smaller rate needs more trees, generalizes better.
- Use early stopping on a validation set.

## Modern gradient boosting libraries

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Key idea | Second-order (gradient + Hessian) objective, regularized | Histogram splits, leaf-wise growth, GOSS, feature bundling | Ordered boosting, symmetric (oblivious) trees |
| Strength | Accurate, mature, widely used | Very fast on large data | Categorical features out of the box, robust defaults |
| Watch out | Slower on very large data | Leaf-wise growth can overfit small data; limit `num_leaves` | Slower training than LightGBM |

- All handle missing values natively.
- XGBoost's split gain uses gradient sums $G$ and Hessian sums $H$: leaf weight $-G/(H+\lambda)$. See the [XGBoost post](/posts/fundamentals/xgboost-second-order-gradient-boosting).

## Key hyperparameters

| Parameter | Effect |
|---|---|
| `learning_rate` | Lower is more robust, needs more trees (0.01–0.1) |
| `n_estimators` | Set high, use early stopping |
| `max_depth` / `num_leaves` | Model complexity; main overfitting lever |
| `min_child_weight` / `min_data_in_leaf` | Prevents tiny, noisy leaves |
| `subsample`, `colsample_bytree` | Row / column sampling; adds randomness, reduces overfit |
| `lambda`, `alpha` | L2 / L1 regularization on leaf weights |

## Stacking and blending

- Train a meta-model on out-of-fold predictions of several base models.
- Gains are usually small; common in competitions, rarer in production.

## Feature importance

- **Split / gain importance:** built-in, biased toward high-cardinality features.
- **Permutation importance:** drop in score when a feature is shuffled; model-agnostic.
- **SHAP (TreeSHAP):** exact, consistent per-prediction attributions for trees.

## Gotchas

- Trees can't extrapolate: predictions stay within the training target range.
- Target encoding must be done out-of-fold to avoid leakage.
- Random forests rarely need much tuning; boosting does.
