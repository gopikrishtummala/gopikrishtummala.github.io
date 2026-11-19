---
layout: ../../layouts/CheatsheetLayout.astro
title: "Trees & Ensembles in 2 Minutes Each"
description: "Intuition cards for decision trees, impurity, bagging, random forests, boosting, gradient boosting, XGBoost, LightGBM, CatBoost, feature importance and stacking."
---

Still the winners on most tabular data.

## Decision tree

Repeatedly split the data on the question that best separates the outcomes.

**Picture it:** a game of twenty questions that the computer learns to ask in the best order.

**Remember:** interpretable and needs no scaling, but one tree is unstable: small data changes can produce a very different tree.

## Impurity (Gini, entropy)

How mixed a group is; splits aim to create purer groups.

**Remember:** a node with all one class has impurity zero. Gini and entropy give nearly identical trees.

## Why ensembles work

Many decent, different models averaged together beat one good model.

**Picture it:** the "wisdom of crowds": a hundred people's averaged guess of a jar's jellybeans beats most individuals.

**Remember:** averaging cancels independent errors (lowers variance); boosting fixes errors in sequence (lowers bias).

## Bagging

Train many trees on random resamples of the data and average them.

**Remember:** reduces variance. Each tree misses about a third of the rows, which gives a free validation estimate ("out-of-bag" error).

## Random forest

Bagging plus random feature subsets at each split.

**Picture it:** a committee where each member is only allowed to look at some of the evidence, so they don't all make the same mistake.

**Remember:** strong out of the box, hard to overfit by adding trees, little tuning needed. A great second baseline after a linear model.

## Boosting

Train models one after another, each focusing on what the previous ones got wrong.

**Picture it:** a student who reviews only the questions they missed on the last practice test.

**Remember:** lowers bias; can overfit, so use a small learning rate and early stopping.

## Gradient boosting

Each new tree predicts the remaining error (the gradient of the loss) of the ensemble so far.

**Remember:** learning rate × number of trees is the key trade-off: smaller steps, more trees, better generalization.

## XGBoost, LightGBM, CatBoost

| Library | Remember it as |
|---|---|
| XGBoost | Accurate and battle-tested; uses curvature (second-order) information and regularized leaves |
| LightGBM | Fastest on big data; histogram splits, grows the most useful leaf first |
| CatBoost | Best with many categorical features; robust defaults |

**Remember:** all three handle missing values. Defaults are good; tune depth / leaves, learning rate, subsampling.

**Go deeper:** [XGBoost](/posts/fundamentals/xgboost-second-order-gradient-boosting).

## Key knobs

| Knob | Turn up to |
|---|---|
| Learning rate | Learn faster (risk overfitting); usually turn down and add trees |
| Max depth / leaves | Capture more interactions (risk overfitting) |
| Min samples per leaf | Smooth out noisy leaves |
| Row / column subsampling | Add randomness, reduce overfitting |

## Feature importance

Which features the model relied on.

**Remember:** built-in importance favors features with many unique values; permutation importance and SHAP values are more trustworthy.

## Stacking

Train a model on the predictions of other models.

**Remember:** small gains, more complexity; common in competitions, rarer in production.

## Gotchas

- Trees can't predict outside the range of targets they saw (no extrapolation).
- Deep learning rarely beats boosted trees on ordinary tabular data.
- Target encoding must be done out-of-fold.
