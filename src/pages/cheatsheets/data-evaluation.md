---
layout: ../../layouts/CheatsheetLayout.astro
title: "Data & Evaluation"
description: "Preprocessing, splits, cross-validation and evaluation metrics."
---

## Preprocessing

| Problem | Fix |
|---|---|
| Missing values | Drop; impute mean / median / mode; model-based imputation; add a "was missing" flag |
| Different scales | Standardize $(x-\mu)/\sigma$; min-max to [0, 1]; robust scaling (median, IQR) for outliers |
| Categorical features | One-hot (low cardinality); target / mean encoding (high cardinality, use out-of-fold); embeddings |
| Skewed features | Log or Box-Cox transform |
| Outliers | Clip, transform, or use robust losses (Huber, MAE) |
| Text | Tokenize; TF-IDF; pretrained embeddings |

- Fit scalers and encoders on **training data only**, then apply to validation and test.
- Scaling matters for k-NN, SVMs, linear models with regularization, and neural nets; not for trees.

## Splits

- Typical: 70/15/15 or 80/10/10. Large data: smaller val/test fractions are fine.
- **Stratify** splits for classification to keep class ratios.
- **Time series:** split by time; never train on the future.
- **Grouped data** (same user, patient, session): split by group so it doesn't appear on both sides.

## Cross-validation

- **k-fold:** split into k folds, train on k−1, validate on 1, rotate; average. k = 5 or 10.
- **Stratified k-fold:** keeps class ratios per fold.
- **Group k-fold:** no group in both train and validation.
- **Time-series CV:** expanding or rolling window, always validating on later data.
- **Nested CV:** inner loop tunes hyperparameters, outer loop estimates performance.
- **Leave-one-out:** k = n; low bias, high variance, expensive.

## Data leakage (the #1 silent bug)

- Features that wouldn't exist at prediction time (future data, the label in disguise).
- Preprocessing fit on all data before splitting.
- Duplicates or near-duplicates across train and test.
- Tuning on the test set.
- Symptom: validation results too good to be true.

## Classification metrics

Confusion matrix: TP, FP, TN, FN.

| Metric | Formula | Use when |
|---|---|---|
| Accuracy | $(TP+TN)/N$ | Classes balanced, errors equally costly |
| Precision | $TP/(TP+FP)$ | False positives are costly (spam filter) |
| Recall (sensitivity, TPR) | $TP/(TP+FN)$ | False negatives are costly (disease screening) |
| F1 | $2PR/(P+R)$ | Balance of precision and recall |
| Specificity | $TN/(TN+FP)$ | True negative rate |
| ROC-AUC | Area under TPR vs FPR | Ranking quality, threshold-free; optimistic under heavy imbalance |
| PR-AUC | Area under precision vs recall | Imbalanced data, positive class matters |
| Log loss | $-\frac{1}{N}\sum y\log p + (1-y)\log(1-p)$ | Probability quality |

- Thresholds: choose on validation for the trade-off you need.
- **Calibration:** do predicted probabilities match frequencies? Check reliability diagrams; fix with Platt scaling or isotonic regression.
- Multi-class: macro average (classes equal), micro average (examples equal), weighted.

## Regression metrics

- **MSE / RMSE:** penalizes large errors; RMSE in target units.
- **MAE:** robust to outliers; median-optimal.
- **$R^2$:** $1 - SS_{res}/SS_{tot}$; fraction of variance explained; can be negative.
- **MAPE:** percentage error; breaks near zero.

## Ranking and other metrics

- **Precision@k, Recall@k, MAP, MRR, NDCG:** ranking and retrieval.
- **BLEU / ROUGE:** text overlap (weak proxies for quality).
- **FID:** image generation realism and diversity.

## Class imbalance

- Use PR-AUC, F1 or recall at fixed precision, not accuracy.
- Resample (undersample majority, oversample minority, SMOTE) or use class weights.
- Tune the decision threshold.
