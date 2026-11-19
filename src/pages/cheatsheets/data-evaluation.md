---
layout: ../../layouts/CheatsheetLayout.astro
title: "Data & Evaluation in 2 Minutes Each"
description: "Intuition cards for normalization, missing values, categorical encoding, imbalance, leakage, splits, precision and recall, ROC vs PR curves, F-beta, calibration, regression metrics, ranking metrics and drift."
---

Good models start with good data and honest measurement.

## Standardization vs normalization

**In one line:** standardize = mean 0, standard deviation 1; normalize = squeeze into [0, 1].

**Remember:** scale features for distance-based and gradient-based models (k-NN, SVMs, linear models, neural nets). Trees don't care. Fit the scaler on training data only.

## Missing values

**In one line:** fill them sensibly, and often record that they were missing.

**Remember:** "missing" can be information itself (a blank income field). Median imputation plus a missing-flag is a solid default; boosted trees handle missing values natively.

## Categorical features

**In one line:** turn categories into numbers the model can use.

**Remember:** one-hot for few categories; target encoding or learned embeddings for many (thousands of zip codes). Target encoding must be computed out-of-fold to avoid leakage.

## Class imbalance

**In one line:** when one class is rare, accuracy lies.

**Picture it:** 99% of transactions are legitimate; a model that always says "legit" is 99% accurate and useless.

**Remember:** use precision, recall, PR-AUC; reweight classes or resample; tune the decision threshold.

## Data leakage

**In one line:** the model sees something during training it won't have at prediction time.

**Remember:** split before preprocessing; check features for "future" information; dedupe across splits. Too-good results → suspect leakage.

## Splitting data

**Remember:** random split for independent examples; time-based for anything temporal; group-based when one user or patient has many rows; stratified to keep class ratios.

## Confusion matrix

**In one line:** a 2×2 table of true positives, false positives, true negatives, false negatives.

**Remember:** every classification metric is a different way to summarize this table.

## Precision vs recall

**In one line:** precision = of what I flagged, how much was right; recall = of what was there, how much did I find.

**Picture it:** a fishing net. Precision: how much of the catch is fish (not junk). Recall: how many of the fish in the lake you caught.

**Remember:** high precision matters when false alarms are costly (spam filter); high recall when misses are costly (cancer screening). Moving the threshold trades one for the other.

## F1 and F-beta

**In one line:** combine precision and recall into one number.

**Remember:** F1 weighs them equally; F2 favors recall; F0.5 favors precision.

## ROC-AUC vs PR-AUC

**In one line:** ROC plots true positive rate vs false positive rate; PR plots precision vs recall.

**Remember:** ROC-AUC = chance a random positive ranks above a random negative. With heavy imbalance, ROC-AUC looks great while PR-AUC reveals the truth.

## Calibration

**In one line:** do predicted probabilities match real frequencies?

**Remember:** a model can rank well (high AUC) and still be badly calibrated. Check with a reliability diagram.

## Regression metrics

| Metric | Intuition |
|---|---|
| MAE | Average miss, in target units; robust to outliers |
| RMSE | Like MAE but punishes big misses more |
| $R^2$ | Fraction of variance explained; 1 is perfect, 0 is "predict the mean" |
| MAPE | Percentage error; breaks when actual values near zero |

## Ranking metrics

**In one line:** for search and recommendations, order matters.

**Remember:** Recall@k (is it in the top k?), MRR (how high is the first correct one?), NDCG (rewards putting the best items at the top).

## Evaluating generative models

**Remember:** BLEU and ROUGE measure word overlap, weak proxies for quality; human ratings and LLM judges are common, each with biases.

## Drift

**In one line:** the world changes after deployment.

**Remember:** data drift = inputs change; concept drift = the input-output relationship changes. Monitor both and retrain.
