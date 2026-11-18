---
layout: ../../layouts/CheatsheetLayout.astro
title: "Intro to ML"
description: "Types of learning, the ML pipeline and key vocabulary."
---

## What ML is

- Learn a function $f: X \to Y$ from data instead of writing rules by hand.
- Use when rules are hard to write (images, language), patterns change, or data is plentiful and expertise scarce.
- Don't use when a simple rule works, data is tiny or unrepresentative, or errors are unacceptable and unexplainable.

## Types of learning

| Type | Data | Goal | Examples |
|---|---|---|---|
| Supervised | Inputs with labels | Predict labels | Classification, regression |
| Unsupervised | Inputs only | Find structure | Clustering, dimensionality reduction |
| Self-supervised | Inputs; labels made from the data | Learn representations | Next-token prediction, masked images, contrastive learning |
| Semi-supervised | Few labels, many unlabeled | Use both | Pseudo-labeling, consistency training |
| Reinforcement | States, actions, rewards | Maximize long-term reward | Games, robotics, RLHF |

## Vocabulary

- **Feature:** an input variable. **Label / target:** what you predict.
- **Model / hypothesis:** the learned function. **Parameters:** learned values. **Hyperparameters:** set before training (learning rate, depth).
- **Loss:** error on one example. **Cost / objective:** average loss, often plus regularization.
- **Training / validation / test set:** fit / tune / final report, once.
- **Generalization:** performance on unseen data. **Overfitting:** fits noise, low train error, high test error. **Underfitting:** too simple, both errors high.
- **Inductive bias:** the assumptions a model makes (linearity, locality in CNNs).
- **Inference:** using a trained model to predict.

## The ML pipeline

1. **Frame the problem:** what decision does the prediction support? What metric matters? What's the baseline?
2. **Collect and label data;** check for leakage and bias.
3. **Explore:** distributions, missing values, class balance, correlations.
4. **Preprocess and engineer features.**
5. **Split** train / validation / test (by time or group if needed).
6. **Train a simple baseline,** then better models.
7. **Tune** on validation, **evaluate** once on test.
8. **Deploy, monitor** for drift, retrain.

## Classification vs regression

| | Classification | Regression |
|---|---|---|
| Output | Category or probability | Continuous number |
| Typical loss | Cross-entropy | Mean squared error, MAE, Huber |
| Typical metrics | Accuracy, precision, recall, F1, AUC | RMSE, MAE, $R^2$ |

## Rules of thumb

- Start with a dumb baseline (majority class, mean, a linear model).
- More good data usually beats a cleverer model.
- Tabular data: gradient-boosted trees first. Images, text, audio: pretrained deep models first.
- Never touch the test set until the end.

## Short history

| Era | Milestones |
|---|---|
| 1950s–60s | Perceptron (1958), early AI optimism |
| 1980s | Backpropagation popularized (1986), decision trees |
| 1990s–2000s | SVMs, boosting, random forests, kernel methods |
| 2012 | AlexNet wins ImageNet; deep learning takes off |
| 2017 | Transformer ("Attention Is All You Need") |
| 2020s | Large language models, diffusion models, foundation models |
