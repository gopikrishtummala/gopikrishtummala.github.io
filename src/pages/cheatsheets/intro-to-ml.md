---
layout: ../../layouts/CheatsheetLayout.astro
title: "ML Foundations in 2 Minutes Each"
description: "Intuition cards for what ML is, learning paradigms, generalization, overfitting, bias-variance, regularization, splits, cross-validation, leakage, inductive bias, the curse of dimensionality and double descent."
---

The ideas everything else is built on.

## What machine learning is

**In one line:** instead of writing rules, show examples and let the computer find the rule.

**Picture it:** you can't write rules for "is this a cat?", but you can show ten thousand labeled photos.

**Remember:** use ML when rules are hard to write and examples are plentiful. If a simple rule works, use the rule.

## Learning paradigms

| Paradigm | Learns from | Example |
|---|---|---|
| Supervised | Inputs with correct answers | Spam detection, price prediction |
| Unsupervised | Inputs only | Customer segments, anomaly detection |
| Self-supervised | Labels made from the data itself | Next-word prediction, masked images |
| Semi-supervised | A few labels, many unlabeled | Pseudo-labeling |
| Reinforcement | Rewards for actions | Games, robots, RLHF |

**Remember:** self-supervised learning on huge unlabeled data is what powers foundation models.

## Classification vs regression

**In one line:** classification predicts a category; regression predicts a number.

**Remember:** "will this customer churn?" is classification; "how much will they spend?" is regression.

## Generalization

**In one line:** doing well on data the model hasn't seen.

**Remember:** training accuracy is easy; generalization is the whole point. Everything about splits, regularization and validation exists to measure or improve it.

## Overfitting and underfitting

**In one line:** overfitting memorizes noise; underfitting misses the pattern.

**Picture it:** a student who memorized last year's exam answers (overfit) vs one who didn't study (underfit). Neither does well on a new exam.

| | Train error | Validation error | Fix |
|---|---|---|---|
| Underfit | High | High | Bigger model, better features, train longer |
| Overfit | Low | High | More data, regularization, simpler model, early stopping |

## Bias-variance trade-off

**In one line:** error comes from wrong assumptions (bias) and from sensitivity to the particular training set (variance).

**Picture it:** darts. High bias: tightly grouped but off-center. High variance: centered on average but scattered.

**Remember:** simple models: high bias, low variance. Complex models: low bias, high variance. Ensembles and regularization move you toward the sweet spot.

## Regularization

**In one line:** anything that discourages a model from fitting noise.

**Picture it:** a tax on complexity.

**Remember:** L2 shrinks all weights; L1 zeroes some out (feature selection); dropout, data augmentation and early stopping regularize neural nets. More data is the best regularizer.

## Train, validation, test

**In one line:** fit on train, tune on validation, report on test, once.

**Picture it:** practice problems, mock exams, the real exam. Studying the real exam questions makes your score meaningless.

**Remember:** split by time for time-dependent data; by user or patient when the same entity appears many times.

## Cross-validation

**In one line:** rotate which slice is the validation set and average the results.

**Remember:** a more reliable estimate when data is limited; 5 or 10 folds is typical.

## Data leakage

**In one line:** information from the answer sneaks into the inputs.

**Picture it:** a model predicting hospital readmission that uses "discharge summary written" — which only exists after the outcome is known.

**Remember:** if results look too good to be true, look for leakage first.

**Go deeper:** [Data & Evaluation](/cheatsheets/data-evaluation).

## Inductive bias

**In one line:** the assumptions a model builds in before seeing data.

**Remember:** CNNs assume nearby pixels matter together; linear models assume straight-line relationships; transformers assume little and need more data. Matching bias to the problem is why some models win.

## Curse of dimensionality

**In one line:** with many features, data becomes sparse and distances lose meaning.

**Picture it:** a few points fill a line, scatter across a square, and are lost in a 1000-dimensional cube.

**Remember:** hurts distance-based methods (k-NN, clustering). Use feature selection, dimensionality reduction or embeddings.

## Double descent

**In one line:** as models grow past the point of fitting the training data exactly, test error can fall again.

**Remember:** why huge, over-parameterized neural networks generalize despite classical theory predicting overfitting.

## No free lunch

**In one line:** no algorithm is best for every problem.

**Remember:** try simple baselines first; tabular data → boosted trees; images, text, audio → pretrained deep models.

## The ML project loop

1. Frame the problem and metric.
2. Get and inspect data.
3. Build a simple baseline.
4. Improve features and models.
5. Evaluate honestly.
6. Deploy, monitor, retrain.

**Go deeper:** [Project & Research Skills](/cheatsheets/project-research-skills).
