---
layout: ../../layouts/CheatsheetLayout.astro
title: "Classical Supervised Learning in 2 Minutes Each"
description: "Intuition cards for linear regression, logistic regression, regularization, k-nearest neighbors, Naive Bayes, decision trees, and how to choose between them."
---

The models every practitioner should be able to explain in a sentence.

## Linear regression

Fit the straight line (or flat plane) that best passes through the data.

**Picture it:** stretching a rubber band through a scatter plot so the total squared distance to the points is smallest.

**Remember:** each coefficient is "how much the prediction changes per unit of this feature, holding others fixed." Fast, interpretable, a great baseline.

**Watch out:** correlated features make coefficients unstable and misleading even when predictions are fine.

## Logistic regression

A linear model whose output is squashed through a sigmoid into a probability.

**Picture it:** a straight line separating two classes, with a soft ramp of probability across it.

**Remember:** despite the name, it's for classification. Coefficients are changes in log-odds. The go-to baseline for binary classification and text with TF-IDF features.

## Regularization (Ridge, Lasso)

Penalize large weights so the model doesn't chase noise.

**Picture it:** Ridge (L2) shrinks every weight a little; Lasso (L1) pushes unimportant ones all the way to zero.

**Remember:** Lasso for automatic feature selection, Ridge for many correlated features, Elastic Net for both. Scale features first.

## k-nearest neighbors

Predict by asking the k most similar training examples.

**Picture it:** "you are the average of your five closest friends."

**Remember:** no training, slow prediction, needs scaled features, struggles in high dimensions. Small k = jumpy; large k = smooth.

## Naive Bayes

Apply Bayes' rule assuming features are independent given the class.

**Picture it:** a spam filter that multiplies "how spammy is each word" together, ignoring that words co-occur.

**Remember:** the independence assumption is wrong, yet it works well for text and tiny data. Probabilities are poorly calibrated; rankings are often fine.

## Decision tree

A flowchart of yes/no questions learned from data.

**Picture it:** "Is income > 50k? → Is age < 30? → approve."

**Remember:** interpretable, needs no scaling, captures interactions; a single tree overfits easily. Ensembles of trees are far stronger.

**Go deeper:** [Trees & Ensembles](/cheatsheets/ensemble-methods).

## Linear vs non-linear boundaries

**Remember:** linear models draw straight boundaries unless you engineer features (polynomials, interactions, binning). Trees, k-NN and kernel SVMs draw curved ones natively.

## Generative vs discriminative

Generative models learn how each class produces data; discriminative models learn the boundary directly.

**Remember:** Naive Bayes is generative; logistic regression is discriminative. Discriminative usually wins with enough data; generative can win with very little.

## Choosing a model

| Situation | Start with |
|---|---|
| Need explanation, small data | Logistic or linear regression |
| Tabular data, best accuracy | Gradient-boosted trees |
| Text classification baseline | TF-IDF + logistic regression |
| Very little data, many features | Regularized linear model or Naive Bayes |
| Irregular boundaries, few features | k-NN or SVM with RBF kernel |

## Common mistakes

- Comparing coefficients across features with different scales.
- Treating logistic regression's probabilities as calibrated without checking.
- Expecting trees or k-NN to extrapolate beyond the training range.
