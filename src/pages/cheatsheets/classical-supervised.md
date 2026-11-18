---
layout: ../../layouts/CheatsheetLayout.astro
title: "Classical Supervised Learning"
description: "Linear and logistic regression, regularization, decision trees, k-NN and Naive Bayes."
---

## Linear regression

- Model: $\hat{y} = w^\top x + b$. Loss: mean squared error.
- Closed form: $w = (X^\top X)^{-1} X^\top y$; use gradient descent for large data.
- Assumptions: linear relationship, independent errors, constant variance, little multicollinearity.
- Coefficients: change in $y$ per unit change in a feature, holding others fixed.

## Logistic regression

- Model: $p(y=1 \mid x) = \sigma(w^\top x + b)$, with $\sigma(z) = 1/(1+e^{-z})$.
- Loss: binary cross-entropy (log loss); convex, no closed form.
- Coefficients are log-odds; $e^{w_j}$ is the odds ratio.
- Multi-class: softmax regression, or one-vs-rest.
- Linear decision boundary unless features are transformed.

## Regularization

| | Penalty | Effect |
|---|---|---|
| L2 (Ridge) | $\lambda \lVert w \rVert_2^2$ | Shrinks all weights; handles correlated features |
| L1 (Lasso) | $\lambda \lVert w \rVert_1$ | Drives some weights to exactly zero; feature selection |
| Elastic Net | mix of L1 and L2 | Sparse but stable with correlated features |

- Larger $\lambda$: simpler model, more bias, less variance. Tune with cross-validation.
- Standardize features first so the penalty treats them equally.
- Bayesian view: L2 = Gaussian prior on weights, L1 = Laplace prior.

## Decision trees

- Split recursively on the feature and threshold that most reduce impurity.
- Impurity: Gini $1-\sum p_k^2$; entropy $-\sum p_k \log p_k$; regression uses variance (MSE).
- Pros: interpretable, no scaling needed, handles mixed types and interactions.
- Cons: high variance, overfits; fix with max depth, min samples per leaf, pruning, or ensembles.

## k-nearest neighbors

- Predict from the k closest training points: majority vote or average.
- Small k: low bias, high variance. Large k: smoother, more bias.
- Needs scaled features; slow at prediction time; suffers in high dimensions (curse of dimensionality).
- Distance: Euclidean, Manhattan, cosine.

## Naive Bayes

- Bayes' rule with the "naive" assumption that features are independent given the class:
  $p(y \mid x) \propto p(y) \prod_j p(x_j \mid y)$.
- Variants: Gaussian (continuous), multinomial (counts, text), Bernoulli (binary).
- Fast, works with little data, strong text baseline; probabilities poorly calibrated.
- Use Laplace smoothing to avoid zero probabilities.

## Choosing quickly

| Situation | Try first |
|---|---|
| Small tabular data, need explanation | Logistic / linear regression, shallow tree |
| Tabular data, best accuracy | Gradient-boosted trees (see Trees & Ensembles) |
| Text classification baseline | TF-IDF + logistic regression, or Naive Bayes |
| Few features, irregular boundaries | k-NN, SVM with RBF kernel |
| Many features, few samples | L1 / L2 regularized linear models |

## Common gotchas

- Multicollinearity makes linear coefficients unstable (not predictions).
- Logistic regression on perfectly separable data: weights diverge without regularization.
- Trees and k-NN don't extrapolate beyond the range of training targets.
