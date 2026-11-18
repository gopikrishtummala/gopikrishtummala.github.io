---
layout: ../../layouts/CheatsheetLayout.astro
title: "Stats & Learning Theory"
description: "Bias-variance, generalization, VC dimension, PAC learning, MLE vs MAP."
---

## Bias-variance trade-off

For squared error: $\mathbb{E}[(y-\hat{f})^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$.

| | High bias (underfit) | High variance (overfit) |
|---|---|---|
| Symptom | Train and validation error both high | Low train error, high validation error |
| Fix | More features, more complex model, less regularization | More data, regularization, simpler model, ensembles (bagging) |

- Learning curves: plot error vs training size to tell which you have.
- Deep networks can show **double descent**: test error falls again past the interpolation point.

## Generalization

- Generalization gap = test error − train error.
- Bounds take the form: test error ≤ train error + complexity term that shrinks with more data.
- Complexity measures: number of parameters (crude), VC dimension, Rademacher complexity, margin.
- In practice: validation sets and cross-validation, not bounds.

## VC dimension

- The largest number of points a model class can **shatter** (classify in every possible labeling).
- Linear classifiers in $d$ dimensions: VC = $d + 1$.
- Higher VC → more expressive → needs more data to generalize.
- Sample complexity roughly scales with VC dimension / $\epsilon$.

## PAC learning

- **Probably Approximately Correct:** with probability ≥ $1-\delta$, error ≤ $\epsilon$.
- Finite hypothesis class, realizable case: $m \ge \frac{1}{\epsilon}\left(\ln|H| + \ln\frac{1}{\delta}\right)$ examples suffice.
- Message: more hypotheses or tighter guarantees need more data.

## Estimation

| | Maximum likelihood (MLE) | Maximum a posteriori (MAP) |
|---|---|---|
| Objective | $\arg\max_\theta p(D \mid \theta)$ | $\arg\max_\theta p(D \mid \theta)\,p(\theta)$ |
| Prior | None | Yes; acts as regularization |
| Link | MSE = Gaussian noise MLE; cross-entropy = categorical MLE | L2 = Gaussian prior; L1 = Laplace prior |

- Full Bayesian: keep the whole posterior $p(\theta \mid D)$, not just its peak.

## Probability essentials

- Bayes: $p(A \mid B) = p(B \mid A)\,p(A) / p(B)$.
- Expectation is linear; variance of a sum adds only for independent variables.
- **Law of large numbers:** averages converge to the mean.
- **Central limit theorem:** sums of many independent variables are approximately normal.
- Common distributions: Bernoulli, binomial, categorical, Poisson, Gaussian, exponential, Beta (prior for probabilities), Dirichlet.

## Hypothesis testing and intervals

- **p-value:** probability of data at least this extreme if the null hypothesis is true. Not the probability the null is true.
- **Confidence interval:** range that would contain the true value in 95% of repeated experiments.
- **Multiple comparisons:** testing many things inflates false positives; use Bonferroni or false discovery rate control.
- **A/B tests:** fix sample size in advance (power analysis); don't stop early when it looks significant.
- **Bootstrap:** resample with replacement to estimate uncertainty of any statistic.

## Information theory

- Entropy $H(p) = -\sum p \log p$: uncertainty.
- Cross-entropy $H(p,q) = -\sum p \log q$: the classification loss.
- KL divergence $D_{KL}(p\|q) = \sum p \log(p/q) \ge 0$; not symmetric.
- Mutual information: how much knowing one variable reduces uncertainty about another.

## No free lunch

No algorithm is best for every problem. Inductive biases matched to the data are what make models work.
