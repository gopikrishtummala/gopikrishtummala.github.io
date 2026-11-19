---
layout: ../../layouts/CheatsheetLayout.astro
title: "SVMs & Bayesian Methods in 2 Minutes Each"
description: "Intuition cards for support vector machines, margins, the kernel trick, Bayesian inference, conjugate priors, Bayesian linear regression, Gaussian processes, MCMC and variational inference."
---

Two classic ideas: draw the widest possible boundary, and keep track of uncertainty.

## Support vector machine

Find the boundary that separates classes with the widest possible gap.

**Picture it:** building the widest road between two neighborhoods; the houses touching the road's edges are the **support vectors**, and only they decide where the road goes.

**Remember:** wide margins generalize well. Only a few training points matter in the end.

## Soft margin and C

Allow some points on the wrong side, at a cost.

**Remember:** large C = few mistakes allowed, narrow road, overfit risk; small C = wider road, more mistakes, smoother boundary.

## Kernel trick

Separate data that isn't linearly separable by implicitly mapping it to a higher-dimensional space.

**Picture it:** red dots in the middle of a ring of blue dots can't be split by a line; lift the middle up into 3D and a flat sheet separates them.

**Remember:** kernels compute similarities as if in the higher space, without ever building it. RBF is the usual default; γ controls how local each point's influence is.

**Watch out:** kernel SVMs scale poorly past tens of thousands of examples; scale features first.

## Bayesian thinking

Start with a belief (prior), update it with data (likelihood), get a new belief (posterior).

**Picture it:** you think a coin is fair; after ten heads in a row, you're less sure. Bayesian methods do this formally.

**Remember:** the payoff is uncertainty: not just "the answer is 5" but "probably between 4 and 6."

## Priors

What you believe before seeing data.

**Remember:** with little data, priors matter a lot; with lots of data, the data wins. Regularization is a prior in disguise.

## Conjugate priors

Prior-likelihood pairs where the posterior has the same form as the prior, so updates are just arithmetic.

**Remember:** Beta for coin flips and click rates, Dirichlet for categories, Gaussian for Gaussians. Beta(successes + 1, failures + 1) is the classic click-rate estimate.

## Bayesian linear regression

Linear regression that returns a distribution over lines instead of one line.

**Remember:** predictions come with error bars that widen away from the data, exactly where you should be less sure.

## Gaussian processes

A distribution over smooth functions, updated by data.

**Picture it:** a bundle of possible curves pinned down near observed points, fanning out between them.

**Remember:** great uncertainty for small data and the engine of Bayesian hyperparameter optimization. Cost grows with the cube of the number of points.

## MCMC

Draw samples from the posterior by a guided random walk.

**Remember:** exact in the long run, slow. Hamiltonian Monte Carlo (NUTS, in Stan and PyMC) is the modern workhorse.

## Variational inference

Approximate the posterior with a simpler distribution fitted by optimization.

**Remember:** much faster than MCMC, tends to underestimate uncertainty. VAEs use it.

## When to reach for these

| Need | Use |
|---|---|
| Strong classifier on small-to-medium tabular or text data | SVM (linear for text, RBF otherwise) |
| Uncertainty with little data | Bayesian models, Gaussian processes |
| Tuning expensive experiments | Bayesian optimization with a GP |
| Principled pooling across groups | Hierarchical Bayesian models |
