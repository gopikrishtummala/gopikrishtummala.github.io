---
layout: ../../layouts/CheatsheetLayout.astro
title: "Probability, Stats & Learning Theory in 2 Minutes Each"
description: "Intuition cards for Bayes' theorem, MLE vs MAP, key distributions, the central limit theorem, p-values, confidence intervals, A/B tests, the bootstrap, entropy and KL divergence, VC dimension and PAC learning."
---

The statistics you need to reason about models and experiments.

## Bayes' theorem

**In one line:** update your belief when you see evidence: posterior ∝ likelihood × prior.

**Picture it:** a test for a rare disease is 99% accurate, yet most positives are false alarms, because the disease is so rare that false positives outnumber true ones.

**Remember:** always ask about the base rate.

## Likelihood vs probability

**In one line:** probability asks "given the model, how likely is this data?"; likelihood asks "given this data, how plausible is each model?"

## MLE vs MAP

**In one line:** MLE picks the parameters that make the data most likely; MAP also weighs a prior belief.

**Remember:** common losses are MLE in disguise: squared error = Gaussian noise, cross-entropy = categorical outcomes. Regularization is MAP: L2 = Gaussian prior, L1 = Laplace prior.

## Distributions worth knowing

| Distribution | Models |
|---|---|
| Bernoulli / binomial | One / many yes-no trials |
| Categorical | One of many classes |
| Gaussian | Sums of many small effects |
| Poisson | Counts of rare events per interval |
| Exponential | Waiting time between events |
| Beta / Dirichlet | Uncertainty about probabilities (priors) |

## Central limit theorem

**In one line:** averages of many independent things are approximately normal, whatever the original distribution.

**Remember:** why error bars and many statistical tests work, and why Gaussians appear everywhere.

## p-value

**In one line:** if there were truly no effect, how surprising would this result be?

**Watch out:** it's *not* the probability that the null hypothesis is true, nor the size of the effect. Small effects can be "significant" with huge samples.

## Confidence interval

**In one line:** a range that would contain the true value in, say, 95% of repeated experiments.

**Remember:** report intervals, not just point estimates. Overlapping intervals don't automatically mean "no difference."

## A/B testing

**In one line:** randomly split users, change one thing for one group, compare outcomes.

**Remember:** decide sample size in advance; don't stop the moment it looks significant ("peeking"); watch for novelty effects and multiple metrics.

## Multiple comparisons

**In one line:** test 20 things at 5% significance and one will look significant by chance.

**Remember:** correct for it (Bonferroni, false discovery rate) or pre-register your main metric.

## Bootstrap

**In one line:** resample your data with replacement many times to see how much a statistic varies.

**Remember:** a general-purpose way to get error bars for anything, including model metrics.

## Entropy, cross-entropy, KL divergence

**In one line:** entropy = uncertainty; cross-entropy = cost of using the wrong distribution; KL = the extra cost.

**Picture it:** entropy is how many yes/no questions you need on average to guess the outcome.

**Remember:** classification loss is cross-entropy; KL measures how far one distribution is from another (not symmetric). Both show up in VAEs, distillation and RLHF.

## Calibration

**In one line:** when a model says 70%, it should be right 70% of the time.

**Remember:** modern neural nets are often overconfident; fix with temperature scaling or isotonic regression. Matters whenever probabilities drive decisions.

## Generalization bounds

**In one line:** test error ≤ training error + a term that grows with model complexity and shrinks with data.

**Remember:** useful for intuition; in practice, validation sets tell you more.

## VC dimension

**In one line:** the largest number of points a model family can label in every possible way.

**Remember:** more expressive models need more data. A line in 2D has VC dimension 3.

## PAC learning

**In one line:** a learner is "probably approximately correct" if, with enough data, it's usually close to the best possible.

**Remember:** formalizes "more data → more confidence," with sample sizes that grow with model complexity.

## Correlation vs causation

**In one line:** two things moving together doesn't mean one causes the other.

**Picture it:** ice cream sales and drownings both rise in summer; heat causes both.

**Remember:** randomized experiments establish causation; observational data needs causal methods and assumptions.
