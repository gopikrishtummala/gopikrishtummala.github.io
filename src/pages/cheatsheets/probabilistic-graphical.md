---
layout: ../../layouts/CheatsheetLayout.astro
title: "Probabilistic & Graphical Models in 2 Minutes Each"
description: "Intuition cards for graphical models, Bayesian networks, Markov random fields, conditional independence, EM, Gaussian mixtures, hidden Markov models, Kalman filters, inference and causal graphs."
---

Reasoning about uncertainty among many connected variables.

## Graphical models

**In one line:** draw variables as nodes and direct dependencies as edges, so a huge joint distribution becomes many small pieces.

**Picture it:** a family tree of causes: rain → wet grass ← sprinkler.

**Remember:** the graph tells you what's independent of what, which is what makes reasoning tractable.

## Bayesian networks

**In one line:** a directed graph where each variable depends only on its parents.

**Remember:** natural for "this causes that" stories and expert systems; the probability of everything = product of each variable given its parents.

## Explaining away

**In one line:** two independent causes become dependent once you observe their shared effect.

**Picture it:** the grass is wet. Learning that the sprinkler was on makes rain *less* likely, even though rain and sprinkler are unrelated.

## Markov random fields

**In one line:** undirected graphs for symmetric relationships, like neighboring pixels tending to match.

**Remember:** conditional random fields (CRFs) are the supervised version, classic for sequence labeling.

## Expectation-maximization (EM)

**In one line:** when some variables are hidden, alternate between guessing them and refitting the model.

**Picture it:** sorting mixed coins by guessing which bag each came from, re-estimating each bag's weight, and repeating.

**Remember:** always improves the fit but can get stuck; restart from several initializations.

## Gaussian mixture models

**In one line:** data drawn from several overlapping bell curves; EM figures out which point came from which.

**Remember:** soft clustering with probabilities; k-means is a hard, simplified version.

## Hidden Markov models

**In one line:** a hidden state moves step by step; you only see noisy signals it emits.

**Picture it:** guessing the weather (hidden) from whether someone carries an umbrella each day (observed).

| Question | Algorithm |
|---|---|
| How likely is this observed sequence? | Forward algorithm |
| What's the most likely hidden sequence? | Viterbi |
| Learn the model from data | Baum-Welch (EM) |

## Kalman filter

**In one line:** track a moving thing by blending a motion prediction with noisy measurements, weighted by how much you trust each.

**Picture it:** estimating a car's position from a fast-but-drifting speedometer and a slow-but-absolute GPS.

**Remember:** the core of tracking and localization; particle filters handle non-linear, non-Gaussian cases.

**Go deeper:** [Module 4: Localization](/posts/robotics/autonomous-stack-module-4-localization).

## Inference

**In one line:** answering questions like "given what I observed, what's the probability of X?"

**Remember:** exact methods (variable elimination, belief propagation on trees) work on simple graphs; sampling (MCMC) and variational methods approximate the rest.

## Topic models (LDA)

**In one line:** each document is a mixture of topics; each topic is a mixture of words.

**Remember:** classic unsupervised text analysis; often replaced today by clustering embeddings.

## Probabilistic programming

**In one line:** write the model as code; the library does the inference.

**Remember:** Stan, PyMC, NumPyro make Bayesian modeling practical.

## Causal graphs

**In one line:** graphs that encode cause and effect, so you can reason about interventions.

**Remember:** "what happens if I *change* X?" differs from "what's usually seen *with* X?" Confounders create misleading correlations; randomized experiments break them.
