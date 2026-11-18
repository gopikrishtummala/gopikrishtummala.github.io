---
layout: ../../layouts/CheatsheetLayout.astro
title: "Probabilistic & Graphical Models"
description: "Mixture models, EM, hidden Markov models, Bayesian networks, Markov random fields and inference."
---

## Why graphical models

- Represent a joint distribution over many variables compactly using conditional independence.
- Nodes = random variables; edges = direct dependence.

| | Bayesian network | Markov random field |
|---|---|---|
| Graph | Directed acyclic | Undirected |
| Factorization | $p(x) = \prod_i p(x_i \mid \text{parents}(x_i))$ | $p(x) = \frac{1}{Z}\prod_c \psi_c(x_c)$ |
| Good for | Causal-style generative stories | Symmetric relations (pixels, spatial) |
| Hard part | Structure learning | Partition function $Z$ |

- **d-separation:** read conditional independence off a Bayesian network.
- Conditional random fields (CRFs): discriminative MRFs for sequence labeling.

## Gaussian mixture models

- $p(x) = \sum_k \pi_k\,\mathcal{N}(x \mid \mu_k, \Sigma_k)$.
- Soft clustering; k-means is the limit with shared spherical covariances and hard assignments.
- Choose k with BIC / AIC.

## Expectation-maximization (EM)

- For models with hidden variables.
- **E-step:** compute the posterior over hidden variables given current parameters (responsibilities).
- **M-step:** maximize expected complete-data log-likelihood to update parameters.
- Never decreases the likelihood; converges to a local optimum, so run several initializations.
- Uses: GMMs, HMMs (Baum-Welch), missing data, topic models.

## Markov chains and hidden Markov models

- Markov property: the future depends only on the present state.
- HMM: hidden states $z_t$ with transitions $p(z_t \mid z_{t-1})$, emitting observations $p(x_t \mid z_t)$.

| Task | Algorithm |
|---|---|
| Likelihood of a sequence | Forward algorithm |
| Most likely hidden sequence | Viterbi (dynamic programming) |
| Posterior of each state | Forward-backward |
| Learn parameters | Baum-Welch (EM) |

- Uses: speech (historically), part-of-speech tagging, bioinformatics, tracking.
- **Kalman filter:** an HMM with linear-Gaussian dynamics; closed-form tracking. **Particle filter:** sampling-based for non-linear cases.

## Inference

| Method | Exact? | Notes |
|---|---|---|
| Variable elimination | Yes | Cost depends on elimination order |
| Belief propagation | Exact on trees | Loopy BP: approximate on graphs with cycles |
| Junction tree | Yes | Exponential in treewidth |
| MCMC (Gibbs, Metropolis-Hastings) | Asymptotically | General, slow |
| Variational inference | No | Fast, optimization-based |

## Topic models

- **LDA (latent Dirichlet allocation):** documents are mixtures of topics; topics are distributions over words.
- Learned with Gibbs sampling or variational inference. Largely replaced by embedding-based clustering for many uses.

## Probabilistic programming

- Write the generative model as code; the system does inference (Stan, PyMC, NumPyro, Pyro).

## Causality (briefly)

- Correlation in observational data isn't causation; confounders create spurious links.
- Tools: randomized experiments, causal graphs (do-calculus), instrumental variables, difference-in-differences, propensity scores.
