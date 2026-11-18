---
layout: ../../layouts/CheatsheetLayout.astro
title: "SVMs & Bayesian Methods"
description: "Margins, kernels, priors, posteriors and Gaussian processes."
---

## Support vector machines

- Find the separating hyperplane with the **largest margin** to the nearest points (support vectors).
- Hard margin: requires separable data. Soft margin: allows violations with slack, penalty $C$.
- Primal: $\min \frac{1}{2}\lVert w\rVert^2 + C\sum \xi_i$ subject to $y_i(w^\top x_i + b) \ge 1 - \xi_i$.
- Equivalent to hinge loss $\max(0, 1 - y f(x))$ plus L2 regularization.
- Only support vectors determine the boundary.

| Hyperparameter | Large value | Small value |
|---|---|---|
| $C$ | Fewer violations, narrower margin, overfit risk | Wider margin, more violations, smoother |
| RBF $\gamma$ | Very local influence, wiggly boundary | Smooth, near-linear boundary |

## Kernels

- **Kernel trick:** replace dot products with $k(x, x')$ to work in a high-dimensional feature space without computing it.
- Linear: $x^\top x'$. Polynomial: $(x^\top x' + c)^d$. RBF: $\exp(-\gamma\lVert x - x'\rVert^2)$.
- Valid kernel: the kernel matrix is positive semi-definite (Mercer).
- Cost: $O(n^2)$ memory, $O(n^2)$–$O(n^3)$ time; impractical past ~100k examples. Use linear SVMs or approximations (random Fourier features) for big data.

## SVM practical notes

- Always scale features.
- Linear kernel for many features (text); RBF as the default otherwise; grid-search $C$ and $\gamma$.
- No probabilities by default; use Platt scaling.
- Multi-class: one-vs-rest or one-vs-one.
- SVR: regression with an $\epsilon$-insensitive tube.

## Bayesian basics

- Posterior ∝ likelihood × prior: $p(\theta \mid D) \propto p(D \mid \theta)\,p(\theta)$.
- **Predictive distribution:** average predictions over the posterior, $p(y \mid x, D) = \int p(y \mid x,\theta)\,p(\theta \mid D)\,d\theta$.
- Benefits: uncertainty estimates, principled regularization, works with little data.
- Cost: integrals are usually intractable.

## Conjugate priors

| Likelihood | Conjugate prior | Posterior |
|---|---|---|
| Bernoulli / binomial | Beta | Beta |
| Categorical / multinomial | Dirichlet | Dirichlet |
| Gaussian (known variance) | Gaussian | Gaussian |
| Poisson | Gamma | Gamma |

## Approximate inference

- **MCMC** (Metropolis-Hastings, Gibbs, Hamiltonian Monte Carlo / NUTS): asymptotically exact samples, slow.
- **Variational inference:** fit a simple distribution by maximizing the ELBO; fast, biased (often underestimates variance).
- **Laplace approximation:** Gaussian around the MAP estimate.

## Bayesian linear regression

- Gaussian prior on weights + Gaussian noise → Gaussian posterior in closed form.
- Predictions come with variance that grows away from the data.

## Gaussian processes

- A distribution over functions: any finite set of points is jointly Gaussian, defined by a mean and a kernel.
- Gives predictions with calibrated uncertainty; great for small data and Bayesian optimization.
- Cost $O(n^3)$; use sparse / inducing-point approximations for larger data.
- The kernel encodes assumptions: smoothness (RBF, Matérn), periodicity, linear trends.
