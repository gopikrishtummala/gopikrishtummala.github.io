---
layout: ../../layouts/CheatsheetLayout.astro
title: "Optimization for ML"
description: "Gradient descent variants, Adam and friends, learning-rate schedules."
---

## Gradient descent

- Update: $\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$.
- **Batch GD:** full dataset per step; exact gradient, slow.
- **Stochastic GD (SGD):** one example; noisy, cheap.
- **Mini-batch:** 32–4096 examples; the standard. Noise can help escape sharp minima.
- **Convex** problems (linear / logistic regression): any local minimum is global. Neural nets: non-convex; saddle points are the usual obstacle.

## Optimizers

| Optimizer | Idea | Notes |
|---|---|---|
| SGD + momentum | Accumulate velocity: $v \leftarrow \beta v + \nabla L$ | Strong for vision with tuned schedule |
| Nesterov | Look-ahead momentum | Slightly better convergence |
| AdaGrad | Per-parameter step shrinks with past squared gradients | Good for sparse features; step decays too fast |
| RMSProp | Exponential moving average of squared gradients | Fixes AdaGrad's decay |
| Adam | Momentum + RMSProp + bias correction | Default for most deep learning |
| AdamW | Adam with decoupled weight decay | Default for transformers |
| Lion, Adafactor, Shampoo, Muon | Memory-saving or second-order-flavored | Used at large scale |

- Adam defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$ (often 0.95 for large models), $\epsilon = 10^{-8}$.
- Adam stores two extra values per parameter: memory cost matters at scale.

## Learning rate

- The most important hyperparameter. Too high: divergence or loss spikes. Too low: slow, stuck.
- **LR range test:** increase LR over a short run; pick where loss falls fastest.
- Scaling rule of thumb: larger batches tolerate larger learning rates (linear for SGD, ~square-root for Adam), with warmup.

## Schedules

| Schedule | Use |
|---|---|
| Warmup (linear, first ~1–5% of steps) | Stabilizes early training, essential for transformers and large batches |
| Step decay | Classic CNN training |
| Cosine decay | Common default |
| Linear decay | Common for LLM fine-tuning |
| One-cycle | Fast training with SGD |
| Warmup-stable-decay | Constant LR then quick decay; popular for LLM pre-training |

## Stability tricks

- **Gradient clipping** by global norm (e.g. 1.0): prevents spikes, standard for RNNs and transformers.
- **Normalization layers** (batch, layer, RMS norm) make optimization easier.
- **Mixed precision:** BF16 compute with FP32 master weights; FP16 needs loss scaling.
- **EMA of weights:** smoother final model, common in vision and diffusion.

## Regularization during optimization

- **Weight decay:** shrink weights each step; with Adam, use AdamW (decoupled).
- **Early stopping:** stop when validation loss stops improving.
- **Dropout, data augmentation, label smoothing:** see Deep Learning.

## Second-order and other methods

- **Newton's method:** uses the Hessian; fast convergence, too expensive for large models.
- **L-BFGS:** approximates curvature; good for small, deterministic problems (classic ML, fine-tuning small heads).
- **Coordinate descent:** used for Lasso.

## Debugging training

| Symptom | Likely cause |
|---|---|
| Loss is NaN or explodes | LR too high, bad initialization, FP16 overflow, division by zero, bad data |
| Loss flat from the start | LR too low, dead activations, bug in labels or loss |
| Train down, validation up | Overfitting: regularize, more data, early stop |
| Loss spikes periodically | Bad batches, LR too high, missing warmup or clipping |
| Can't overfit a single batch | Bug in the model or data pipeline |
