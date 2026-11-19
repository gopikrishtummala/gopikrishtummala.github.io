---
layout: ../../layouts/CheatsheetLayout.astro
title: "Optimization in 2 Minutes Each"
description: "Intuition cards for gradient descent, backpropagation, SGD and mini-batches, momentum, Adam and AdamW, learning rates, warmup and schedules, batch size, vanishing gradients, clipping, mixed precision and debugging training."
---

How models actually learn their weights.

## Backpropagation

Compute how much each weight contributed to the error by passing blame backward through the network with the chain rule.

**Picture it:** a manager traces a mistake back through the team, giving each person a share of the blame proportional to their part.

**Remember:** one backward pass costs about the same as one forward pass and gives every weight's gradient.

**Go deeper:** [Backpropagation](/posts/fundamentals/backpropagation-mlp-math-derivation).

## SGD and mini-batches

Estimate the gradient from a small random batch instead of all the data.

**Remember:** noisy but cheap, and the noise helps escape bad spots. Mini-batches of tens to thousands are standard.

## Momentum

Keep some of the previous step's direction.

**Picture it:** a ball rolling downhill builds speed through flat stretches and doesn't zig-zag in narrow valleys.

## Adam and AdamW

Momentum plus a separate, adaptive step size for each weight.

**Remember:** Adam is the default for deep learning; AdamW fixes how weight decay is applied and is the default for transformers. Costs two extra numbers of memory per weight.

## Learning rate

The most important hyperparameter.

**Remember:** if training diverges, lower it; if it's painfully slow, raise it. A quick learning-rate range test finds a good region.

## Warmup and schedules

Start the learning rate small, ramp up, then decay.

**Picture it:** don't sprint before warming up; slow down as you approach the finish.

**Remember:** warmup prevents early instability (essential for transformers); cosine or linear decay helps settle into a good minimum.

## Batch size

How many examples per update.

**Remember:** bigger batches = smoother gradients and better hardware use, but need higher learning rates and warmup, and can generalize slightly worse. Use gradient accumulation to simulate big batches on small GPUs.

## Vanishing and exploding gradients

In deep networks, gradients can shrink to nothing or blow up as they pass back through many layers.

**Remember:** fixed by ReLU-family activations, careful initialization, normalization layers and residual connections.

## Gradient clipping

Cap the gradient's size before each step.

**Remember:** a cheap insurance policy against sudden loss spikes; standard for transformers and RNNs.

## Weight decay and early stopping

Shrink weights a little each step; stop when validation stops improving.

**Remember:** both are regularization through the optimizer.

## Mixed precision

Do most math in 16-bit numbers, keep a 32-bit master copy of the weights.

**Remember:** roughly halves memory and speeds training. BF16 is easier than FP16 (no loss scaling needed).

## Convex vs non-convex

Convex losses have one valley; neural networks have many.

**Remember:** linear and logistic regression are convex, so any minimum is the best. Neural nets aren't, yet SGD finds good solutions in practice; saddle points are a bigger obstacle than bad local minima.

## Debugging training

| Symptom | First suspect |
|---|---|
| Loss becomes NaN | Learning rate too high, FP16 overflow, bad data |
| Loss doesn't move | Learning rate too low, bug in labels or loss, dead activations |
| Can't overfit one small batch | Bug in model or data pipeline |
| Train improves, validation worsens | Overfitting |
| Periodic spikes | Bad batches, no warmup or clipping |
