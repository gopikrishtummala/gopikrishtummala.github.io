---
layout: ../../layouts/CheatsheetLayout.astro
title: "Deep Learning in 2 Minutes Each"
description: "Intuition cards for neurons and layers, activations, losses, initialization, batch and layer norm, residual connections, dropout, CNNs, receptive fields, RNNs and LSTMs, embeddings, transfer learning and graph neural networks."
---

The building blocks of neural networks.

## Activation functions

| Activation | Remember it as |
|---|---|
| ReLU | Zero for negatives, identity for positives; the simple default |
| GELU / SiLU | Smooth ReLU variants; default in transformers |
| Sigmoid | Squashes to 0–1; for binary outputs, not hidden layers |
| Tanh | Squashes to −1 to 1; used in older RNNs |
| Softmax | Turns scores into class probabilities |

**Watch out:** sigmoid and tanh saturate, killing gradients in deep nets; ReLU units can "die" and output zero forever.

## Initialization

Start weights at a scale that keeps signals from shrinking or exploding through layers.

**Remember:** He initialization for ReLU, Xavier for tanh. Bad initialization is a classic reason a network won't train.

## Batch norm and layer norm

Rescale activations to a stable range inside the network.

**Remember:** batch norm normalizes across the batch (great for CNNs, depends on batch size); layer norm across features of each example (transformers, RNNs). Both make training faster and more stable.

## Residual (skip) connections

Add a layer's input to its output, so each layer learns a correction.

**Picture it:** an express lane that lets information and gradients bypass layers.

**Remember:** the reason networks with hundreds of layers train at all (ResNet, transformers).

## Dropout

Randomly switch off neurons during training.

**Picture it:** a team where anyone might be absent, so nobody can rely on a single colleague.

**Remember:** reduces overfitting; turned off at inference. Less common in very large pretrained models.

## Convolutional networks (CNNs)

Slide small learned filters across an image to detect patterns anywhere.

**Picture it:** a flashlight scanning the image with the same "edge detector" in every position.

**Remember:** weight sharing and locality make CNNs data-efficient for images; pooling and stride shrink the image as features get more abstract.

## Receptive field

How much of the input one neuron can "see."

**Remember:** grows with depth; deep layers see large regions, which is how CNNs recognize whole objects.

## RNNs and LSTMs

Process a sequence one step at a time, carrying a memory forward.

**Remember:** LSTMs and GRUs add gates that decide what to remember and forget, fixing plain RNNs' short memory. Largely replaced by transformers, which process sequences in parallel.

## Embeddings

Learned vectors representing discrete things (words, users, products).

**Remember:** similar things end up close together; the representation is learned for the task.

## Transfer learning

Start from a model pretrained on lots of data and adapt it.

**Remember:** almost always better than training from scratch for images, text and audio. Freeze early layers for small data; fine-tune more for larger data.

## Graph neural networks

Each node updates itself by gathering messages from its neighbors.

**Picture it:** gossip spreading through a social network, one hop per layer.

**Remember:** for molecules, social networks, road networks and recommendation graphs.

## Transformers

Layers of attention, where every token looks at every other token.

**Go deeper:** [Transformers & LLMs cheatsheet](/cheatsheets/transformers-llms).

## Training recipe

- Overfit a single batch first to confirm the setup works.
- Use a proven architecture, AdamW, warmup and cosine decay.
- Watch train and validation loss together.
- Regularize with data augmentation before anything fancier.
