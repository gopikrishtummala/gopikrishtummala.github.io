---
layout: ../../layouts/CheatsheetLayout.astro
title: "Deep Learning"
description: "Layers, activations, initialization, normalization, regularization, CNNs, RNNs and transformers."
---

## Building blocks

- Layer: $h = \phi(Wx + b)$. Stack layers; train with backpropagation (chain rule) and an optimizer.
- Universal approximation: one wide hidden layer can approximate any continuous function; depth makes it efficient.

## Activations

| Activation | Formula | Notes |
|---|---|---|
| Sigmoid | $1/(1+e^{-x})$ | Output probabilities; saturates, vanishing gradients in deep nets |
| Tanh | $\tanh(x)$ | Zero-centered; still saturates |
| ReLU | $\max(0, x)$ | Default for CNNs; can "die" (always zero) |
| Leaky ReLU / PReLU | small slope for $x<0$ | Avoids dead units |
| GELU | $x\,\Phi(x)$ | Default in transformers (BERT, GPT) |
| SiLU / Swish | $x\,\sigma(x)$ | Used in many modern nets; SwiGLU in LLM feed-forward layers |
| Softmax | $e^{z_i}/\sum_j e^{z_j}$ | Multi-class output |

## Losses

- Classification: cross-entropy (with softmax), binary cross-entropy (with sigmoid), focal loss for imbalance.
- Regression: MSE, MAE, Huber.
- Embeddings: contrastive, triplet, InfoNCE.

## Initialization

- **Xavier / Glorot:** variance $2/(n_{in}+n_{out})$; for tanh / sigmoid.
- **He / Kaiming:** variance $2/n_{in}$; for ReLU.
- Goal: keep activation and gradient scale roughly constant across layers.
- Residual networks often zero-initialize the last layer of each block.

## Normalization

| | Normalizes over | Used in |
|---|---|---|
| Batch norm | Batch, per channel | CNNs; depends on batch size |
| Layer norm | Features, per example | Transformers, RNNs |
| RMS norm | Features, no mean subtraction | Modern LLMs |
| Group norm | Groups of channels | Small-batch vision, diffusion models |

- Pre-norm (norm before attention / MLP) trains more stably than post-norm for deep transformers.

## Regularization

- Weight decay, dropout (0.1–0.5), data augmentation, early stopping, label smoothing (0.1), stochastic depth, mixup / cutmix.
- More data and pretraining beat all of them.

## Architectures

| Family | Key idea | Good for |
|---|---|---|
| MLP | Fully connected layers | Tabular, small inputs, heads |
| CNN | Local filters, weight sharing, pooling | Images, audio spectrograms |
| ResNet | Skip connections $h + F(h)$ | Very deep nets train stably |
| RNN / LSTM / GRU | Recurrent state; gates control memory | Sequences (largely replaced by transformers) |
| Transformer | Self-attention over all tokens | Text, images (ViT), audio, multimodal |
| U-Net | Encoder-decoder with skip connections | Segmentation, diffusion models |
| Autoencoder / VAE | Compress and reconstruct | Representation learning, generation |

## Attention

- $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$.
- Multi-head: several attention patterns in parallel.
- Cost is quadratic in sequence length; mitigations: FlashAttention (exact, memory-efficient), sparse / linear attention, sliding windows.
- Positional information: sinusoidal, learned, or rotary (RoPE, standard in LLMs).
- See [The Transformer](/posts/fundamentals/transformers-attention-mechanism).

## Representation learning and transfer

- Pretrain on large data (supervised, self-supervised), then fine-tune or use frozen features.
- Fine-tuning options: full, last layers only, adapters / LoRA (see [Post-Training](/posts/mlops/modern-post-training-peft-2026)).
- Embeddings: dense vectors where similar inputs are close.

## Training checklist

- Overfit one batch first. Then scale up.
- Normalize inputs. Use a known-good architecture and optimizer (AdamW, cosine schedule, warmup).
- Monitor train and validation loss, gradient norm, learning rate.
- Use mixed precision and gradient checkpointing for memory.
