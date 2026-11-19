---
layout: ../../layouts/CheatsheetLayout.astro
title: "Scaling & Production ML in 2 Minutes Each"
description: "Intuition cards for training memory, data/tensor/pipeline/expert parallelism, FSDP, checkpointing, data loading, quantization, distillation, serving, batching, KV cache, speculative decoding, hyperparameter tuning, monitoring and MLOps."
---

Getting models trained at scale and served reliably.

## Where training memory goes

**In one line:** weights are the smallest part; gradients, optimizer state and activations dominate.

**Remember:** about 16 bytes per parameter with mixed precision and Adam, before activations. A 7B model needs ~112 GB of training state.

**Go deeper:** [Training Frameworks](/posts/mlops/hidden-engine-of-ai).

## Data parallelism

**In one line:** copy the model to every GPU, give each a different slice of data, average the gradients.

**Remember:** speeds up training but doesn't save memory per GPU.

## ZeRO / FSDP

**In one line:** instead of copying everything, split weights, gradients and optimizer state across GPUs.

**Picture it:** a shared library where each person keeps a few volumes and lends them out when needed, instead of everyone owning every book.

**Remember:** what lets big models fit; costs extra communication.

## Tensor and pipeline parallelism

**In one line:** tensor parallel splits each layer across GPUs; pipeline parallel puts different layers on different GPUs.

**Remember:** tensor parallel needs very fast links (inside a server); pipeline parallel works across servers but leaves GPUs idle at the start and end ("bubble"). Big runs combine data, tensor and pipeline: "3D parallelism."

## Activation checkpointing

**In one line:** throw away intermediate results and recompute them during the backward pass.

**Remember:** trade about a third more compute for much less memory.

## Data loading

**In one line:** the GPU is often waiting for data, not computing.

**Remember:** decode and augment in parallel workers, prefetch, use columnar and sharded formats. Check GPU utilization first.

**Go deeper:** [Datasets and Dataloaders](/posts/mlops/datasets-and-dataloaders).

## Quantization

**In one line:** store and compute with fewer bits (8 or 4 instead of 16).

**Remember:** 8-bit is usually near-lossless; 4-bit loses a little. It cuts memory and speeds memory-bound inference.

## Distillation and pruning

**In one line:** train a smaller student, or remove unimportant weights.

**Remember:** the main ways to make a model cheap enough to serve widely.

## Batch vs online serving

**In one line:** precompute predictions on a schedule, or compute each on request.

**Remember:** batch is cheap and simple; online is needed for fresh, per-request predictions with tight latency.

## LLM serving essentials

**Remember:**
- The KV cache limits how many conversations fit on a GPU.
- Continuous batching keeps the GPU busy as requests come and go.
- Paged attention (vLLM) allocates KV cache memory in small blocks to waste almost none.
- Speculative decoding: a fast draft guesses several tokens, the big model checks them at once.

**Go deeper:** [LLM Serving](/posts/mlops/vllm-trilogy-of-modern-llm-scaling), [Speculative Decoding](/posts/mlops/speculative-decoding-guess-and-verify).

## Hyperparameter tuning

**In one line:** search smartly and stop bad runs early.

**Remember:** random search beats grid search; Bayesian optimization is sample-efficient; Hyperband / ASHA stop weak trials early. Tune the learning rate first.

## Safe deployment

**Remember:** shadow mode (run alongside without affecting users), canary (a small slice of traffic), A/B test, fast rollback.

## Monitoring

**In one line:** models degrade silently as the world changes.

**Remember:** watch input distributions (data drift), prediction distributions, latency, errors, cost, and real outcomes when labels arrive.

## Training-serving skew

**In one line:** features computed differently offline and online.

**Remember:** a top cause of "great offline, bad in production." Share feature code or use a feature store.

## MLOps basics

**Remember:** version code, data and models; track experiments; automate pipelines with evaluation gates; keep lineage of which data trained which model.

**Go deeper:** [The MLOps map](/posts/mlops/infrastructure-first-mlops-roadmap).
