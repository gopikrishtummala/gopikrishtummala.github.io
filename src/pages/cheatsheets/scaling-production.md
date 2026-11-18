---
layout: ../../layouts/CheatsheetLayout.astro
title: "Scaling & Production ML"
description: "Distributed training, memory math, hyperparameter tuning, serving, monitoring and MLOps."
---

## Training memory (mixed precision + Adam)

- About **16 bytes per parameter** before activations: weights 2 + gradients 2 + FP32 master weights 4 + Adam moments 8.
- 7B parameters ≈ 112 GB of state. Activations add more, growing with batch size and sequence length.
- Cut memory: activation checkpointing, sharding (ZeRO / FSDP), 8-bit optimizers, LoRA, smaller batches with gradient accumulation.

## Parallelism

| Type | Splits | Communication | Use when |
|---|---|---|---|
| Data parallel (DDP) | Data; full model per GPU | All-reduce gradients | Model fits on one GPU |
| ZeRO-1/2/3, FSDP | Optimizer state, gradients, weights | Gather / scatter | Model state doesn't fit |
| Tensor parallel | Each layer's matrices | Every layer; needs NVLink | Very large layers, inside a node |
| Pipeline parallel | Layers across devices | Activations between stages | Very deep models, across nodes |
| Expert parallel | MoE experts | All-to-all | Mixture-of-experts models |
| Context / sequence parallel | Long sequences | Attention communication | Very long contexts |

See [Training Frameworks](/posts/mlops/hidden-engine-of-ai).

## Data pipeline

- GPU utilization is often limited by data loading: decode, augment and prefetch in parallel workers.
- Columnar formats (Parquet), sharded files, streaming datasets. See [Datasets and Dataloaders](/posts/mlops/datasets-and-dataloaders).

## Hyperparameter tuning

| Method | Notes |
|---|---|
| Grid search | Exhaustive, wasteful in many dimensions |
| Random search | Better than grid for the same budget |
| Bayesian optimization | Models the objective (GP, TPE / Optuna); sample-efficient |
| Successive halving / Hyperband / ASHA | Stop bad trials early |
| Population-based training | Evolves hyperparameters during training |

- Tune learning rate first. Search on a log scale. Use small proxy runs, then confirm at full scale.

## Serving

- **Batch vs online:** precompute predictions vs respond per request.
- Latency levers: smaller or distilled models, quantization (INT8, FP8, INT4), batching, caching, compiled runtimes (TensorRT, ONNX Runtime).
- LLM serving: KV cache memory limits concurrency; continuous batching and paged attention (vLLM); speculative decoding for latency. See [LLM Serving](/posts/mlops/vllm-trilogy-of-modern-llm-scaling).
- Deploy safely: shadow mode, canary, A/B tests, quick rollback.

## Monitoring

- **Data drift:** input distribution changes (PSI, KS test).
- **Concept drift:** relationship between inputs and labels changes.
- **Prediction drift:** output distribution changes.
- Track latency (P50 / P99), errors, throughput, cost, and business metrics; alert on thresholds.
- Training-serving skew: features computed differently offline and online; fix with a feature store or shared code.

## MLOps practices

- Version code, data and models; track experiments (MLflow, Weights & Biases).
- Reproducible pipelines; orchestrators for schedules and lineage. See [Pipeline Orchestration](/posts/mlops/ml-pipeline-orchestration-layers).
- Model registry with stages; automated evaluation gates before promotion.
- Retrain on a schedule or on drift triggers.

## AutoML

- Automates feature engineering, model selection and tuning (AutoGluon, H2O, auto-sklearn).
- Strong baseline for tabular data; still needs good problem framing and data.

## Cost rules of thumb

- Training compute for transformers ≈ $6 \times$ parameters × tokens FLOPs.
- Inference is often the larger long-term cost; optimize the serving path.
