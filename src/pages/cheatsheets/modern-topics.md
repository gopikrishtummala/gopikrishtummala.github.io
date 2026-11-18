---
layout: ../../layouts/CheatsheetLayout.astro
title: "Modern Topics"
description: "Self-supervised, transfer, meta, federated and continual learning, plus reinforcement learning basics."
---

## Self-supervised learning

- Create labels from the data itself, learn general representations, then fine-tune.

| Approach | Pretext task | Examples |
|---|---|---|
| Masked modeling | Predict hidden parts | BERT, MAE |
| Next-token prediction | Predict the next token | GPT-style LLMs |
| Contrastive | Match augmented views or paired modalities | SimCLR, MoCo, CLIP |
| Self-distillation | Student matches teacher across views | BYOL, DINO |

## Transfer learning

- Reuse a pretrained model. Options, cheapest first: linear probe (frozen features) → fine-tune head → partial fine-tune → full fine-tune.
- Parameter-efficient fine-tuning: LoRA, adapters, prefix tuning. See [Post-Training](/posts/mlops/modern-post-training-peft-2026).
- **Domain shift:** pretrained data differs from target; fine-tune or adapt.
- **Catastrophic forgetting:** fine-tuning erases old skills; use lower learning rates, replay, or LoRA.

## Few-shot, zero-shot and in-context learning

- Zero-shot: no examples (e.g. CLIP matching images to class names).
- Few-shot: a handful of labeled examples.
- In-context learning: examples in the prompt; no weight updates (LLMs).

## Meta-learning ("learning to learn")

- Train across many tasks so a new task can be learned from few examples.
- Metric-based (prototypical networks), optimization-based (MAML: learn an initialization that fine-tunes fast), model-based.

## Federated learning

- Train across devices or institutions without centralizing data; share model updates, not data.
- **FedAvg:** clients train locally, server averages weights.
- Challenges: non-IID data across clients, communication cost, stragglers, privacy leakage through updates (add secure aggregation, differential privacy).

## Continual / lifelong learning

- Learn a sequence of tasks without forgetting earlier ones.
- Methods: replay buffers, regularization (EWC penalizes changing important weights), separate parameters per task.

## Reinforcement learning basics

- Agent observes state $s$, takes action $a$, gets reward $r$, maximizes discounted return $\sum_t \gamma^t r_t$.
- **Value function** $V(s)$, **action-value** $Q(s,a)$. **Policy** $\pi(a \mid s)$.
- Bellman equation: $Q(s,a) = r + \gamma \max_{a'} Q(s', a')$.

| Family | Examples | Notes |
|---|---|---|
| Value-based | Q-learning, DQN | Discrete actions; off-policy |
| Policy gradient | REINFORCE, PPO | Directly optimize the policy; on-policy |
| Actor-critic | A2C, SAC, TD3 | Policy + learned value baseline |
| Model-based | Dyna, MuZero, Dreamer | Learn a world model, plan with it |
| Offline RL | CQL, IQL | Learn from logged data only |

- Exploration vs exploitation: ε-greedy, entropy bonuses, optimism.
- RL for LLMs: RLHF (reward model + PPO), DPO (preferences, no RL loop), GRPO (group-relative rewards, verifiable tasks). See [RL intuition](/posts/fundamentals/reinforcement-learning-intuition-to-algorithms).

## Other topics worth knowing

- **Active learning:** pick the most informative examples to label next (uncertainty, diversity).
- **Weak supervision:** noisy labeling functions combined into training labels.
- **Knowledge distillation:** train a small student to match a large teacher's outputs.
- **Multimodal learning:** joint models of text, images, audio (CLIP, VLMs).
- **Scaling laws:** loss falls predictably as a power law with model size, data and compute; compute-optimal training balances parameters and tokens (Chinchilla).
