---
layout: ../../layouts/CheatsheetLayout.astro
title: "Post-Training & Alignment in 2 Minutes Each"
description: "Intuition cards for fine-tuning, SFT, LoRA, QLoRA, RLHF, reward models, DPO, GRPO, distillation, LLM-as-a-judge, prompt and context engineering."
---

How a base model that only predicts text becomes an assistant that follows instructions, and how models are adapted cheaply.

## Base model vs assistant

**In one line:** a pretrained model completes text; post-training teaches it to respond helpfully.

**Picture it:** ask a base model "What's the capital of France?" and it may continue with "What's the capital of Germany?" because that's how quiz lists look. An assistant answers "Paris."

**Remember:** pretraining = knowledge, post-training = behavior.

**Go deeper:** [Post-Training](/posts/mlops/modern-post-training-peft-2026).

## Fine-tuning

**In one line:** keep training a pretrained model on your smaller, specific dataset.

**Picture it:** a general doctor doing a residency in cardiology. Most knowledge stays; a specialty is added.

**Remember:** fine-tune for format, style and narrow skills. For new or changing facts, retrieval (RAG) is usually better.

**Watch out:** catastrophic forgetting: narrow fine-tuning can erase general abilities. Use low learning rates and mix in general data.

## Supervised fine-tuning (SFT)

**In one line:** train on example prompts paired with ideal responses.

**How it works:** standard next-token loss, but only on the response tokens, using the model's chat template (special tokens marking user and assistant turns).

**Remember:** quality beats quantity. A few thousand excellent examples can matter more than a million mediocre ones.

**Watch out:** a mismatched chat template between training and serving is one of the most common fine-tuning bugs.

## LoRA

**In one line:** freeze the model and learn a small low-rank correction for each weight matrix.

**Picture it:** instead of repainting a whole mural, add a thin transparent overlay. Peel it off and the original is intact; swap overlays for different tasks.

**How it works:** the update $\Delta W$ is written as two thin matrices $B \times A$ of rank $r$ (e.g. 16), so a 4096×4096 layer trains ~131k numbers instead of ~16.8M. $B$ starts at zero, so training starts exactly from the base model.

**Remember:** tiny adapters, one per task, cheap to train, store and swap.

## QLoRA

**In one line:** LoRA on top of a base model compressed to 4 bits.

**Remember:** that's how a 65–70B model fits on a single 80 GB GPU for fine-tuning. The frozen base is 4-bit; the adapters train in higher precision.

## RLHF

**In one line:** learn from human preferences rather than from ideal answers.

**Picture it:** it's easier for people to say "answer A is better than B" than to write the perfect answer.

**How it works:** collect pairs of responses ranked by people; train a **reward model** to predict which is preferred; then optimize the language model with reinforcement learning (PPO) to get high reward, with a penalty for drifting too far from the starting model.

**Remember:** RLHF is what made ChatGPT-style assistants feel helpful. It's powerful and complex: several models in memory, unstable training.

## Reward models and reward hacking

**In one line:** a model that scores responses, standing in for human judgment.

**Watch out:** **reward hacking**: the policy finds ways to score high without being better, such as longer answers, flattery or confident tone. Keep a leash (KL penalty) to the original model and check samples by hand.

## DPO

**In one line:** learn directly from preferred-vs-rejected pairs, with no separate reward model or RL loop.

**Picture it:** "make answers like this more likely and answers like that less likely," relative to where the model started.

**How it works:** a simple classification-style loss on preference pairs; a parameter β controls how far the model may move from the reference model.

**Remember:** simpler and more stable than RLHF, so it became a default for preference tuning.

## GRPO and verifiable rewards

**In one line:** sample several answers to the same question, reward the ones that are checkably correct, and push toward better-than-average answers.

**Picture it:** a teacher grading eight attempts at one math problem and telling the student "do more of what the right ones did."

**How it works:** the group's average reward is the baseline, so no separate critic network is needed. Works best when correctness can be checked automatically (math answers, unit tests).

**Remember:** this is how reasoning models are trained. Questions the model always or never gets right teach nothing.

## Knowledge distillation

**In one line:** train a small student model to imitate a large teacher.

**Picture it:** an apprentice copying a master's work, including the master's uncertainty ("this is probably a cat, maybe a fox").

**Remember:** the teacher's soft probabilities carry more information than hard labels. Distilled models are how capability gets cheap enough to deploy.

## LLM-as-a-judge

**In one line:** use a strong model to grade outputs instead of (or before) humans.

**Remember:** scalable and surprisingly useful; biased toward longer answers, its own style and the first option shown. Calibrate against human ratings, randomize order, use clear rubrics.

## Prompt engineering

**In one line:** writing instructions and examples so the model does what you want.

**What works:** clear task and role, relevant context, examples of good output, explicit format, asking the model to reason step by step for hard problems, and breaking big tasks into steps.

**Remember:** treat prompts like code: version them and test them against a set of cases.

## Context engineering

**In one line:** deciding what information goes into the model's limited context window at each step.

**Remember:** "the smallest set of high-signal tokens." Retrieve just in time, summarize old history, keep notes outside the window, use sub-agents with fresh context.

**Go deeper:** [Agentic AI, Part 2](/posts/agentic-ai/agentic-ai-design-patterns-part-2#context-engineering-the-scarce-resource).

## Which tool when

| Goal | Try |
|---|---|
| Model doesn't know your facts | RAG |
| Wrong format, tone or style | Prompting, then SFT / LoRA |
| Needs a narrow skill | LoRA on good examples |
| Answers are correct but not preferred | DPO on preference pairs |
| Needs to reason on checkable tasks | RL with verifiable rewards (GRPO) |
| Too slow or expensive | Distill into a smaller model |
