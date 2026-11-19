---
layout: ../../layouts/CheatsheetLayout.astro
title: "Learning Paradigms in 2 Minutes Each"
description: "Intuition cards for self-supervised, transfer, few-shot, meta, multi-task, federated, continual, active and weakly supervised learning, plus knowledge distillation, multimodal learning and scaling laws."
---

Ways to learn beyond "lots of labeled examples."

## Self-supervised learning

**In one line:** make up a task from unlabeled data whose answer is in the data itself.

**Picture it:** hide words in a sentence and learn to fill them in; to do that well, you must understand language.

**Remember:** the recipe behind foundation models: pretrain self-supervised on huge data, then adapt.

## Transfer learning

**In one line:** reuse what a model learned on one task for another.

**Picture it:** a pianist learning organ is far ahead of a beginner.

**Remember:** cheapest to most expensive: use frozen features → train a new head → fine-tune some layers → fine-tune everything (or LoRA).

## Few-shot and zero-shot learning

**In one line:** do a new task from a handful of examples, or from only a description.

**Remember:** CLIP classifies images zero-shot by matching them to class names; LLMs learn tasks from examples in the prompt.

## Meta-learning

**In one line:** learn how to learn, so new tasks need very few examples.

**Picture it:** after learning ten languages, the eleventh comes fast.

**Remember:** MAML learns a starting point that fine-tunes quickly; prototypical networks classify by distance to class averages.

## Multi-task learning

**In one line:** train one model on several related tasks at once.

**Remember:** shared representations help each task, especially those with little data; conflicting tasks can hurt each other.

## Federated learning

**In one line:** train on data that stays on users' devices; only model updates are shared.

**Picture it:** many hospitals improve one model without any patient record leaving its hospital.

**Remember:** data across devices is uneven and communication is costly; updates can still leak information, so add secure aggregation or differential privacy.

## Differential privacy

**In one line:** add carefully calibrated noise so no single person's data noticeably changes the result.

**Remember:** a formal privacy guarantee, paid for with some accuracy.

## Continual learning

**In one line:** keep learning new things without forgetting old ones.

**Remember:** neural nets forget catastrophically; fixes include replaying old data, protecting important weights, or adding new parameters per task.

## Active learning

**In one line:** let the model choose which examples a human should label next.

**Remember:** label the examples the model is most unsure about or that cover new ground; saves labeling cost.

## Weak supervision

**In one line:** combine many noisy, cheap labeling rules into usable training labels.

**Remember:** faster than hand-labeling; label quality must be checked.

## Knowledge distillation

**In one line:** a small student learns to imitate a large teacher's outputs.

**Remember:** the teacher's probabilities ("90% cat, 8% fox") teach more than hard labels; the standard way to make models cheaper.

## Multimodal learning

**In one line:** models that connect text, images, audio and video in one representation.

**Remember:** CLIP aligns images and captions; vision-language models plug a vision encoder into an LLM.

**Go deeper:** [When AI Sees and Speaks](/posts/computer-vision/vision-language-models-explained).

## Scaling laws

**In one line:** performance improves predictably as model size, data and compute grow.

**Remember:** for a fixed budget, balance parameters and training tokens; bigger isn't automatically better.

## Reinforcement learning

**Go deeper:** [Reinforcement Learning cheatsheet](/cheatsheets/reinforcement-learning).
