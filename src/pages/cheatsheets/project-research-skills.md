---
layout: ../../layouts/CheatsheetLayout.astro
title: "Project & Research Skills"
description: "Problem framing, baselines, experiment design, model selection, debugging and communicating results."
---

## Frame the problem first

- What decision will the prediction change? Who uses it, and how?
- What does success look like in business or user terms? Which ML metric tracks it?
- What's the cost of each error type (false positive vs false negative)?
- Is ML needed? Could a rule, a heuristic or a lookup do it?
- What data exists, what labels, and what's available at prediction time?

## Baselines

- Always build: trivial baseline (majority class, mean, last value), simple heuristic, simple model (linear / logistic / small tree ensemble).
- A complex model that barely beats a simple one usually isn't worth its cost.

## Experiment design

- Change one thing at a time; keep a log of every run (config, data version, code commit, result).
- Fix random seeds, but report variance across several seeds.
- Use the same splits for every model you compare.
- Decide the evaluation metric and protocol **before** looking at results.
- Ablations: remove components one at a time to see what actually matters.

## Error analysis

- Look at the worst predictions by hand; group them into categories.
- Slice metrics by segment (class, region, device, time).
- Ask: data problem (labels, coverage), model problem (capacity, features), or task problem (ambiguous)?
- Fix the biggest category first.

## Debugging a model

| Symptom | Check |
|---|---|
| Too good to be true | Data leakage, test set contamination, duplicates |
| Can't beat baseline | Label quality, features, bugs in preprocessing, wrong metric |
| Training doesn't converge | Learning rate, normalization, overfit one batch |
| Great offline, bad online | Training-serving skew, distribution shift, feedback loops |
| Unstable results | Seeds, small validation set, data order |

## Model selection

- Pick on validation; report on test once.
- Consider latency, memory, cost, interpretability and maintenance, not just accuracy.
- Statistical significance: confidence intervals or paired tests across folds or seeds.

## Reading papers fast

1. Abstract and figures: what problem, what claim?
2. Method overview and main results table.
3. Ablations and limitations: what actually matters, what's missing?
4. Check: baselines fair? Same compute? Code available? Evaluated where you care?

## Research mindset

- Start from a clear question or hypothesis.
- Small, fast experiments before big ones.
- Try to break your own result.
- Negative results are information; write them down.

## Communicating results

- Lead with the conclusion and the decision it supports.
- Show uncertainty (intervals, variance across runs).
- Compare against baselines and the current system.
- State limitations and failure cases plainly.
- One clear figure beats five tables.
