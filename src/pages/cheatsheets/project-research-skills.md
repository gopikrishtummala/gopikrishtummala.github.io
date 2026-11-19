---
layout: ../../layouts/CheatsheetLayout.astro
title: "Project & Research Skills in 2 Minutes Each"
description: "Intuition cards for problem framing, choosing metrics, baselines, experiment design, ablations, error analysis, debugging models, reading papers and communicating results."
---

The habits that separate useful ML work from busy ML work.

## Frame the problem

**In one line:** start from the decision the model will change, not from the model.

**Remember:** ask who uses the prediction, what they'll do differently, what each kind of error costs, and whether a simple rule would do.

## Pick the metric before the model

**In one line:** decide how success is measured before you see results.

**Remember:** the offline metric should track the real goal; if it doesn't, optimizing it is wasted effort.

## Baselines

**In one line:** always beat something simple first.

**Remember:** majority class or average; a simple rule; a logistic regression or boosted trees. If a complex model barely beats them, it probably isn't worth it.

## Look at the data

**In one line:** spend real time looking at raw examples before modeling.

**Remember:** most surprises (bad labels, duplicates, leakage, weird formats) are visible by eye.

## Experiment hygiene

**Remember:** change one thing at a time; log every run's config, data version, code and result; use the same splits for every comparison; run several seeds.

## Ablations

**In one line:** remove one component at a time to see what actually helps.

**Remember:** the fastest way to find out that half your clever ideas don't matter.

## Error analysis

**In one line:** read the model's worst mistakes and group them.

**Remember:** fix the biggest group first; slice metrics by segment to find where it fails.

## Debugging a model

| Symptom | Look for |
|---|---|
| Too good to be true | Leakage, duplicates across splits |
| Can't beat the baseline | Label noise, preprocessing bugs, wrong metric |
| Training won't converge | Learning rate, normalization; try overfitting one batch |
| Great offline, poor online | Training-serving skew, distribution shift |
| Results jump between runs | Seeds, tiny validation set |

## Reading papers quickly

**Remember:** abstract and figures → method overview → main results → ablations and limitations. Then ask: fair baselines? Same compute? Tested where you care?

## Research mindset

**Remember:** start from a question; run small, fast experiments; try to break your own result; write down negative results.

## Communicating results

**In one line:** lead with the conclusion and the decision it supports.

**Remember:** compare against the baseline and current system, show uncertainty, state limitations plainly, and prefer one clear chart to five tables.

## Working with AI coding tools

**Remember:** understand what the code does and how it fits before accepting it; keep diffs small; write the tests yourself.

**Go deeper:** [Coding With AI Without Losing Your Edge](/posts/interview-prep/ai-assisted-coding-conscious-practice-and-interviews).
