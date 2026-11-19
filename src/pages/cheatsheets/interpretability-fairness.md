---
layout: ../../layouts/CheatsheetLayout.astro
title: "Interpretability & Fairness in 2 Minutes Each"
description: "Intuition cards for global vs local explanations, permutation importance, partial dependence, LIME, SHAP, counterfactuals, saliency maps, mechanistic interpretability, sources of bias, fairness definitions and their trade-offs."
---

Understanding why a model decides, and whether it treats people fairly.

## Global vs local explanations

**In one line:** global explains the model overall; local explains one prediction.

**Remember:** "income matters most to this model" is global; "you were declined mainly because of your debt ratio" is local.

## Interpretable by design

**In one line:** models simple enough to read directly: linear models, small trees, rule lists, additive models.

**Remember:** for high-stakes decisions, an interpretable model that's nearly as accurate often beats a black box plus an explanation.

## Permutation importance

**In one line:** shuffle a feature and see how much the model's accuracy drops.

**Remember:** model-agnostic and simple; misleading when features are correlated (the model can lean on a substitute).

## Partial dependence

**In one line:** how the prediction changes, on average, as one feature varies.

**Remember:** shows the shape of a relationship ("risk rises sharply after age 60"); individual curves (ICE) reveal when the average hides different behaviors.

## LIME

**In one line:** explain one prediction by fitting a simple model to the black box's behavior in the neighborhood of that example.

**Remember:** intuitive but can be unstable between runs.

## SHAP

**In one line:** split a prediction among features fairly, using Shapley values from game theory.

**Picture it:** dividing a group's prize according to how much each member added across every possible team they could have joined.

**Remember:** contributions add up exactly to the prediction minus the average prediction. Fast and exact for tree models (TreeSHAP).

## Counterfactual explanations

**In one line:** the smallest change that would flip the decision.

**Remember:** "you'd have been approved with a slightly lower debt ratio" is actionable for the person affected.

## Saliency maps

**In one line:** highlight which input pixels most affect the output.

**Watch out:** they can look convincing even for broken models; sanity-check them. Grad-CAM and Integrated Gradients are more reliable variants.

## Mechanistic interpretability

**In one line:** reverse-engineer the internal circuits and features of a neural network.

**Remember:** an active research area; sparse autoencoders find interpretable features inside large models.

## Where unfairness comes from

- **Historical bias:** labels reflect past discrimination.
- **Representation bias:** some groups are underrepresented.
- **Measurement bias:** features mean different things for different groups.
- **Proxies:** removing a sensitive attribute doesn't help if zip code stands in for it.

## Fairness definitions

| Definition | Means |
|---|---|
| Demographic parity | Each group gets positive outcomes at the same rate |
| Equal opportunity | Qualified people in each group are approved at the same rate |
| Equalized odds | Equal true and false positive rates across groups |
| Calibration | A score of 0.7 means the same risk in every group |

**Remember:** when groups' base rates differ, you can't satisfy all of these at once. Choosing one is a values decision about which harm matters most.

## Mitigation

**Remember:** fix data (reweight, collect more), constrain training, or adjust thresholds per group, and always report metrics per group, not just overall.

## Responsible practice

**Remember:** document models (model cards) and data (datasheets), keep humans in high-stakes loops, allow appeals, and monitor fairness after launch because data shifts.
