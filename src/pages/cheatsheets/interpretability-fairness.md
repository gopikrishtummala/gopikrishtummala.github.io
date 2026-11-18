---
layout: ../../layouts/CheatsheetLayout.astro
title: "Interpretability & Fairness"
description: "Feature attribution, SHAP, explanations for deep models, fairness definitions and their trade-offs."
---

## Interpretability: kinds of explanation

| | Global (whole model) | Local (one prediction) |
|---|---|---|
| Model-agnostic | Permutation importance, partial dependence, global surrogate | LIME, SHAP values, counterfactuals |
| Model-specific | Linear coefficients, tree structure, attention maps (with care) | Gradients, Integrated Gradients, Grad-CAM |

- **Intrinsically interpretable:** linear models, small trees, rule lists, GAMs (generalized additive models).
- **Post-hoc:** explain a black box after training.

## Methods

- **Permutation importance:** shuffle a feature, measure the drop in score. Misleading with correlated features.
- **Partial dependence (PDP):** average prediction as one feature varies. **ICE:** one curve per example. **ALE:** handles correlated features better.
- **LIME:** fit a simple model to the black box's behavior around one example. Unstable between runs.
- **SHAP:** Shapley values from game theory; each feature's fair share of the difference from the average prediction. Additive and consistent. TreeSHAP is exact and fast for trees; KernelSHAP is slow and approximate.
- **Counterfactual explanations:** the smallest change that flips the decision ("approved if income were $5k higher").
- **Saliency / gradients:** sensitivity of output to input pixels; noisy. **Integrated Gradients** fixes some issues. **Grad-CAM:** class-specific heatmaps for CNNs.
- **Mechanistic interpretability:** reverse-engineer circuits and features inside networks (probing, sparse autoencoders).

## Interpretability pitfalls

- Explanations explain the model, not the world; correlated features share or swap credit.
- Attention weights are not reliable explanations by themselves.
- Saliency maps can look plausible even for random models; sanity-check them.

## Sources of unfairness

- **Historical bias:** labels reflect past discrimination.
- **Representation bias:** some groups under-sampled.
- **Measurement bias:** proxies measured differently across groups.
- **Proxy features:** zip code standing in for race; removing the sensitive attribute doesn't remove bias.

## Fairness definitions

| Definition | Requires | Notes |
|---|---|---|
| Demographic parity | Equal positive rate across groups | Ignores whether labels differ |
| Equal opportunity | Equal true positive rate | Fair to qualified people |
| Equalized odds | Equal TPR and FPR | Stricter |
| Predictive parity | Equal precision | Equal meaning of a positive prediction |
| Calibration within groups | Scores mean the same in each group | |
| Individual fairness | Similar people treated similarly | Needs a similarity measure |

- **Impossibility:** when base rates differ, calibration and equal error rates can't all hold at once (Kleinberg et al., Chouldechova). Choose based on the harm you care about.

## Mitigation

- **Pre-processing:** reweight or resample data, fix labels.
- **In-processing:** fairness constraints or adversarial debiasing during training.
- **Post-processing:** group-specific thresholds.
- Always measure metrics per group, not just overall.

## Responsible AI practice

- Model cards and datasheets: document intended use, data, limitations, per-group metrics.
- Human review for high-stakes decisions; appeal processes.
- Privacy: differential privacy, federated learning, data minimization.
- Monitor fairness after deployment, since data shifts.
