---
layout: ../../layouts/CheatsheetLayout.astro
title: "Unsupervised Learning in 2 Minutes Each"
description: "Intuition cards for k-means, hierarchical clustering, DBSCAN, Gaussian mixtures, choosing the number of clusters, PCA, t-SNE and UMAP, autoencoders, anomaly detection and self-supervised representation learning."
---

Finding structure without labels.

## k-means

Place k centers, assign each point to its nearest center, move each center to the middle of its points, repeat.

**Picture it:** k pizza shops repeatedly relocating to the middle of their customers.

**Remember:** fast and simple; assumes round, similar-sized clusters; you must choose k; use k-means++ initialization.

## Hierarchical clustering

Repeatedly merge the two closest clusters, building a tree.

**Remember:** no need to pick k up front; cut the tree (dendrogram) where you like. Expensive for large data.

## DBSCAN

Clusters are dense regions; points in sparse regions are noise.

**Picture it:** islands of people in a park; loners aren't forced into a group.

**Remember:** finds any shape and flags outliers; sensitive to its distance setting; struggles when clusters have different densities (HDBSCAN helps).

## Gaussian mixture models

Soft clustering where each cluster is a stretched bell shape.

**Remember:** gives membership probabilities ("70% cluster A"), handles elliptical clusters, fitted with EM.

## Choosing the number of clusters

**Remember:** elbow plot, silhouette score, BIC for mixtures, and most importantly whether the clusters mean something for your use case. Clustering always finds clusters, even in random noise.

## PCA

Find the directions where the data varies most and keep only those.

**Picture it:** photographing a 3D object from the angle that shows the most detail.

**Remember:** linear, fast, great for compression and denoising. Standardize first. Components are combinations of features, not individual features.

## t-SNE and UMAP

Squash high-dimensional data into 2D for plotting while keeping neighbors near each other.

**Remember:** for visualization, not analysis: cluster sizes and distances between clusters in the plot don't mean much. UMAP is faster and keeps more global structure.

## Autoencoders

A network that compresses its input and reconstructs it.

**Remember:** the compressed code is a learned representation; high reconstruction error flags unusual inputs.

## Anomaly detection

Find what doesn't look like the rest.

**Remember:** isolation forest (anomalies are easy to isolate with few random splits), local outlier factor (lower density than neighbors), autoencoder reconstruction error. Labels for anomalies are rare, so evaluation is hard.

## Self-supervised representation learning

Invent a task from unlabeled data whose solution requires understanding it.

| Approach | Task |
|---|---|
| Contrastive (SimCLR, CLIP) | Pull matching pairs together, push others apart |
| Masked prediction (BERT, MAE) | Fill in hidden parts |
| Self-distillation (DINO) | Match a teacher's view of a different crop |

**Remember:** this is how modern foundation models learn from unlabeled data at scale.

## Generative models

**Go deeper:** [Generative Models cheatsheet](/cheatsheets/generative-models).

## Watch out

- Distance-based methods depend heavily on feature scaling.
- High dimensions make distances meaningless; reduce dimensions first.
- Validate clusters with domain knowledge, not just scores.
