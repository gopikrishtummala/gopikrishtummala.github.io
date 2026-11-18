---
layout: ../../layouts/CheatsheetLayout.astro
title: "Unsupervised Learning"
description: "Clustering, dimensionality reduction, density estimation, anomaly detection and generative models."
---

## Clustering

| Method | Idea | Pros | Cons |
|---|---|---|---|
| k-means | Assign to nearest centroid, recompute centroids | Fast, simple | Needs k; assumes round, similar-size clusters; sensitive to init (use k-means++) |
| Hierarchical (agglomerative) | Merge closest clusters repeatedly | Dendrogram, no k upfront | $O(n^2)$ memory; linkage choice matters (single, complete, average, Ward) |
| DBSCAN | Dense regions separated by sparse ones | Any shape, finds outliers, no k | Sensitive to `eps`, `min_samples`; struggles with varying density |
| HDBSCAN | Hierarchical DBSCAN | Handles varying density | Slower |
| Gaussian mixture | Soft assignments, elliptical clusters | Probabilities, flexible shapes | Needs k; EM can hit local optima |
| Spectral | Cluster in eigenvectors of a similarity graph | Non-convex shapes | Expensive for large n |

### Choosing k and evaluating

- Elbow method (inertia vs k), silhouette score (−1 to 1, higher better), Davies-Bouldin (lower better), BIC / AIC for mixtures.
- With labels: adjusted Rand index, normalized mutual information.

## Dimensionality reduction

| Method | Type | Use |
|---|---|---|
| PCA | Linear; directions of max variance | Compression, denoising, preprocessing; keep components explaining ~90–95% variance |
| SVD / truncated SVD | Linear | Sparse matrices, text (LSA), recommenders |
| t-SNE | Non-linear, preserves local neighborhoods | 2D/3D visualization only; distances between clusters not meaningful |
| UMAP | Non-linear, faster than t-SNE, keeps more global structure | Visualization, sometimes as features |
| Autoencoder | Neural, non-linear | Learned compression, anomaly detection |
| LDA (linear discriminant) | Supervised, maximizes class separation | Reduction for classification |

- Standardize features before PCA.

## Density estimation and anomaly detection

- Histograms, kernel density estimation, Gaussian mixtures, normalizing flows.
- Anomaly detection: isolation forest, one-class SVM, local outlier factor, autoencoder reconstruction error, density thresholds.

## Generative models

| Family | Idea | Notes |
|---|---|---|
| VAE | Encoder to a latent distribution, decoder back; maximize the ELBO | Smooth latent space; samples can be blurry. See [VAEs](/posts/fundamentals/variational-autoencoders-from-compression-to-creation) |
| GAN | Generator vs discriminator game | Sharp samples; unstable training, mode collapse |
| Normalizing flow | Invertible transforms; exact likelihood | Architecture constraints |
| Autoregressive | Predict next element given previous | LLMs, PixelCNN; slow sequential sampling |
| Diffusion / flow matching | Learn to reverse a noising process | State of the art for images and video. See [Diffusion Foundations](/posts/generative-ai/diffusion-from-molecules-to-machines) |

## Self-supervised representation learning

- Contrastive: pull augmented views of the same item together, push others apart (SimCLR, MoCo, CLIP).
- Masked prediction: hide parts and predict them (BERT, MAE).
- Self-distillation: student matches teacher on different views (DINO).

## Gotchas

- Clusters always appear, even in random data. Validate that they mean something.
- Scaling changes distance-based results completely.
- t-SNE and UMAP plots depend heavily on hyperparameters; don't over-read them.
