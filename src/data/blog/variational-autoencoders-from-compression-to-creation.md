---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-15T00:00:00Z
modDatetime: 2025-01-15T00:00:00Z
title: Variational Autoencoders — From Compression to Creation
slug: variational-autoencoders-from-compression-to-creation
featured: true
draft: false
tags:
  - generative-ai
  - machine-learning
  - neural-networks
  - computer-vision
description: An intuitive introduction to Variational Autoencoders — how compressing data into probabilistic codes enables machines to generate realistic images, sounds, and structures.
---

## 1. From Autoencoders to Generation

Imagine a neural network that looks at an image, compresses it into a few numbers, then reconstructs the image. That's an **autoencoder**.

Now imagine it goes one step further: instead of learning a fixed point — a single code like $(2.3, -1.1)$ — it learns a probability distribution over possible codes.  
Where an autoencoder says "this image maps to $z = (2.3, -1.1)$," a VAE says "this image maps to a Gaussian cloud centered at $(2.3, -1.1)$ with some spread."

This probabilistic twist makes VAEs powerful **generative models**: they can sample new points from that latent distribution and generate realistic new data.

---

## 2. The Standard Autoencoder: A Quick Review

**Given**: A data vector $\mathbf{x} \in \mathbb{R}^D$ (e.g., a flattened 28×28 image, $D = 784$).

**Encoder**: $\mathbf{z} = f(\mathbf{x})$, where $\mathbf{z} \in \mathbb{R}^K$ and $K \ll D$.  
**Decoder**: $\hat{\mathbf{x}} = g(\mathbf{z})$.

**Objective**: Minimize the reconstruction error:

$$
\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \|\mathbf{x} - g(f(\mathbf{x}))\|^2
$$

This forces $\mathbf{z}$ to be a compact representation — a bottleneck that captures only the essential information.

The standard autoencoder compresses information into a fixed vector — a single point in space.  
But what if we could describe *uncertainty* about what that point should be?

That's where **probability** enters the picture.

---

## 3. From Points to Distributions

A standard autoencoder maps $\mathbf{x} \mapsto \mathbf{z}$ deterministically.  
A VAE maps

$$
\mathbf{x} \mapsto q(\mathbf{z}|\mathbf{x})
$$

to a probability distribution $q(\mathbf{z}|\mathbf{x})$.

We choose a **Gaussian** form:

$$
q(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z} ; \boldsymbol{\mu}(\mathbf{x}), \,\text{diag}(\boldsymbol{\sigma}^2(\mathbf{x})))
$$

The encoder neural network now outputs **two vectors**:

$$
\boldsymbol{\mu}(\mathbf{x}), \quad \log\boldsymbol{\sigma}^2(\mathbf{x})
$$

(We output log-variance for numerical stability.)

The Gaussian assumption means: given an input $\mathbf{x}$, our encoder believes the hidden code $\mathbf{z}$ is likely to lie near $\boldsymbol{\mu}(\mathbf{x})$, but not exactly — there's some uncertainty $\boldsymbol{\sigma}(\mathbf{x})$.

---

## 4. Sampling with the Reparameterization Trick

To generate a latent code $\mathbf{z}$, we sample:

$$
\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \,\text{diag}(\boldsymbol{\sigma}^2))
$$

Direct sampling breaks back-propagation because the sampling operation is not differentiable.  
**Solution**: the **reparameterization trick**:

$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})
$$

where $\odot$ is element-wise multiplication.

**Proof of equivalence**:  
Let $\varepsilon_j \sim \mathcal{N}(0,1)$. Then:

$$
z_j = \mu_j + \sigma_j \varepsilon_j \sim \mathcal{N}(\mu_j, \sigma_j^2)
$$

The randomness $\boldsymbol{\varepsilon}$ is external; gradients now flow through $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$.

**An intuitive analogy**:  
Imagine a painter: if it draws the same picture every time, it isn't creative.  
Controlled randomness adds exploration without losing structure.

---

## 5. The Decoder and Reconstruction Loss

The decoder is a neural network:

$$
p(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; \hat{\mathbf{x}}(\mathbf{z}), \,\mathbf{I})
$$

For pixel values in $[0,1]$, we use a **Bernoulli distribution**:

$$
p(\mathbf{x}|\mathbf{z}) = \prod_{d=1}^D x_d^{\hat{x}_d(\mathbf{z})} (1-x_d)^{1-\hat{x}_d(\mathbf{z})}
$$

**Reconstruction loss** (negative log-likelihood):

$$
\mathcal{L}_{\text{recon}} = -\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})]
$$

For Gaussian (MSE):

$$
\mathcal{L}_{\text{recon}} = \|\mathbf{x} - \hat{\mathbf{x}}(\mathbf{z})\|^2
$$

In practice, we use a Monte Carlo estimate with one sample:

$$
\mathcal{L}_{\text{recon}} \approx \|\mathbf{x} - g(\mathbf{z})\|^2, \quad \mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}
$$

---

## 6. The Regularization Term: KL Divergence

We want $q(\mathbf{z}|\mathbf{x})$ close to a **standard normal prior**

$$
p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})
$$

This encourages a compact, organized latent space where codes follow a standard Gaussian.

We measure distance using **Kullback–Leibler divergence**:

$$
D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) = \mathbb{E}_q[\log q(\mathbf{z}|\mathbf{x}) - \log p(\mathbf{z})]
$$

**Special case** ($p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})$, diagonal $\boldsymbol{\sigma}^2$):

$$
D_{\text{KL}} = \frac{1}{2} \sum_{j=1}^K \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)
$$

This term acts as a **regularizer**: it pushes means toward zero and variances toward one, producing a smooth, continuous latent space.

---

## 7. The Evidence Lower Bound (ELBO)

The **log marginal probability** is:

$$
\log p(\mathbf{x}) = \log \int p(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) \, d\mathbf{z}
$$

This integral is intractable.  
Introducing $q(\mathbf{z}|\mathbf{x})$ and using Jensen's inequality:

$$
\log p(\mathbf{x}) \ge \mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

This is the **Evidence Lower BOund (ELBO)**.  
Maximizing the ELBO $\equiv$ minimizing:

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + D_{\text{KL}}
$$

---

## 8. Full Training Objective

For a dataset $\{\mathbf{x}^{(i)}\}_{i=1}^N$, we perform stochastic gradient descent on:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \left[ \|\mathbf{x}^{(i)} - g(\boldsymbol{\mu}^{(i)} + \boldsymbol{\sigma}^{(i)} \odot \boldsymbol{\varepsilon})\|^2 + \frac{1}{2} \sum_{j=1}^K (\mu_j^{(i)2} + \sigma_j^{(i)2} - \log \sigma_j^{(i)2} - 1) \right]
$$

**Algorithm** (one step):  
1. Sample minibatch  
2. Forward: compute $\boldsymbol{\mu}, \log\boldsymbol{\sigma}^2$  
3. Sample $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$  
4. $\mathbf{z} = \boldsymbol{\mu} + \exp(0.5\log\boldsymbol{\sigma}^2) \odot \boldsymbol{\varepsilon}$  
5. Reconstruct $\hat{\mathbf{x}} = g(\mathbf{z})$  
6. Compute $\mathcal{L}$, back-propagate

---

## 9. Latent Space Structure

After training, the map $\mathbf{x} \mapsto \boldsymbol{\mu}(\mathbf{x})$ embeds data in $\mathbb{R}^K$.

Points are **continuously distributed**.  
Think of latent space as a map of concepts.  
Linear interpolation in latent space produces smooth transitions in data space.

**Example** (MNIST, $K=2$):  
- $\boldsymbol{\mu}_1$: controls digit identity ($0 \to 9$)  
- $\boldsymbol{\mu}_2$: controls stroke thickness

You can walk through latent space and watch digits morph.

---

## 10. Connection to Diffusion Models

Diffusion models also learn probability distributions, but instead of encoding images into a latent space, they gradually destroy and rebuild data using noise.

VAEs compress data into a probabilistic code; diffusion expands noise into data.  
Both aim to learn $p(\mathbf{x})$.

Together, VAEs and diffusion form two paths to generative modeling:
- **VAEs**: learn a probabilistic compression/expansion
- **Diffusion**: learn a reverse noising/denoising process

Both share the same goal: learning the distribution of the data.

---

## 11. Applications

### 11.1 Image Generation

VAEs can sample $\mathbf{z} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ and decode to new images.  
Likely images map to likely codes; sampling from a standard Gaussian yields new samples.

### 11.2 Anomaly Detection

An image far from the training distribution will reconstruct poorly.  
The reconstruction error can flag anomalies.

### 11.3 Latent Editing

Factorizing concepts in latent space enables targeted edits (e.g., smile).

### 11.4 Data Augmentation

Sample from $q(\mathbf{z}|\mathbf{x})$ to generate variations of a training example.

---

## 12. Why VAEs Matter

| Aspect | VAEs |
|--------|------|
| **Goal** | Learn $p(\mathbf{x})$ via probabilistic latent codes |
| **Mechanism** | Encode to Gaussian, decode with reparameterization |
| **Training** | Maximize ELBO = reconstruction + regularization |
| **Output** | Continuous latent space for interpolation and generation |

VAEs are **stable to train**: unlike GANs, there's no discriminator, and the objective is bounded.

They bridge **compression** and **generation**: a single architecture learns efficient representations and generates new data.

---

## 13. Limitations and Trade-offs

The Gaussian assumption is simplistic for complex data, and the ELBO is a lower bound, not an exact likelihood.  
Weighting reconstruction vs. KL ($\beta$-VAE) exposes a trade-off: more emphasis on realism vs. regularization.

Contemporary approaches blend VAEs and diffusion for improved generation quality.

---

## 14. Summary

- **Autoencoders** compress to fixed codes; **VAEs** use probabilistic distributions.
- The **reparameterization trick** enables differentiable sampling.
- **ELBO = reconstruction + KL divergence** guides training.
- A continuous latent space supports interpolation and generation.

From compression to creation, VAEs show how adding probability to neural networks enables generation from learned structure.

