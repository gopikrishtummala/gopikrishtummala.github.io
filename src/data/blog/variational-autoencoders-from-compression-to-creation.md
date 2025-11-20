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
track: Fundamentals
difficulty: Intermediate
interview_relevance:
  - Theory
estimated_read_time: 30
---

## 1. From Autoencoders to Generation

Imagine a neural network that looks at an image, compresses it into a few numbers, then reconstructs the image. That's an **autoencoder**.

Now imagine it goes one step further: instead of learning a fixed point — a single code like $(2.3, -1.1)$ — it learns a probability distribution over possible codes.  
Where an autoencoder says "this image maps to $z = (2.3, -1.1)$," a VAE says "this image maps to a Gaussian cloud centered at $(2.3, -1.1)$ with some spread."

**Why this matters**: The traditional autoencoder's latent space is like **Swiss Cheese** — every data point is a small piece of cheese, but the space *between* the pieces is full of empty, meaningless air. If you sample a random point between codes, you get garbage output.  

The VAE's latent space is like a smooth, continuous **lump of dough**. Because the codes are forced to overlap as Gaussian distributions, any point you poke in the dough yields a meaningful, smooth result. This is the **completeness** property — the latent space is "filled in" and supports generation from any point.

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

**Why this matters**: The reparameterization trick is the **crux of the VAE's trainability**. It cleverly converts a non-differentiable stochastic (random) operation—the $\mathbf{z} \sim q(\mathbf{z}|\mathbf{x})$ sampling—into a differentiable, deterministic function $\mathbf{z} = f(\boldsymbol{\mu}, \boldsymbol{\sigma}, \boldsymbol{\varepsilon})$ that allows us to use standard gradient descent to train the network.

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

### 6.1 Understanding KL Divergence Intuitively

**The Kullback–Leibler (KL) divergence** measures the difference between two probability distributions. It quantifies how much information is lost when using an approximation distribution $Q$ instead of the true distribution $P$.

**An intuitive analogy**: Imagine you have two bags of marbles.

- **Bag P (True Distribution)**: 8 red marbles, 2 blue marbles (80% red, 20% blue). This is the actual reality.
- **Bag Q (Approximation)**: 5 red marbles, 5 blue marbles (50% red, 50% blue). This is your model or belief about the bag.

The KL divergence tells you how different your belief (Bag Q) is from reality (Bag P).

**The core idea**: In information theory, we think about "surprise." An event with a low probability is very surprising, while a likely event is not. The KL divergence measures the average extra surprise you experience when you use the wrong distribution ($Q$) instead of the correct one ($P$) to make predictions.

- If your belief $Q$ is identical to the true distribution $P$, the KL divergence is zero. You have no extra surprise.
- The more different $Q$ is from $P$, the higher the KL divergence value.
- It's not a true "distance" because it's not symmetric: the difference from $P$ to $Q$ is not the same as from $Q$ to $P$.

### 6.2 The Mathematical Formula

For discrete probability distributions $P$ and $Q$ over the same set of events, the KL divergence of $Q$ from $P$ (often written as $D_{KL}(P||Q)$) is:

$$
D_{KL}(P||Q) = \sum_{x} P(x) \cdot \log \left(\frac{P(x)}{Q(x)}\right)
$$

This can also be written as:

$$
D_{KL}(P||Q) = \sum_{x} P(x) \cdot (\log(P(x)) - \log(Q(x)))
$$

**Notation:**
- $P(x)$: The actual probability of event $x$ happening.
- $Q(x)$: Your estimated (approximating) probability of event $x$ happening.
- $\log$: Usually base 2 for "bits" of information, or base $e$ (natural log) for "nats."
- $\sum_{x}$: Sum over all possible outcomes $x$.

**Key insight**: The formula calculates a weighted average of the logarithmic difference between the true and approximate probabilities, where the weights are the actual probabilities $P(x)$. This means events that are likely to happen in reality (high $P(x)$) have a bigger impact on the final score.

**A philosophical perspective (Plato's Cave)**: The observed data $\mathbf{x}$ are like the shadows on the wall of Plato's cave. The true latent code $\mathbf{z}$ is the real object casting the shadow. The KL divergence forces your learned approximation $q(\mathbf{z}|\mathbf{x})$ to be a well-structured map of the "true reality" $p(\mathbf{z})$, ensuring the codes you learn are genuine "forms," not just arbitrary shadows. It ensures that the latent representations capture meaningful structure rather than being arbitrary encodings.

### 6.3 KL Divergence in VAEs

In VAEs, we want the learned distribution $q(\mathbf{z}|\mathbf{x})$ to match a standard normal prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})$.

**Special case** ($p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})$, diagonal $\boldsymbol{\sigma}^2$):

$$
D_{\text{KL}} = \frac{1}{2} \sum_{j=1}^K \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)
$$

Let's break down what each term does:

- **$\mu_j^2$**: Penalizes large mean values, pushing them toward zero. If the encoder outputs a mean far from zero, this term grows quadratically.
- **$\sigma_j^2$**: Encourages variance to be close to 1. If variance is too small or too large, this term increases.
- **$-\log \sigma_j^2$**: Prevents variance from collapsing to zero. As $\sigma_j \to 0$, $-\log \sigma_j^2 \to +\infty$, creating infinite penalty.
- **$-1$**: Normalization constant to ensure the KL divergence is zero when $\boldsymbol{\mu} = \mathbf{0}$ and $\boldsymbol{\sigma}^2 = \mathbf{1}$.

### 6.4 Why This Regularization Matters

This term acts as a **regularizer**: it pushes means toward zero and variances toward one, producing a smooth, continuous latent space.

**Without KL regularization**: The encoder could learn arbitrary distributions for different inputs. Some might have means at $(10, -5, 3)$ while others at $(-8, 2, -1)$. The latent space would be fragmented, with codes scattered arbitrarily. Interpolation between codes wouldn't make sense, and sampling from $\mathcal{N}(\mathbf{0},\mathbf{I})$ wouldn't correspond to realistic images.

**With KL regularization**: All learned distributions are pushed toward the standard normal. The latent space becomes organized and continuous. Codes cluster around the origin, and smooth interpolation becomes possible. Sampling from $\mathcal{N}(\mathbf{0},\mathbf{I})$ now corresponds to sampling from regions where the model has seen training data.

**The trade-off**: There's a tension between reconstruction quality and regularization:
- Too much KL weight ($\beta$-VAE with high $\beta$): Excellent latent space structure, but blurrier reconstructions because the model is constrained.
- Too little KL weight: Better reconstructions, but a less organized latent space that may have "holes" or discontinuities.

This is why $\beta$-VAE introduces a hyperparameter $\beta$ to weight the KL term:

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}
$$

Tuning $\beta$ lets you balance between faithful reconstructions and a well-structured latent space.

---

## 7. The Evidence Lower Bound (ELBO): The Math Behind the Optimization Goal

> 💡 *This section is for students curious about the deeper mathematical foundation. If you are focused on intuition and applications, you can skip to Section 8 or 9.*

This mathematical expression is the basis for the Variational Autoencoder (VAE) and Variational Inference methods. It shows how an intractable log-likelihood can be approximated with a tractable lower bound.

### 7.0 The Loss Function's "Tug-of-War"

Before diving into the mathematical derivation, it's helpful to understand the intuitive tension in the VAE loss function. The VAE loss $\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + D_{\text{KL}}$ is a perfect example of constrained optimization, where two competing forces pull in opposite directions:

| Loss Term | What it Pushes For | Analogy |
| :--- | :--- | :--- |
| **$\mathcal{L}_{\text{recon}}$ (Reconstruction)** | **Fidelity:** Forces the decoder to output a sharp, accurate version of the input. | **A Photographer:** Demands perfect copies, pushing codes far apart to avoid confusion. |
| **$D_{\text{KL}}$ (Regularization)** | **Structure:** Forces the encoder's output distributions to overlap and conform to $\mathcal{N}(\mathbf{0}, \mathbf{I})$. | **A Librarian:** Demands all codes be stored neatly in a specific, central filing system, pushing codes closer together. |

This "tug-of-war" creates a balance: the reconstruction term wants perfect fidelity (spreading codes apart), while the KL term wants perfect organization (clustering codes together). The optimal solution lies somewhere in between — good enough reconstruction with a well-structured latent space.

### 7.1 Step 1: Rewriting the Log-Likelihood

The initial expression for the log-likelihood of a data point $\mathbf{x}$ is given by:

$$
\log p(\mathbf{x}) = \log \int p(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}
$$

The integral over the latent variable $\mathbf{z}$ is often intractable because it requires integrating over all possible values of $\mathbf{z}$.

To address this, we introduce an arbitrary distribution $q(\mathbf{z}|\mathbf{x})$, which we can choose to be a simple, tractable distribution (e.g., a normal distribution). We can multiply and divide the integrand by this distribution $q(\mathbf{z}|\mathbf{x})$:

$$
\log p(\mathbf{x}) = \log \int \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} q(\mathbf{z}|\mathbf{x}) \, d\mathbf{z}
$$

This can be re-written as an expectation with respect to $q(\mathbf{z}|\mathbf{x})$:

$$
\log p(\mathbf{x}) = \log \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right]
$$

### 7.2 Step 2: Applying Jensen's Inequality

Jensen's inequality states that for a concave function $f$ (like the logarithm), and a random variable $X$:

$$
f(\mathbb{E}[X]) \ge \mathbb{E}[f(X)]
$$

Applying this to the expression from Step 1:

$$
\log \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right] \ge \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right]
$$

The expression on the right-hand side is a lower bound on the log-likelihood, often called the **Evidence Lower Bound (ELBO)**.

### 7.3 Step 3: Expanding the Lower Bound

We can expand the ELBO:

$$
\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right] = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}, \mathbf{z}) - \log q(\mathbf{z}|\mathbf{x})]
$$

Next, we use the property of conditional probability, $p(\mathbf{x}, \mathbf{z}) = p(\mathbf{x}|\mathbf{z}) p(\mathbf{z})$:

$$
\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z}) + \log p(\mathbf{z}) - \log q(\mathbf{z}|\mathbf{x})]
$$

By rearranging the terms and splitting the expectation, we arrive at the final form of the lower bound:

$$
\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] + \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{z}) - \log q(\mathbf{z}|\mathbf{x})]
$$

### 7.4 Recognizing the KL Divergence

The second term is the negative of the Kullback–Leibler (KL) divergence between the two distributions $q(\mathbf{z}|\mathbf{x})$ and $p(\mathbf{z})$. The KL divergence is defined as $D_{KL}(A||B) = \mathbb{E}_A[\log(A/B)]$. Therefore:

$$
-D_{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log \frac{p(\mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right] = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{z}) - \log q(\mathbf{z}|\mathbf{x})]
$$

### 7.5 The Final ELBO Expression

Putting it all together, we have:

$$
\log p(\mathbf{x}) \ge \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] - D_{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

**Summary**: The equation $\log p(\mathbf{x}) \ge \mathbb{E}_{q}[\log p(\mathbf{x}|\mathbf{z})] - D_{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$ is derived by using Jensen's inequality on the logarithm of the log-likelihood. This converts the intractable integral over the latent variable $\mathbf{z}$ into a tractable lower bound, called the Evidence Lower Bound (ELBO).

This lower bound consists of two parts:
1. **Reconstruction term** $\mathbb{E}_{q}[\log p(\mathbf{x}|\mathbf{z})]$: Measures how well the model can reconstruct the input $\mathbf{x}$ from a latent code $\mathbf{z}$ sampled from $q(\mathbf{z}|\mathbf{x})$.
2. **Regularization term** $-D_{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$: Measures the difference between the approximate posterior $q(\mathbf{z}|\mathbf{x})$ and the prior $p(\mathbf{z})$. This keeps the learned distributions close to the prior.

Since we want to maximize the log-likelihood $\log p(\mathbf{x})$, we maximize its lower bound (the ELBO). This is equivalent to minimizing:

$$
\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})] + D_{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))
$$

Or, in terms of the components we've seen:

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + D_{\text{KL}}
$$

where $\mathcal{L}_{\text{recon}} = -\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})]$ is the reconstruction loss.

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

## 12. Extensions: Conditional VAEs and Vector Quantized VAEs

The standard VAE framework has been extended in several directions to address specific limitations and enable new capabilities. Two important variants are Conditional VAEs (CVAEs) and Vector Quantized VAEs (VQ-VAEs).

---

## 12.1 Conditional Variational Autoencoders (CVAEs)

A **Conditional Variational Autoencoder (CVAE)** is a modification of the traditional VAE that introduces conditional generation based on additional information such as class labels, attributes, or other input conditions.

### 12.1.1 The Conditional Framework

In a standard VAE, we model $p(\mathbf{x})$ unconditionally. In a CVAE, we condition both the encoder and decoder on additional information $\mathbf{y}$ (e.g., class labels, text descriptions, or other attributes):

- **Conditional Encoder**: $q(\mathbf{z}|\mathbf{x}, \mathbf{y})$ — encodes input $\mathbf{x}$ into latent code $\mathbf{z}$ given condition $\mathbf{y}$
- **Conditional Decoder**: $p(\mathbf{x}|\mathbf{z}, \mathbf{y})$ — decodes latent code $\mathbf{z}$ to data $\mathbf{x}$ given condition $\mathbf{y}$

The ELBO becomes:

$$
\log p(\mathbf{x}|\mathbf{y}) \ge \mathbb{E}_{q(\mathbf{z}|\mathbf{x},\mathbf{y})}[\log p(\mathbf{x}|\mathbf{z},\mathbf{y})] - D_{KL}(q(\mathbf{z}|\mathbf{x},\mathbf{y}) \| p(\mathbf{z}|\mathbf{y}))
$$

Typically, we assume the prior $p(\mathbf{z}|\mathbf{y}) = p(\mathbf{z}) = \mathcal{N}(\mathbf{0},\mathbf{I})$ is independent of the condition, simplifying to:

$$
\log p(\mathbf{x}|\mathbf{y}) \ge \mathbb{E}_{q(\mathbf{z}|\mathbf{x},\mathbf{y})}[\log p(\mathbf{x}|\mathbf{z},\mathbf{y})] - D_{KL}(q(\mathbf{z}|\mathbf{x},\mathbf{y}) \| p(\mathbf{z}))
$$

### 12.1.2 Implementation Strategy

The condition $\mathbf{y}$ is typically incorporated by concatenating it to the input at various stages:

**Encoder**: The condition is concatenated with the input $\mathbf{x}$ before encoding. For images, this often means:
- One-hot encoding the label to match spatial dimensions
- Concatenating along the channel dimension: $[\mathbf{x}, \mathbf{y}]$
- Passing the concatenated tensor through the encoder network

**Decoder**: The condition is concatenated with the latent code $\mathbf{z}$ before decoding:
- Embedding $\mathbf{y}$ to match $\mathbf{z}$'s dimensionality
- Concatenating: $[\mathbf{z}, \text{embed}(\mathbf{y})]$
- Passing through the decoder network

### 12.1.3 Why CVAEs Matter

**Controlled Generation**: CVAEs enable generating samples with specific attributes. For example:
- Generate images of a specific digit class (MNIST: generate only "7"s)
- Create faces with particular features (CelebA: generate smiling faces)
- Produce text-conditioned images (generate "a cat sitting on grass")

**Better Latent Structure**: By conditioning on class labels, the model learns to separate class-relevant information in the latent space, potentially improving disentanglement.

**Practical Applications**:
- **Data Augmentation**: Generate class-specific training examples
- **Content Creation**: Controlled generation for creative applications
- **Semi-supervised Learning**: Leverage both labeled and unlabeled data

---

## 12.2 Vector Quantized Variational Autoencoders (VQ-VAEs)

**Vector Quantized Variational Autoencoders (VQ-VAEs)** replace the continuous latent space with a **discrete codebook**, enabling the model to learn discrete latent representations instead of continuous ones.

### 12.2.1 The Discrete Latent Space

Unlike standard VAEs that use continuous Gaussian distributions, VQ-VAEs use:

- **Discrete Latent Variables**: Instead of sampling from $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$, we select from a finite set of vectors
- **Codebook**: A learned dictionary $\mathbf{E} = \{\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_K\}$ of $K$ embedding vectors, each of dimension $d$
- **Quantization**: The encoder output is mapped to the nearest codebook vector

### 12.2.2 Architecture Overview

**Encoder**: Maps input $\mathbf{x}$ to continuous output $\mathbf{z}_e$ (same as standard VAE encoder)

**Vector Quantization (VQ) Layer**: 
1. Reshape encoder output $\mathbf{z}_e$ into vectors: $(n \times h \times w, d)$
2. For each vector, find the nearest codebook entry:
   $$
   \mathbf{z}_q = \mathbf{e}_k \text{ where } k = \arg\min_j \|\mathbf{z}_e - \mathbf{e}_j\|^2
   $$
3. Reshape quantized vectors back to spatial dimensions

**Decoder**: Reconstructs from quantized codes $\mathbf{z}_q$

### 12.2.3 The Challenge: Differentiability

The quantization step (argmin) is not differentiable, preventing gradient flow. VQ-VAE solves this with the **straight-through estimator**:

- **Forward pass**: Use quantized codes $\mathbf{z}_q$ for decoding
- **Backward pass**: Copy gradients from $\mathbf{z}_q$ directly to $\mathbf{z}_e$, bypassing the quantization step

This allows training while effectively treating quantization as an identity function during backpropagation.

### 12.2.4 Loss Function

VQ-VAE uses three loss components:

**1. Reconstruction Loss**:
$$
\mathcal{L}_{\text{recon}} = -\log p(\mathbf{x}|\mathbf{z}_q)
$$

**2. Codebook Loss** (Vector Quantization Loss):
$$
\mathcal{L}_{\text{codebook}} = \|\text{sg}[\mathbf{z}_e] - \mathbf{e}_k\|^2
$$
where $\text{sg}[\cdot]$ is the stop-gradient operator. This moves codebook vectors toward encoder outputs.

**3. Commitment Loss**:
$$
\mathcal{L}_{\text{commit}} = \beta \|\mathbf{z}_e - \text{sg}[\mathbf{z}_q]\|^2
$$
where $\beta$ is a hyperparameter (typically 0.25). This prevents the encoder from growing unboundedly by encouraging it to commit to codebook vectors.

**Total Loss**:
$$
\mathcal{L}_{\text{VQ-VAE}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{codebook}} + \mathcal{L}_{\text{commit}}
$$

### 12.2.5 Why VQ-VAEs Matter

**Discrete Representations**: Many real-world concepts are inherently discrete (categories, objects, words). VQ-VAEs capture these naturally without forcing continuous interpolation.

**Posterior Collapse Mitigation**: Standard VAEs can suffer from posterior collapse where the decoder ignores the latent code. VQ-VAEs' discrete bottleneck forces meaningful use of the latent space.

**Hierarchical Modeling**: VQ-VAEs enable multi-scale discrete representations, useful for modeling complex structures (e.g., images at multiple resolutions).

**Applications**:
- **High-Quality Image Generation**: VQ-VAE-2 achieves state-of-the-art results
- **Audio Generation**: Discrete tokens are natural for audio codecs
- **Language Modeling**: Can be combined with autoregressive models for text generation
- **Foundation Models**: VQ-VAE components appear in models like DALL·E

### 12.2.6 Connection to Autoregressive Models

VQ-VAEs are often combined with autoregressive models (e.g., Transformers) to model the discrete latent sequence:

1. **VQ-VAE** learns to compress data into discrete tokens
2. **Autoregressive Model** learns the distribution over these tokens: $p(\mathbf{z}_q)$
3. **Generation**: Sample tokens autoregressively, then decode with VQ-VAE decoder

This two-stage approach separates representation learning (VQ-VAE) from generation modeling (autoregressive model), enabling both high-quality compression and powerful generation.

---

## 13. Why VAEs Matter

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

**Extensions**:
- **CVAEs** enable conditional generation by incorporating additional information (labels, attributes) into both encoder and decoder.
- **VQ-VAEs** use discrete codebooks instead of continuous distributions, capturing inherently discrete concepts and mitigating posterior collapse.

From compression to creation, VAEs show how adding probability to neural networks enables generation from learned structure. Their extensions—conditional and discrete variants—expand the framework's applicability to controlled generation and hierarchical modeling.

