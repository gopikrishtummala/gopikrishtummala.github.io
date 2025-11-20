---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-25T00:00:00Z
modDatetime: 2025-01-25T00:00:00Z
title: 'Image Diffusion Models: From U-Net to DiT'
slug: image-diffusion-models-unet-to-dit
featured: true
draft: false
tags:
  - generative-ai
  - diffusion-models
  - computer-vision
  - transformers
  - machine-learning
  - deep-learning
description: 'The evolution of image diffusion models from U-Net architectures to Diffusion Transformers (DiT). Covers latent diffusion, the DiT revolution, and the complete image generation pipeline.'
track: GenAI Systems
difficulty: Advanced
interview_relevance:
  - Theory
  - System Design
  - ML-Infra
estimated_read_time: 15
---

*By Gopi Krishna Tummala*

---

<div class="series-nav" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Diffusion Models Series</div>
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;">
    <a href="/posts/diffusion-from-molecules-to-machines" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 1: Foundations</a>
    <a href="/posts/image-diffusion-models-unet-to-dit" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Part 2: Image Diffusion</a>
    <a href="/posts/video-diffusion-fundamentals" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 3: Video Fundamentals</a>
    <a href="/posts/pre-training-post-training-video-diffusion" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 4: Pre-Training & Post-Training</a>
    <a href="/posts/modern-video-models-sora-veo-opensora" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 5: Modern Models & Motion</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Part 2: Image Diffusion Models</strong> — From U-Net to DiT</div>
</div>

---

<div id="article-toc" class="article-toc">
  <div class="toc-header">
    <h3>Table of Contents</h3>
    <button id="toc-toggle" class="toc-toggle" aria-label="Toggle table of contents"><span>▼</span></button>
  </div>
  <div class="toc-search-wrapper">
    <input type="text" id="toc-search" class="toc-search" placeholder="Search sections..." autocomplete="off">
  </div>
  <nav class="toc-nav" id="toc-nav">
    <ul>
      <li><a href="#image-diffusion">Image Diffusion Models: From U-Net to DiT</a></li>
      <li><a href="#unet-era">The U-Net Era: Convolutional Foundations</a></li>
      <li><a href="#dit-revolution">The DiT Revolution: Scalable Transformers</a></li>
      <li><a href="#architecture-evolution">Image Diffusion Architecture Evolution</a></li>
      <li><a href="#latent-diffusion">Latent Diffusion: The Efficiency Breakthrough</a></li>
      <li><a href="#generation-pipeline">Image Generation Pipeline</a></li>
    </ul>
  </nav>
</div>

---

<a id="image-diffusion"></a>
## Image Diffusion Models: From U-Net to DiT

Diffusion models revolutionized image generation by learning to reverse a noise process. The journey from U-Net-based architectures to Transformer-based models (DiT) represents a fundamental shift in how we approach generative modeling.

<a id="unet-era"></a>
### The U-Net Era: Convolutional Foundations

Early diffusion models like **Stable Diffusion** used U-Net architectures:

* **Encoder-decoder structure**: Compress images to latent space, then denoise
* **Convolutional layers**: Local feature extraction
* **Skip connections**: Preserve spatial details during denoising

The forward diffusion process:

$$
q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

The reverse denoising process:

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}\big(\mu_\theta(x_t,t), \Sigma_t\big)
$$

Where $\mu_\theta$ is learned by a U-Net that predicts the noise to remove.

<a id="dit-revolution"></a>
### The DiT Revolution: Scalable Transformers

**Diffusion Transformers (DiT)** (Peebles & Xie, 2023) replaced U-Nets with Vision Transformers, enabling better scaling:

1. **Patch Embedding**: Image is split into patches (e.g., 16×16 pixels)
2. **Transformer Blocks**: Self-attention processes all patches globally
3. **Conditional Generation**: Cross-attention with text embeddings

The key advantage: **global receptive field** from the start, rather than building it through stacked convolutions.

$$
\text{Attention}(Q,K,V) = \text{softmax}\Big( \frac{QK^\top}{\sqrt{d}} \Big)V
$$

<a id="architecture-evolution"></a>
### Image Diffusion Architecture Evolution

| Architecture | Year | Key Innovation |
| ----------- | ---- | -------------- |
| **DDPM** | 2020 | U-Net denoiser, simple noise schedule |
| **Stable Diffusion** | 2022 | Latent diffusion (VAE encoder/decoder) |
| **DiT** | 2023 | Pure Transformer, no convolutions |
| **SDXL** | 2023 | Larger U-Net, better text conditioning |

<a id="latent-diffusion"></a>
### Latent Diffusion: The Efficiency Breakthrough

**Stable Diffusion** (Rombach et al., 2022) introduced latent diffusion:

* Images are encoded to a lower-dimensional latent space (e.g., 512×512 → 64×64)
* Diffusion happens in latent space
* Decoder reconstructs high-resolution images

This reduces compute by ~16× while maintaining quality.

<a id="generation-pipeline"></a>
### Image Generation Pipeline

1. **Text Encoding**: CLIP or T5 encodes text prompt
2. **Noise Sampling**: Start with random noise in latent space
3. **Iterative Denoising**: DiT/U-Net removes noise step-by-step
4. **VAE Decoding**: Convert latent back to pixel space

The result: high-quality images from text prompts, with fine-grained control through guidance and conditioning.

---

## References

**DDPM (Foundational Paper)**
* Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS. [arXiv](https://arxiv.org/abs/2006.11239)

**DiT Architecture**
* Peebles, W., & Xie, S. (2023). *Scalable Diffusion Models with Transformers*. ICCV. [arXiv](https://arxiv.org/abs/2212.09748)
* **OpenDiT / PixArt-α**: Open-source implementations on GitHub demonstrating DiT scalability

**Latent Diffusion (LDM)**
* Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR. [arXiv](https://arxiv.org/abs/2112.10752)

---

## Further Reading

* **Diffusion Models Series Part 1**: [From Molecules to Machines](/posts/diffusion-from-molecules-to-machines)
* **Part 3**: [Video Diffusion Fundamentals](/posts/video-diffusion-fundamentals)

---

*This is Part 2 of the Diffusion Models Series. Part 1 covered the foundations of diffusion models. Part 3 will explore video diffusion fundamentals.*

