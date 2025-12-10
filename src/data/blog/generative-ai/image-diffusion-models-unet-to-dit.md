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

<div class="series-nav" style="background: linear-gradient(135deg, #6366f1 0%, #9333ea 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Diffusion Models Series</div>
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;">
    <a href="/posts/diffusion-from-molecules-to-machines" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 1: Foundations</a>
    <a href="/posts/image-diffusion-models-unet-to-dit" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Part 2: Image Diffusion</a>
    <a href="/posts/sampling-guidance-diffusion-models" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 3: Sampling & Guidance</a>
    <a href="/posts/video-diffusion-fundamentals" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 4: Video Fundamentals</a>
    <a href="/posts/pre-training-post-training-video-diffusion" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 5: Pre-Training & Post-Training</a>
    <a href="/posts/diffusion-for-action-trajectories-policy" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 6: Diffusion for Action</a>
    <a href="/posts/modern-video-models-sora-veo-opensora" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 7: Modern Models & Motion</a>
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

Early diffusion models like **DDPM** and **Stable Diffusion** used U-Net architectures. But **why** is the U-Net architecture ideal for denoising?

#### Why U-Net for Denoising?

Denoising requires solving a **multiscale problem**:

1. **Global Structure Recognition** (Encoder): The model must understand the high-level content — "Is this a cat or a car?" This requires **compression** through downsampling layers to capture semantic structure.

2. **Fine Detail Reconstruction** (Decoder + Skip Connections): The model must remove noise pixel-by-pixel while preserving sharp edges and textures. This requires **high-resolution detail** that would be lost during compression.

**The U-Net Solution:**

* **Encoder (Downsampling Path)**: Compresses the image through convolutional layers, capturing global structure and context. Think of it as "zooming out" to see the big picture.

* **Decoder (Upsampling Path)**: Reconstructs the image at full resolution, using the learned global structure to guide denoising.

* **Skip Connections**: These are the critical innovation — they carry fine-grained details directly from encoder to decoder, bypassing the compression bottleneck. Like a "highway" that preserves pixel-level information.

**Visual Analogy for Skip Connections:**

Imagine restoring a damaged painting:
- The **encoder** is like stepping back to see the overall composition (global structure)
- The **decoder** is like zooming in to fix individual brushstrokes (local details)
- **Skip connections** are like having a reference photo at full resolution — you can always check the original fine details without losing them through compression

Without skip connections, the network would lose high-frequency details during compression. Without downsampling, it couldn't capture the semantic content needed to guide denoising.

The forward diffusion process:

$$
q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

The reverse denoising process:

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}\big(\mu_\theta(x_t,t), \Sigma_t\big)
$$

Where $\mu_\theta$ is learned by a U-Net that predicts the noise to remove using the loss function:

$$
\mathcal{L} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

<a id="dit-revolution"></a>
### The DiT Revolution: Scalable Transformers

**Diffusion Transformers (DiT)** (Peebles & Xie, 2023) replaced U-Nets with Vision Transformers, enabling better scaling:

1. **Patch Embedding**: Image is split into patches (e.g., 16×16 pixels)
2. **Transformer Blocks**: Self-attention processes all patches globally
3. **Conditional Generation**: Cross-attention with text embeddings

The key advantage: **global receptive field** from the start, rather than building it through stacked convolutions.

#### Patches as Tokens: Treating Images Like Language

**The Paradigm Shift:** Older models (U-Nets) looked at images as a grid of pixels to be convolved. Newer models (Sora, Veo, DiT) treat images like *language*.

**The Analogy:** 
* **U-Nets**: Like reading a book word-by-word, but only seeing nearby words. You struggle to connect a character mentioned on page 1 with their action on page 100.
* **DiT**: Like having the entire book laid out as a single sentence. You can see all "words" (patches) at once and understand how they relate across the entire image.

**Why This Matters:**

U-Nets struggle to "see" things that are far apart in an image. For example:
* A hand on the left side of an image matching a foot on the right (same person, same pose)
* A shadow on the ground matching the object casting it
* Text in one corner matching a logo in another corner

**The Solution (Patches as Tokens):**

We chop the image into little squares called **patches** (like puzzle pieces). We treat these patches exactly like words in a sentence (tokens). This allows the model to use **Transformers** (the brain of ChatGPT).

**The Benefit:** Suddenly, the model understands "context" across the entire image/video at once. This leads to:
* Better consistency in physics (objects don't randomly change)
* Object permanence (things stay the same across frames)
* Global understanding (the model "sees" the whole scene at once)

This is why modern models (Sora, Veo, Stable Diffusion 3) use DiT architectures — they can maintain consistency across large spatial and temporal scales.

$$
\text{Attention}(Q,K,V) = \text{softmax}\Big( \frac{QK^\top}{\sqrt{d}} \Big)V
$$

#### Why Transformers Scale Better: Resolution and Computation

Transformers excel at high-resolution images ($1024 \times 1024$ and up) where U-Nets struggle:

**The Scaling Problem:**
* **U-Nets**: Computation scales with image size through stacked convolutions. For $1024 \times 1024$ images, you need many layers to build a global receptive field, making training and inference expensive.
* **Transformers**: Computation scales more predictably. Self-attention gives every patch access to every other patch in a single layer, regardless of image size.

**Resolution Advantage:**
* At $512 \times 512$: U-Nets work well, but Transformers are competitive
* At $1024 \times 1024$: Transformers become significantly more efficient
* At $2048 \times 2048$: Transformers are the clear choice — U-Nets become computationally prohibitive

This is why modern high-resolution image generation (SDXL, Imagen) uses Transformer-based architectures. The global receptive field isn't just a nice-to-have — it's essential for scaling to production-quality resolutions.

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

#### Latent Diffusion as a System Design Pattern

**Interview Question:** *"How do you make Stable Diffusion fast on consumer hardware?"*

**Answer: Latent Diffusion (LDM).**

This is a critical **System Design** pattern for production GenAI systems:

1. **Problem**: Pixel-space diffusion on $512 \times 512$ images requires massive compute — impractical for consumer GPUs.

2. **Solution**: Compress images to latent space ($64 \times 64$) using a pre-trained VAE encoder, run diffusion in this compressed space, then decode back to pixels.

3. **Tradeoff**: Slight quality loss from compression, but massive speedup (16×) makes it production-viable.

4. **Production Impact**: This is why Stable Diffusion runs on consumer GPUs while pixel-space models require data center infrastructure.

**Key Insight**: The VAE encoder/decoder learns a "visual grammar" — it compresses images into a space that preserves semantic information while discarding pixel-level redundancy. Diffusion in this compressed space is both faster and often produces better results because the model focuses on structure rather than noise.

#### The VAE Bottleneck: The "JPEG" Compression Artifacts of AI

**The Problem:** Latent diffusion works by compressing the image first using a VAE (Variational Autoencoder). But the compressor is imperfect — it's like a "zipping" tool that loses data when compressed too aggressively.

**The Analogy:** Think of the VAE as a JPEG compressor. If you compress a photo too much, you get:
* Blocky artifacts
* Blurred details
* Lost fine textures

The same happens with VAE compression in diffusion models.

**The Symptom:** Have you noticed AI-generated images where:
* Text looks garbled or unreadable?
* Faces look waxy or smoothed out?
* Fine details (like hair strands, fabric textures) are missing?

That's often **not** the diffusion model's fault — it's the *compressor's* fault. The VAE literally "blurred" the details before the diffusion model even started working.

**The Fix:** Modern models address this in several ways:

1. **Larger, Better VAEs**: Models like SDXL and Flux use larger VAE encoders that preserve more detail
2. **Skip Compression for Critical Details**: Some models use a hybrid approach — compress most of the image, but keep text and fine details in pixel space
3. **Better Training**: Training VAEs specifically to preserve important details (like text, faces, fine textures)

**Production Impact:** This is why you see different quality levels across models. A model with a poor VAE will struggle with text and fine details, even if the diffusion model itself is excellent. The VAE is a critical bottleneck that determines the upper bound on image quality.

**Interview Insight:** When asked "Why do AI images sometimes look blurry or have garbled text?", the answer is often the VAE bottleneck, not the diffusion model itself.

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

* **Part 1**: [From Molecules to Machines](/posts/diffusion-from-molecules-to-machines)
* **Part 3**: [Sampling & Guidance](/posts/sampling-guidance-diffusion-models)

---

*This is Part 2 of the Diffusion Models Series. Part 1 covered the foundations of diffusion models. Part 3 will explore sampling and guidance techniques.*

