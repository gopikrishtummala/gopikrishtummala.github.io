---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-25T00:00:00Z
modDatetime: 2025-01-25T00:00:00Z
title: 'Sampling & Guidance: The Dialects of Noise'
slug: sampling-guidance-diffusion-models
featured: true
draft: false
tags:
  - generative-ai
  - diffusion-models
  - computer-vision
  - machine-learning
  - inference
  - optimization
description: 'How to accelerate diffusion sampling and control output quality. Covers DDIM, DPM-Solver, Classifier-Free Guidance (CFG), negative prompting, and inference optimization techniques.'
track: GenAI Systems
difficulty: Advanced
interview_relevance:
  - Theory
  - System Design
  - ML-Infra
estimated_read_time: 18
---

*By Gopi Krishna Tummala*

---

<div class="series-nav" style="background: linear-gradient(135deg, #6366f1 0%, #9333ea 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Diffusion Models Series</div>
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;">
    <a href="/posts/diffusion-from-molecules-to-machines" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 1: Foundations</a>
    <a href="/posts/image-diffusion-models-unet-to-dit" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 2: Image Diffusion</a>
    <a href="/posts/sampling-guidance-diffusion-models" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Part 3: Sampling & Guidance</a>
    <a href="/posts/video-diffusion-fundamentals" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 4: Video Fundamentals</a>
    <a href="/posts/pre-training-post-training-video-diffusion" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 5: Pre-Training & Post-Training</a>
    <a href="/posts/diffusion-for-action-trajectories-policy" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 6: Diffusion for Action</a>
    <a href="/posts/modern-video-models-sora-veo-opensora" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 7: Modern Models & Motion</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Part 3: Sampling & Guidance</strong> — The dialects of noise</div>
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
      <li><a href="#sampling-intro">The Sampling Problem</a></li>
      <li><a href="#ddpm-sampling">DDPM: Stochastic Sampling</a></li>
      <li><a href="#ddim">DDIM: Deterministic Fast Sampling</a></li>
      <li><a href="#dpm-solver">DPM-Solver: High-Order Solvers</a></li>
      <li><a href="#flow-matching">Flow Matching: Straightening the Path</a></li>
      <li><a href="#cfg">Classifier-Free Guidance (CFG)</a></li>
      <li><a href="#negative-prompting">Negative Prompting</a></li>
      <li><a href="#inference-optimization">Inference Optimization</a></li>
    </ul>
  </nav>
</div>

---

<a id="sampling-intro"></a>
## The Sampling Problem

Training a diffusion model requires 1000 denoising steps. But generating a single image with 1000 forward passes is **slow** — often taking 10-30 seconds on consumer hardware.

The challenge: **How do we accelerate sampling without sacrificing quality?**

This is a critical **System Design** problem for production GenAI systems. The answer lies in understanding that not all denoising steps are equally important, and we can use smarter algorithms to skip steps intelligently.

---

<a id="ddpm-sampling"></a>
## DDPM: Stochastic Sampling

The original DDPM uses **stochastic sampling** — each step adds randomness:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

Where $z \sim \mathcal{N}(0, I)$ is random noise.

**Characteristics:**
* Requires 1000 steps for high quality
* Stochastic (random) — same prompt produces different results
* High quality but slow

**When to use:** Research, when quality is paramount, when you have compute to spare.

---

<a id="ddim"></a>
## DDIM: Deterministic Fast Sampling

**DDIM (Denoising Diffusion Implicit Models)** (Song et al., 2020) made a key insight: **you can skip steps deterministically**.

The DDIM update rule:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t)
$$

**Key Properties:**
* **Deterministic**: Same noise seed + prompt = same output (reproducible)
* **Fast**: Can use 20-50 steps instead of 1000
* **Quality**: Maintains quality with 10-20× speedup

**How it works:** DDIM uses a deterministic mapping between noise levels. Instead of following the stochastic path, it takes a direct "shortcut" through the noise schedule.

**Tradeoff:** Slightly less diversity (deterministic), but much faster.

---

<a id="dpm-solver"></a>
## DPM-Solver: High-Order Solvers

**DPM-Solver** (Lu et al., 2022) treats the reverse diffusion as an **ordinary differential equation (ODE)** and uses high-order numerical solvers.

The insight: The reverse process can be written as an ODE:

$$
\frac{dx}{dt} = f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)
$$

DPM-Solver uses **Runge-Kutta methods** (like those in physics simulations) to solve this ODE efficiently.

**Performance:**
* **10-20 steps** for high-quality generation (vs. 1000 for DDPM)
* **50-100× speedup** over DDPM
* Quality matches or exceeds DDPM

**Why it works:** High-order solvers can "look ahead" and make larger, smarter steps through the noise schedule, rather than taking many small steps.

**Production Impact:** This is why modern diffusion models (Stable Diffusion, SDXL) can generate images in 1-2 seconds instead of 30 seconds.

---

<a id="flow-matching"></a>
## Flow Matching: Straightening the Path

**Flow Matching** (also called **Rectified Flow**) is a modern approach that makes diffusion generation much faster and mathematically cleaner. It's used in state-of-the-art models like Stable Diffusion 3, Flux, and Sora.

### The Analogy: Walking in a Straight Line

Imagine you are in a dense forest (Noise) and want to get to your house (Image).

* **Standard Diffusion (DDPM)**: You wander randomly, bumping into trees, slowly finding your way home. It takes 100 steps, and each step is uncertain.

* **Flow Matching**: You draw a straight line on a map from the forest to your house and walk directly along it. It takes 10 steps, and the path is deterministic and efficient.

### Why Standard Diffusion is "Jittery"

Standard diffusion (DDPM) is like a "drunken walk" — it removes noise in a jittery, random path. Each step adds randomness:

$$
x_{t-1} = x_t - \text{noise\_prediction} + \text{random\_noise}
$$

This randomness is necessary to match the training distribution, but it makes the path inefficient.

### How Flow Matching Works

Flow Matching learns a **straight path** from noise to data:

$$
\frac{dx}{dt} = v_\theta(x_t, t)
$$

Where $v_\theta$ is a velocity field that points directly from the current noisy state toward the target image.

**Key Insight:** Instead of learning to remove noise (diffusion), Flow Matching learns to **transport** the noise directly to the image along the shortest path.

### Benefits

1. **Faster Generation**: 4-10 steps instead of 20-50 steps (DPM-Solver) or 1000 steps (DDPM)
2. **Mathematically Cleaner**: No need for complex noise schedules or stochastic sampling
3. **Better Quality**: The straight path often produces higher quality results with fewer artifacts
4. **Deterministic**: Same noise seed produces the same result (unlike stochastic DDPM)

### Why It Matters

Flow Matching represents a fundamental shift in how we think about generative models:
* **Old way**: Remove noise step-by-step (diffusion)
* **New way**: Transport noise directly to data (flow)

This is why modern models (Stable Diffusion 3, Flux) can generate high-quality images in just 4-8 steps — they're following a straight path, not wandering through noise space.

**Production Impact:** Flow Matching enables real-time generation on consumer hardware, making it practical for interactive applications.

---

<a id="cfg"></a>
## Classifier-Free Guidance (CFG): Controlling Output Quality

**Classifier-Free Guidance (CFG)** is how we make diffusion models **follow prompts better** and produce **higher quality outputs**.

### The Problem: Weak Conditioning

Without guidance, conditional diffusion models often ignore the prompt or produce low-quality outputs. The model might generate "a cat" when you ask for "a majestic cat on a throne."

### The Solution: Guidance Scale

CFG uses both **conditional** and **unconditional** predictions:

$$
\hat{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing))
$$

Where:
* $\epsilon_\theta(x_t, t, \varnothing)$ is the **unconditional** prediction (no prompt)
* $\epsilon_\theta(x_t, t, c)$ is the **conditional** prediction (with prompt $c$)
* $w$ is the **guidance scale** (typically 7.5 for Stable Diffusion)

### How Guidance Works

**Intuition:** The unconditional prediction represents "what the model thinks is realistic." The conditional prediction represents "what the model thinks matches the prompt." Guidance **amplifies the difference** between them.

**The Guidance Scale $w$:**
* $w = 1$: No guidance (just conditional prediction)
* $w = 7.5$: Strong guidance (default for Stable Diffusion)
* $w > 15$: Very strong guidance (may produce artifacts, over-saturated colors)

**Mathematical Effect:**

The guidance formula pushes the model toward regions where:
$$
p(x|c) \gg p(x)
$$

That is, regions where the conditional probability is much higher than the unconditional probability — exactly where the prompt is most relevant.

### Production Tuning

**Interview Question:** *"How do you tune guidance scale for production?"*

**Answer:**
1. **Start with $w = 7.5$** (Stable Diffusion default)
2. **Increase** if prompts aren't being followed ($w = 10-12$)
3. **Decrease** if outputs look over-saturated or unnatural ($w = 5-7$)
4. **A/B test** different values for your use case

---

<a id="negative-prompting"></a>
## Negative Prompting: Pushing Away from Unwanted Concepts

**Negative prompting** is a powerful technique: instead of just saying what you want, you also say what you **don't** want.

**Example:**
* **Positive prompt**: "a beautiful landscape"
* **Negative prompt**: "blurry, low quality, distorted, watermark"

### Why Negative Prompting Works

Negative prompting works by **pushing the distribution away** from unwanted concepts.

Mathematically, we can think of it as:

$$
p(x | c_+, c_-) \propto \frac{p(x | c_+)}{p(x | c_-)}
$$

Where:
* $c_+$ is the positive prompt
* $c_-$ is the negative prompt

The model generates samples where the positive prompt probability is **high** and the negative prompt probability is **low**.

### Practical Applications

**Common Negative Prompts:**
* **Quality**: "blurry, low quality, distorted, artifacts"
* **Style**: "cartoon, anime, painting" (if you want photorealistic)
* **Content**: "text, watermark, signature" (to avoid unwanted text)

**Production Tip:** Create a **default negative prompt** for your application that filters common unwanted artifacts. This improves output quality consistently.

---

<a id="inference-optimization"></a>
## Inference Optimization: Making Sampling Production-Ready

### Step Reduction Strategies

**1. Fewer Steps with Better Schedulers:**
* DDPM: 1000 steps
* DDIM: 20-50 steps
* DPM-Solver: 10-20 steps

**2. Adaptive Step Sizing:**
* Use more steps in high-noise regions (early in the process)
* Use fewer steps in low-noise regions (late in the process)

### Model Optimization

**1. Quantization:**
* FP16 instead of FP32: 2× speedup, minimal quality loss
* INT8 quantization: 4× speedup, some quality loss

**2. Model Pruning:**
* Remove redundant attention heads
* Prune less important layers

**3. Caching:**
* Cache text embeddings (CLIP outputs)
* Cache VAE encoder/decoder activations

### System Design Considerations

**Latency vs. Quality Tradeoff:**
* **Real-time applications** (e.g., live image editing): Use DPM-Solver with 10 steps, FP16
* **Batch generation** (e.g., generating 100 images): Can use more steps, higher quality
* **Interactive applications**: Cache embeddings, use quantized models

**Production Architecture:**
1. **Text Encoder**: Cache CLIP embeddings (they don't change per step)
2. **Diffusion Model**: Run denoising steps (the bottleneck)
3. **VAE Decoder**: Decode final latent to pixels (fast, can be parallelized)

---

## Summary: The Sampling & Guidance Toolkit

| Technique | Speedup | Quality | Use Case |
| --------- | ------- | ------- | -------- |
| **DDPM** | 1× (baseline) | Highest | Research, quality-critical |
| **DDIM** | 10-20× | High | Fast generation, deterministic |
| **DPM-Solver** | 50-100× | High | Production, real-time apps |
| **CFG ($w=7.5$)** | N/A | Higher | Standard for all conditional models |
| **Negative Prompting** | N/A | Higher | Filtering unwanted artifacts |

**Production Recommendation:**
* Use **DPM-Solver with 20 steps** for best speed/quality balance
* Set **CFG scale to 7.5-10** depending on prompt adherence needs
* Always use **negative prompts** to filter common artifacts
* **Quantize to FP16** for 2× speedup with minimal quality loss

---

## References

**DDIM (Deterministic Sampling)**
* Song, J., Meng, C., & Ermon, S. (2020). *Denoising Diffusion Implicit Models*. ICLR. [arXiv](https://arxiv.org/abs/2010.02502)

**DPM-Solver (Fast ODE Solvers)**
* Lu, C., et al. (2022). *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps*. NeurIPS. [arXiv](https://arxiv.org/abs/2206.00927)

**Classifier-Free Guidance**
* Ho, J., & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. NeurIPS Workshop. [arXiv](https://arxiv.org/abs/2207.12598)

---

## Further Reading

* **Part 2**: [Image Diffusion Models](/posts/image-diffusion-models-unet-to-dit)
* **Part 4**: [Video Diffusion Fundamentals](/posts/video-diffusion-fundamentals)

---

*This is Part 3 of the Diffusion Models Series. Part 2 covered image diffusion architectures. Part 4 will explore video diffusion fundamentals.*

