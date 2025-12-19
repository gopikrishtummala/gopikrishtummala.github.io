---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-25T00:00:00Z
modDatetime: 2025-01-25T00:00:00Z
title: 'Video Diffusion Fundamentals: The Temporal Challenge'
slug: video-diffusion-fundamentals
featured: true
draft: false
tags:
  - generative-ai
  - video-generation
  - diffusion-models
  - computer-vision
  - transformers
  - machine-learning
description: 'Why video is harder than images, the DiT revolution for video, and how diffusion models learn temporal consistency. Covers V-DiT, AsymmDiT, and the mathematical foundations of video generation.'
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
    <a href="/posts/sampling-guidance-diffusion-models" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 3: Sampling & Guidance</a>
    <a href="/posts/video-diffusion-fundamentals" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Part 4: Video Fundamentals</a>
    <a href="/posts/pre-training-post-training-video-diffusion" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 5: Pre-Training & Post-Training</a>
    <a href="/posts/diffusion-for-action-trajectories-policy" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 6: Diffusion for Action</a>
    <a href="/posts/modern-video-models-sora-veo-opensora" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 7: Modern Models & Motion</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Part 4: Video Diffusion Fundamentals</strong> — The temporal challenge</div>
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
      <li><a href="#video-diffusion">Video Diffusion Models: The Temporal Challenge</a></li>
      <li><a href="#why-video-is-harder">Why Video is Harder Than Images</a></li>
      <li><a href="#dit-revolution">The DiT Revolution: Transformers Replace U-Nets</a></li>
      <li><a href="#diffusion-video">Diffusion for Video: Intuition → Math</a></li>
    </ul>
  </nav>
</div>

---

<a id="video-diffusion"></a>
## Video Diffusion Models: The Temporal Challenge

Video generation extends image diffusion to the temporal dimension, introducing new challenges and architectural innovations.

<a id="why-video-is-harder"></a>
### Why Video is Harder Than Images

Imagine you ask an artist to draw a bird.

One frame? Easy.

Now tell the artist:

> "Draw the **same** bird, but flying — for 5 seconds — at 24 frames per second."

Suddenly the problem explodes:

* The wings must flap naturally.
* The body must follow a smooth trajectory.
* The shadows must move correctly.
* The bird cannot randomly change color or species between frames.

What the artist feels here is exactly what video models feel:

**Temporal constraints are like hidden physical laws.**

**Break them once and the illusion collapses.**

#### The Mathematical Challenge

For images, your model learns a distribution:

$$
p(x)
$$

For video, it must learn:

$$
p(x_1, x_2, \ldots, x_T)
$$

Where each frame must satisfy:

$$
x_{t+1} \approx f(x_t) \quad \text{(smooth motion constraint)}
$$

and

$$
(x_t)_{\text{object identity}} = (x_{t+1})_{\text{object identity}}
\quad \text{(identity preservation constraint)}
$$

This is a **high-dimensional Markov process**, except the "transition dynamics" (the physics of how things move) are not given — the model must *learn* them.

#### 2024–2025 Research Consensus

Recent papers (CVPR, NeurIPS, ICLR) hammer this point:

* **"Video Diffusion Models are Amortized Physical Simulators"** (NeurIPS 2024 spotlight)
* **"TempoFlow: Learning Coherent Motion Priors for Video Synthesis"** (CVPR 2025)
* **"DynamiCrafter 2: Learning Temporal Scene Geometry and Non-Rigid Motion"** (NeurIPS 2024)

They all converge on one central idea:

> **To generate believable video, the model must implicitly learn physics.**

> Even if no one tells it Newton's laws.

The model discovers that objects have momentum, that light sources cast consistent shadows, that water flows downhill — not through explicit programming, but through the statistical structure of billions of video frames.

---

<a id="dit-revolution"></a>
### The DiT Revolution: Transformers Replace U-Nets

#### Why U-Net Fails at Video Scaling

U-Nets use convolutions that operate locally:

$$
\text{conv}(x)(i,j) = \sum W_{k} \cdot x(i+k,j+k)
$$

Great for images.

Disastrous for long-range temporal structure.

To model a 10-second clip (240 frames), the receptive field needs to explode.

Transformers solve this by making the receptive field **global**:

$$
\text{Attention}(Q,K,V) = \text{softmax}\Big( \frac{QK^\top}{\sqrt{d}} \Big)V
$$

This is the key: **every patch attends to every other patch**, including across time.

#### DiT → V-DiT → AsymmDiT

Open-source and industry models in 2024–2025 evolved like this:

| Year | Architecture                       | Major Contribution                                                  |
| ---- | ---------------------------------- | ------------------------------------------------------------------- |
| 2023 | **DiT**                            | Replace U-Net with pure ViT for diffusion denoiser                  |
| 2024 | **V-DiT / Video DiT**              | Extend DiT into temporal dimension                                  |
| 2025 | **AsymmDiT / Dual-path Attention** | Separate spatial vs. temporal attention → faster + higher coherence |

**AsymmDiT** (e.g., **Mochi 1** (Genmo, Apache 2.0), Pyramidal Video LDMs from CVPR 2025) uses:

* **Spatial Attention:**

  $$
  \text{Attn}_s = \text{softmax}\Big(\frac{Q_sK_s^\top}{\sqrt{d}}\Big)V_s
  $$

* **Temporal Attention:**

  $$
  \text{Attn}_t = \text{softmax}\Big(\frac{Q_tK_t^\top}{\sqrt{d}}\Big)V_t
  $$

And mixes them with a learned gate:

$$
h = \alpha \cdot \text{Attn}_s + (1-\alpha)\cdot \text{Attn}_t
$$

Why this works:

* **Spatial attention** learns image content
* **Temporal attention** learns object permanence, motion physics
* **A learned α** lets the model gradually shift focus depending on frame structure

This is one of the most powerful "physics proxies" in modern video generation.

#### The Architecture in Practice

In a typical V-DiT block:

1. **Patch Embedding**: Video is split into 3D patches (height × width × time)
2. **Spatial Self-Attention**: Patches within the same frame attend to each other
3. **Temporal Self-Attention**: Patches across frames attend to each other
4. **Cross-Attention**: Text prompts condition the generation
5. **Feed-Forward**: Standard MLP layers

The key insight: **separating spatial and temporal attention allows the model to learn different types of structure independently**, then combine them.

---

<a id="diffusion-video"></a>
### Diffusion for Video: Intuition → Math

Diffusion models learn to reverse a noise process, transforming random noise into structured video content.

Forward process (adding noise):

$$
q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

Reverse process (denoising):

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}\big(\mu_\theta(x_t,t), \Sigma_t\big)
$$

For video, the latent includes time:

$$
x_t \in \mathbb{R}^{H \times W \times F}
$$

with $F$ = number of frames.

#### The Temporal Consistency Trick

The trick:

Noise is added *independently* to each frame, but the denoiser must *jointly* reconstruct all frames with temporal consistency.

This forces the model to learn temporal structure because that's the only way to solve the puzzle.

If the model tries to denoise each frame independently, it will produce flickering, inconsistent motion. The only way to generate smooth video is to learn the temporal dependencies.

#### Modern Video Diffusion Scale

Modern video diffusion datasets (Wan 2.2, HunyuanVideo, **Open-Sora 2** (open-source), VeGa) use up to:

* **1024×1024 resolution**
* **8–24 fps**
* **2–14 seconds per clip**

This is orders of magnitude larger than early video diffusion.

For a 10-second clip at 24fps and 1024×1024 resolution:

$$
\text{Data per clip} = 240 \text{ frames} \times 1024 \times 1024 \times 3 \text{ channels} = 755 \text{ MB}
$$

Training on billions of such clips requires:

* Efficient latent compression (VAE encoders)
* Temporal downsampling strategies
* Hierarchical generation (generate keyframes, then interpolate)

---

## References

**V-DiT / Temporal Attention**
* **Latte / Video DiT**: Early works adapting DiT for video with temporal attention mechanisms
* **Stable Video Diffusion (SVD)**: Demonstrates inflating pre-trained 2D models with temporal layers

**AsymmDiT (Asymmetric DiT)**
* **Mochi 1** (Genmo): Open-source model (Apache 2.0) using Asymmetric Diffusion Transformer to separate spatial vs. temporal attention. [GitHub](https://github.com/genmo-ai/mochi)

---

## Further Reading

* **Part 3**: [Sampling & Guidance](/posts/sampling-guidance-diffusion-models)
* **Part 5**: [Pre-Training & Post-Training](/posts/pre-training-post-training-video-diffusion)

---

*This is Part 4 of the Diffusion Models Series. Part 3 covered sampling and guidance. Part 5 will explore pre-training and post-training pipelines.*

