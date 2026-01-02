---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-25T00:00:00Z
modDatetime: 2025-01-25T00:00:00Z
title: 'Pre-Training & Post-Training: Building and Aligning Video Diffusion Models'
slug: pre-training-post-training-video-diffusion
featured: true
draft: false
tags:
  - generative-ai
  - video-generation
  - diffusion-models
  - machine-learning
  - alignment
  - rlhf
description: 'How video diffusion models are built through pre-training and aligned through post-training. Covers the billion-frame training problem, DPO, RLHF, and the complete training pipeline.'
track: GenAI Systems
difficulty: Advanced
interview_relevance:
  - Theory
  - System Design
  - ML-Infra
estimated_read_time: 20
---

*By Gopi Krishna Tummala*

---

<div class="series-nav" style="background: linear-gradient(135deg, #6366f1 0%, #9333ea 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Diffusion Models Series</div>
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;">
    <a href="/posts/diffusion-from-molecules-to-machines" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 1: Foundations</a>
    <a href="/posts/image-diffusion-models-unet-to-dit" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 2: Image Diffusion</a>
    <a href="/posts/sampling-guidance-diffusion-models" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 3: Sampling & Guidance</a>
    <a href="/posts/video-diffusion-fundamentals" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 4: Video Fundamentals</a>
    <a href="/posts/pre-training-post-training-video-diffusion" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Part 5: Pre-Training & Post-Training</a>
    <a href="/posts/diffusion-for-action-trajectories-policy" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 6: Diffusion for Action</a>
    <a href="/posts/modern-video-models-sora-veo-opensora" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 7: Modern Models & Motion</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Part 5: Pre-Training & Post-Training</strong> — Building and aligning video models</div>
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
      <li><a href="#pre-training">Pre-Training: Learning the Grammar of the World</a></li>
      <li><a href="#pre-training-components">Key Components of Pre-Training</a></li>
      <li><a href="#training-data">Training Data: The Billion-Frame Problem</a></li>
      <li><a href="#recaptioning">Recaptioning: The Data Engine</a></li>
      <li><a href="#pre-training-challenges">Pre-Training Challenges and Tradeoffs</a></li>
      <li><a href="#post-training">Post-Training: Alignment and Human Preferences</a></li>
      <li><a href="#post-training-methods">Post-Training Methods</a></li>
      <li><a href="#dpo">Direct Preference Optimization (DPO)</a></li>
      <li><a href="#post-training-challenges">Post-Training Challenges</a></li>
    </ul>
  </nav>
</div>

---

<a id="pre-training"></a>
## Pre-Training: Learning the Grammar of the World

Pre-training is like teaching a child the *grammar of the world* — the model sees millions of images and video clips, and learns the "language" of how pixels, patches, and motion behave.

It's not just about "make pretty video"; it's about learning the distribution of real-world spatio-temporal phenomena so that the model can later be *directed* via prompts.

<a id="pre-training-components"></a>
### Key Components of Pre-Training

**Large-Scale Data:**
* Videos and images of variable resolutions, durations, and aspect ratios
* As seen in Sora's technical report: unified representation for images & video with variable duration/ratio
* The model must learn to handle diverse content types

**Latent Compression:**
* Videos are compressed into a smaller latent space (both spatially & temporally) before diffusion
* Sora's approach: "spacetime patches" as tokens — video is broken into patches that span both space and time
* This reduces computational requirements while preserving essential information

**Transformer Backbone:**
* The denoising network is a transformer over spatio-temporal tokens
* Diffusion loss operates on patch-based latent space
* The model learns to reverse the noise process in this compressed representation

**Mathematical Foundation:**

The pre-training objective is to learn the reverse diffusion process:

$$
\mathcal{L} = \mathbb{E}_{z, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, c)\|^2 \right]
$$

Where:
* $z$ is the latent representation of video
* $\epsilon$ is the noise to predict
* $t$ is the diffusion timestep
* $c$ is conditioning (e.g., text prompts)

<a id="training-data"></a>
### Training Data: The Billion-Frame Problem

#### What the Leading 2025 Video Models Use

| Model                     | Year | Frames      | Notes                                      |
| ------------------------- | ---- | ----------- | ------------------------------------------ |
| **HunyuanVideo**          | 2024 | ~1B         | Strongest open-source text-to-video (2024) |
| **Wan 2.2**               | 2025 | ~12B        | Uses aesthetic + cinematic scoring         |
| **Open-Sora 2** (open-source) | 2025 | ~4B         | Fully open pipeline, detailed technical report |
| **Pika 1.5 (commercial)** | 2024 | undisclosed | High-quality proprietary dataset           |

#### Data Quality Requirements

New datasets lean heavily on:

* **Scene description consistency**: Captions accurately describe what's happening
* **Temporal captions**: "at 1s, camera pans left…" — describing actions over time
* **Action-rich clips**: Sports, wildlife, driving — clips with clear motion
* **Cinematic metadata**: Shot types, lenses, lighting — professional filmmaking knowledge

<a id="recaptioning"></a>
#### Recaptioning: The Data Engine (The Missing Link)

**The Problem:** Raw data from the internet is noisy and poorly labeled. You might find:
* Images labeled "IMG_001.jpg" or "holiday 2012"
* Videos with generic descriptions like "nice view" or "cat video"
* Alt text that's completely wrong or missing

If you train on bad labels, you get a model that ignores prompts. The model never learns the connection between words like "vibrant," "calm," or "silhouette" and their visual meanings.

**The Analogy:** Imagine trying to learn what a "sunset" looks like, but the teacher only shows you photos labeled "holiday 2012" or "nice view." You'd never learn the connection between the word "sunset" and the orange sky.

**The Fix (Recaptioning):** Before training the image/video generator, researchers use a *different* AI (a Vision-Language Model like GPT-4V or CLIP) that is already smart to look at every training image/video and write a detailed, accurate description.

**Example:**
* **Original caption**: "holiday 2012"
* **Recaptioned**: "A vibrant orange sunset over a calm ocean with silhouette palm trees, warm golden light reflecting on the water, tropical beach scene"

**The Result:** The image/video generator is now trained on these "perfect" synthetic captions. It learns exactly what "vibrant," "calm," and "silhouette" mean visually.

**Key Takeaway:** **Better captions > More data.** This is the secret sauce behind DALL-E 3 and Sora's ability to follow complex instructions.

**How It Works:**

1. **Pre-trained Vision-Language Model**: Use a model like GPT-4V that can understand images/videos and generate detailed descriptions
2. **Batch Recaptioning**: Process all training data through the VLM to generate high-quality captions
3. **Training on Synthetic Captions**: Train the diffusion model on these recaptioned pairs instead of original noisy captions

**Production Impact:** This is why modern models (DALL-E 3, Sora, Veo) can follow complex, multi-part prompts. They were trained on captions that actually describe what's in the image/video, not generic filenames or poor alt text.

**Interview Insight:** When asked "How do you make a model follow prompts better?", the answer is often "better training data" — specifically, recaptioning with high-quality vision-language models. This is more important than model architecture improvements.

#### Framewise Aesthetic Reward Models (FARM, 2025)

A new reward function for RLHF on video aesthetic quality:

$$
R = \sum_{t=1}^T \text{Aesthetic}(x_t) + \lambda \cdot \text{Temporal\_Coherence}(x_{1:T})
$$

This rewards:

1. **Frame-level quality**: Each frame is visually appealing
2. **Temporal coherence**: Frames flow smoothly together

The challenge: **balancing aesthetic quality with temporal consistency**. A model that generates beautiful individual frames but flickers between them is useless.

#### Data Curation Pipeline

Modern video datasets go through:

1. **Web scraping**: Billions of video-text pairs from the internet
2. **Quality filtering**: Remove low-resolution, corrupted, or irrelevant videos
3. **Caption generation**: Use vision-language models to generate detailed captions
4. **Aesthetic scoring**: Rank videos by visual quality
5. **Temporal annotation**: Label actions, camera movements, scene changes
6. **Deduplication**: Remove near-duplicate clips

The result: a curated dataset where each clip is:
* High quality
* Well-described
* Temporally rich
* Aesthetically pleasing

<a id="pre-training-challenges"></a>
### Pre-Training Challenges and Tradeoffs

**Scale Requirements:**
* How much compute & data is needed? Modern models train on billions of frames
* Training costs can reach millions of dollars in compute
* Requires massive distributed training infrastructure

**Representational Capacity:**
* Balancing spatial detail vs temporal coherence
* Higher resolution means more parameters and compute
* Longer videos require more memory and temporal modeling capacity

**Data Diversity:**
* Ensuring the model sees enough variation in movement, scene types, camera angles
* Avoiding bias toward common patterns (e.g., certain camera movements, scene compositions)
* Handling edge cases: rare motions, unusual perspectives, complex interactions

**Efficiency vs Quality:**
* Latent compression reduces compute but may lose fine details
* Temporal downsampling speeds training but limits motion fidelity
* Hierarchical generation (keyframes + interpolation) trades quality for speed

---

<a id="post-training"></a>
## Post-Training: Alignment and Human Preferences

After pre-training, the raw diffusion model is *powerful but unaligned*. It might generate motion that's physically weird, or video that's misaligned with user intent. Post-training is how we teach the model to *behave*.

Think of it like giving the child not just grammar but *style guides* — what we actually want them to write, ethically and aesthetically.

**The Core Distinction:**

* **Pre-training** teaches **what the world looks like** — the statistical distribution of video data
* **Post-training** teaches **what humans *want* to see** — alignment with preferences, safety, and controllability

<a id="post-training-methods"></a>
### Post-Training Methods

**1. Supervised Fine-Tuning (SFT):**
* Training on prompt–video (or video + caption) pairs to align with desired outputs
* Improves prompt following and style consistency
* Typically uses a smaller, high-quality curated dataset

**2. Preference-Based Alignment (RLHF / DPO):**
* Humans rank generated videos; the model is trained to prefer higher-ranked ones
* **RLHF**: Requires training a separate reward model, then using reinforcement learning
* **DPO**: Directly optimizes preferences without a reward model (more stable, dominant in 2025)

**3. Safety & Moderation Layers:**
* Content filters to prevent harmful or inappropriate content
* Watermarking and detection systems for synthetic content
* Content provenance metadata for tracking generated media

<a id="dpo"></a>
### Mathematical Foundation: Direct Preference Optimization (DPO)

Recent research has successfully applied DPO to video generation (HuViDPO, Flow-DPO). The preference-ranking loss:

$$
\mathcal{L}_{\text{DPO}} = -\log\left(\frac{\exp(\pi_\theta(x^+))}{\exp(\pi_\theta(x^+)) + \exp(\pi_\theta(x^-))}\right)
$$

Where:
* $x^+$ is a preferred video (rated higher by humans)
* $x^-$ is a less preferred video
* $\pi_\theta$ is the model's probability of generating that video

DPO has become *the* dominant alignment technique in 2025 because it:

* Doesn't require training a separate reward model
* Directly optimizes for human preferences
* Is more stable than RLHF

### Cinematic Reward Models

Some models (OpenAI Sora successors, proprietary) also use **"Cinematic Reward Models"** which grade:

* **Shot composition**: Rule of thirds, leading lines, framing
* **Color grading**: Consistent color palette, mood
* **Motion smoothness**: No jitter, natural camera movement
* **Camera trajectory realism**: Camera moves like a real camera operator would

This is why modern models suddenly produce near-Hollywood-level videos.

The model learns not just to generate video, but to generate *cinematic* video — video that looks like it was shot by a professional filmmaker.

### The Alignment Process

1. **Collect preferences**: Show humans pairs of videos, ask which is better
2. **Train reward model**: Learn to predict human preferences
3. **Optimize policy**: Use RLHF or DPO to align model with preferences
4. **Iterate**: Repeat with new data, refine preferences

The result: models that generate videos humans actually want to watch.

<a id="post-training-challenges"></a>
### Post-Training Challenges

**Compute Constraints:**
* Very large video models are hard to fine-tune because of compute + data requirements
* Full model fine-tuning may be impractical; adapter-based methods are common

**Physical Realism:**
* Post-training for *physical realism* (motion, causality) is still under research
* Models may generate physically implausible motion even after alignment
* Artifact detection remains an active area (e.g., Simple Visual Artifact Detection in Sora-generated Videos)

**Balancing Tradeoffs:**
* Alignment may reduce creative diversity
* Safety filters may be overly conservative
* Quality vs. controllability: more control may reduce output quality

---

## References

**Training Data & Datasets**
* **Open-Sora 2.0**: Comprehensive open-source project detailing hierarchical data pyramid, multi-stage training, and data filtering. [GitHub](https://github.com/PKU-Alignment/Open-Sora-2)
* **WebVid-2M / Kinetics / UCF-101**: Public datasets for video action recognition and T2V benchmarking
* **Wan 2.2**: Research on aesthetic and cinematic scoring in data curation pipelines

**Post-Training & Alignment**
* **HuViDPO / Flow-DPO**: First successful applications of Direct Preference Optimization (DPO) to Text-to-Video generation. [arXiv](https://arxiv.org/abs/2406.12321)
* **Improving Video Generation with Human Feedback**: Systematic pipeline using human feedback with multi-dimensional video reward models. [arXiv](https://arxiv.org/abs/2406.12321)
* **HuggingFace TRL Library**: Open-source implementations of DPO, PPO (RLHF), and alignment methods for Transformer-based models. [Documentation](https://huggingface.co/docs/trl)

---

## Further Reading

* **Part 4**: [Video Diffusion Fundamentals](/posts/video-diffusion-fundamentals)
* **Part 6**: [Diffusion for Action](/posts/diffusion-for-action-trajectories-policy)

---

*This is Part 5 of the Diffusion Models Series. Part 4 covered video diffusion fundamentals. Part 6 will explore diffusion for robotics and action planning.*

