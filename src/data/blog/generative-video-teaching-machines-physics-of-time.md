---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-25T00:00:00Z
modDatetime: 2025-01-25T00:00:00Z
title: 'From Images to Video: Diffusion Models in Generative AI'
slug: generative-video-teaching-machines-physics-of-time
featured: true
draft: false
tags:
  - generative-ai
  - video-generation
  - diffusion-models
  - computer-vision
  - transformers
  - machine-learning
  - deep-learning
description: 'A comprehensive exploration of diffusion models for images and video. From DiT architectures to temporal attention, motion fields, and the billion-frame training problem — how modern AI generates visual content.'
track: GenAI Systems
difficulty: Advanced
interview_relevance:
  - Theory
  - System Design
  - ML-Infra
estimated_read_time: 45
---

*By Gopi Krishna Tummala*

---

<div class="series-nav" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Diffusion Models Series</div>
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;">
    <a href="/posts/diffusion-from-molecules-to-machines" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 1: From Molecules to Machines</a>
    <a href="/posts/generative-video-teaching-machines-physics-of-time" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Part 2: Images to Video</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Part 2: From Images to Video</strong> — Diffusion models for visual generation</div>
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
      <li><a href="#image-diffusion">1. Image Diffusion Models: From U-Net to DiT</a></li>
      <li><a href="#video-diffusion">2. Video Diffusion Models: The Temporal Challenge</a></li>
      <li><a href="#why-video-is-harder">2.1. Why Video is Harder Than Images</a></li>
      <li><a href="#dit-revolution">2.2. The DiT Revolution: Transformers Replace U-Nets</a></li>
      <li><a href="#diffusion-video">2.3. Diffusion for Video: Intuition → Math</a></li>
      <li><a href="#learning-motion">2.4. How Models Learn Motion: Geometry, Optical Flow, and Diffusion Fields</a></li>
      <li><a href="#training-data">2.5. Training Data: The Billion-Frame Problem</a></li>
      <li><a href="#post-training">2.6. Post-Training: How Models Learn Taste, Art, and Cinematics</a></li>
      <li><a href="#putting-together">3. Putting It All Together</a></li>
    </ul>
  </nav>
</div>

---

<a id="image-diffusion"></a>
## 1. Image Diffusion Models: From U-Net to DiT

Diffusion models revolutionized image generation by learning to reverse a noise process. The journey from U-Net-based architectures to Transformer-based models (DiT) represents a fundamental shift in how we approach generative modeling.

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

### The DiT Revolution: Scalable Transformers

**Diffusion Transformers (DiT)** (Peebles & Xie, 2023) replaced U-Nets with Vision Transformers, enabling better scaling:

1. **Patch Embedding**: Image is split into patches (e.g., 16×16 pixels)
2. **Transformer Blocks**: Self-attention processes all patches globally
3. **Conditional Generation**: Cross-attention with text embeddings

The key advantage: **global receptive field** from the start, rather than building it through stacked convolutions.

$$
\text{Attention}(Q,K,V) = \text{softmax}\Big( \frac{QK^\top}{\sqrt{d}} \Big)V
$$

### Image Diffusion Architecture Evolution

| Architecture | Year | Key Innovation |
| ----------- | ---- | -------------- |
| **DDPM** | 2020 | U-Net denoiser, simple noise schedule |
| **Stable Diffusion** | 2022 | Latent diffusion (VAE encoder/decoder) |
| **DiT** | 2023 | Pure Transformer, no convolutions |
| **SDXL** | 2023 | Larger U-Net, better text conditioning |

### Latent Diffusion: The Efficiency Breakthrough

**Stable Diffusion** (Rombach et al., 2022) introduced latent diffusion:

* Images are encoded to a lower-dimensional latent space (e.g., 512×512 → 64×64)
* Diffusion happens in latent space
* Decoder reconstructs high-resolution images

This reduces compute by ~16× while maintaining quality.

### Image Generation Pipeline

1. **Text Encoding**: CLIP or T5 encodes text prompt
2. **Noise Sampling**: Start with random noise in latent space
3. **Iterative Denoising**: DiT/U-Net removes noise step-by-step
4. **VAE Decoding**: Convert latent back to pixel space

The result: high-quality images from text prompts, with fine-grained control through guidance and conditioning.

---

<a id="video-diffusion"></a>
## 2. Video Diffusion Models: The Temporal Challenge

Video generation extends image diffusion to the temporal dimension, introducing new challenges and architectural innovations.

<a id="why-video-is-harder"></a>
### 2.1. Why Video is Harder Than Images

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

### The Mathematical Challenge

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

### 2024–2025 Research Consensus

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
### 2.2. The DiT Revolution: Transformers Replace U-Nets

### Why U-Net Fails at Video Scaling

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

### DiT → V-DiT → AsymmDiT

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

### The Architecture in Practice

In a typical V-DiT block:

1. **Patch Embedding**: Video is split into 3D patches (height × width × time)
2. **Spatial Self-Attention**: Patches within the same frame attend to each other
3. **Temporal Self-Attention**: Patches across frames attend to each other
4. **Cross-Attention**: Text prompts condition the generation
5. **Feed-Forward**: Standard MLP layers

The key insight: **separating spatial and temporal attention allows the model to learn different types of structure independently**, then combine them.

---

<a id="diffusion-video"></a>
### 2.3. Diffusion for Video: Intuition → Math

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

### The Temporal Consistency Trick

The trick:

Noise is added *independently* to each frame, but the denoiser must *jointly* reconstruct all frames with temporal consistency.

This forces the model to learn temporal structure because that's the only way to solve the puzzle.

If the model tries to denoise each frame independently, it will produce flickering, inconsistent motion. The only way to generate smooth video is to learn the temporal dependencies.

### Modern Video Diffusion Scale

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

<a id="learning-motion"></a>
### 2.4. How Models Learn Motion: Geometry, Optical Flow, and Diffusion Fields

Recent research (2024–2025) shows a shift:

**Models now explicitly learn motion fields, not just pixels.**

### FlowVid 2.0 (CVPR 2024)

Uses **motion priors** learned from optical flow to stabilize animations (FlowVid 2.0, CVPR 2024).

The model learns to predict optical flow:

$$
\mathcal{L}_{\text{flow}} = \big\Vert f_\theta(x_t) - \hat{f}_{\text{optical}}(x_t)\big\Vert_1
$$

Where $f_\theta$ is the predicted flow field and $\hat{f}_{\text{optical}}$ is the ground-truth optical flow.

This ensures that:

* Objects move smoothly
* Motion is physically plausible
* Temporal consistency is maintained

### DynamiCrafter 2 (NeurIPS 2024)

Learns **scene geometry** as a latent NeRF-like volume (DynamiCrafter 2, NeurIPS 2024):

$$
x(t) = V_\theta(r(t))
$$

Where $V_\theta$ is a 3D volume representation and $r(t)$ is a ray through the scene.

This gives the model:

* **3D understanding**: Objects have depth and structure
* **View consistency**: The same object looks correct from different angles
* **Motion in 3D space**: Objects move through 3D, not just 2D pixels

### Diffusion Video Fields (CVPR 2025)

Represents video as a 4D continuous function:

$$
v = F_\theta(x, y, t, \sigma)
$$

Where:
* $(x, y)$ are spatial coordinates
* $t$ is time
* $\sigma$ is the noise level (diffusion timestep)

This gives better:

* **Identity preservation**: Objects maintain their appearance across frames
* **Motion stability**: Smooth, continuous motion
* **Controllability**: Easy to manipulate camera movement, object motion

The key insight: **representing video as a continuous function allows the model to interpolate smoothly between frames**, rather than generating discrete frames independently.

---

<a id="training-data"></a>
### 2.5. Training Data: The Billion-Frame Problem

### What the Leading 2025 Video Models Use

| Model                     | Year | Frames      | Notes                                      |
| ------------------------- | ---- | ----------- | ------------------------------------------ |
| **HunyuanVideo**          | 2024 | ~1B         | Strongest open-source text-to-video (2024) |
| **Wan 2.2**               | 2025 | ~12B        | Uses aesthetic + cinematic scoring         |
| **Open-Sora 2** (open-source) | 2025 | ~4B         | Fully open pipeline, detailed technical report |
| **Pika 1.5 (commercial)** | 2024 | undisclosed | High-quality proprietary dataset           |

### Data Quality Requirements

New datasets lean heavily on:

* **Scene description consistency**: Captions accurately describe what's happening
* **Temporal captions**: "at 1s, camera pans left…" — describing actions over time
* **Action-rich clips**: Sports, wildlife, driving — clips with clear motion
* **Cinematic metadata**: Shot types, lenses, lighting — professional filmmaking knowledge

### Framewise Aesthetic Reward Models (FARM, 2025)

A new reward function for RLHF on video aesthetic quality:

$$
R = \sum_{t=1}^T \text{Aesthetic}(x_t) + \lambda \cdot \text{Temporal\_Coherence}(x_{1:T})
$$

This rewards:

1. **Frame-level quality**: Each frame is visually appealing
2. **Temporal coherence**: Frames flow smoothly together

The challenge: **balancing aesthetic quality with temporal consistency**. A model that generates beautiful individual frames but flickers between them is useless.

### Data Curation Pipeline

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

---

<a id="post-training"></a>
### 2.6. Post-Training: How Models Learn Taste, Art, and Cinematics

Pre-training teaches **what the world looks like**.

Post-training teaches **what humans *want* to see**.

### 2025 State-of-the-Art Post-Training Methods

1. **Video-SFT (Supervised Fine-Tuning)**: Instruction + style conditioning
2. **Video-RLHF**: Reinforcement Learning from Human Feedback
3. **DPO-V (Direct Preference Optimization for Video)**: Directly optimize for human preferences
4. **V-QPA (Video Quality Preference Alignment)**: Align model outputs with quality metrics

### Direct Preference Optimization (DPO)

Recent research has successfully applied DPO to video generation (HuViDPO, Flow-DPO). A typical preference-ranking loss:

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

---

<a id="putting-together"></a>
## 3. Putting It All Together

### The Complete Pipeline

A modern video generation model (2025) works like this:

1. **Pre-training** (billions of frames):
   - Learn the structure of video data
   - Learn temporal dependencies
   - Learn to denoise video latents

2. **Architecture** (DiT/V-DiT/AsymmDiT):
   - Spatial attention for image content
   - Temporal attention for motion
   - Cross-attention for text conditioning

3. **Motion Learning** (explicit motion fields):
   - Optical flow for smooth motion
   - 3D geometry for view consistency
   - Continuous fields for interpolation

4. **Post-training** (human preferences):
   - DPO for preference alignment
   - Cinematic reward models for aesthetics
   - Instruction following for controllability

### The Physics Connection

The remarkable thing: **none of this explicitly programs physics**.

The model learns:
* Objects have momentum (from watching things move)
* Light casts shadows (from watching lighting)
* Water flows downhill (from watching water)
* Camera movement is smooth (from watching camera work)

Not through equations, but through **statistical patterns in billions of frames**.

The model becomes an **amortized physical simulator** — it doesn't solve physics equations, but it has learned to generate videos that satisfy physical laws because those laws are encoded in the training data.

### Why This Matters

This approach to video generation has implications beyond entertainment:

* **Robotics**: Models that understand motion can plan robot trajectories
* **Scientific simulation**: Generate plausible simulations of physical processes
* **Education**: Visualize complex phenomena (fluid dynamics, particle physics)
* **Creative tools**: Enable new forms of artistic expression

The future: models that don't just generate video, but understand the physics underlying motion.

---

## Conclusion

Generative video is one of the most exciting frontiers in AI.

It requires:
* **Massive scale**: Billions of frames, trillions of parameters
* **Novel architectures**: DiT, temporal attention, motion fields
* **Sophisticated training**: Pre-training, post-training, alignment
* **Implicit physics**: Learning physical laws from data

The result: models that can generate videos that are:
* **Visually stunning**: High resolution, cinematic quality
* **Temporally coherent**: Smooth motion, consistent objects
* **Physically plausible**: Motion that makes sense
* **Controllable**: Follow text prompts, user instructions

We're teaching machines the physics of time — not through equations, but through the statistical structure of motion itself.

---

## References

### Image Diffusion Models

**DDPM (Foundational Paper)**
* Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS. [arXiv](https://arxiv.org/abs/2006.11239)

**DiT Architecture**
* Peebles, W., & Xie, S. (2023). *Scalable Diffusion Models with Transformers*. ICCV. [arXiv](https://arxiv.org/abs/2212.09748)
* **OpenDiT / PixArt-α**: Open-source implementations on GitHub demonstrating DiT scalability

**Latent Diffusion (LDM)**
* Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR. [arXiv](https://arxiv.org/abs/2112.10752)

### Video Diffusion Models

**V-DiT / Temporal Attention**
* **Latte / Video DiT**: Early works adapting DiT for video with temporal attention mechanisms
* **Stable Video Diffusion (SVD)**: Demonstrates inflating pre-trained 2D models with temporal layers

**AsymmDiT (Asymmetric DiT)**
* **Mochi 1** (Genmo): Open-source model (Apache 2.0) using Asymmetric Diffusion Transformer to separate spatial vs. temporal attention. [GitHub](https://github.com/genmo-ai/mochi)

**Motion Learning & Optical Flow**
* **FlowVid 2.0** (CVPR 2024): Explicit optical flow priors for temporal coherence. [arXiv](https://arxiv.org/abs/2403.12934)
* **DynamiCrafter 2** (NeurIPS 2024): Learning temporal scene geometry and non-rigid motion. [arXiv](https://arxiv.org/abs/2405.21060)

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

* **Diffusion Models Series Part 1**: [From Molecules to Machines](/posts/diffusion-from-molecules-to-machines)
* **Vision-Language Models**: [Vision-Language Models Explained](/posts/vision-language-models-explained)
* **Physics-Aware Video**: [Physics-Aware Video Diffusion Models](/posts/physics-aware-video-diffusion-models)

---

*This is Part 2 of the Diffusion Models Series. Part 1 covered the foundations of diffusion models. Future parts will explore specialized applications, optimization techniques, and the latest research frontiers.*

