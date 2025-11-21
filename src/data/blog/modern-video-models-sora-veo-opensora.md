---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-25T00:00:00Z
modDatetime: 2025-01-25T00:00:00Z
title: 'Modern Video Models & Motion: Sora, Veo 3, Open-Sora, and Motion Modeling'
slug: modern-video-models-sora-veo-opensora
featured: true
draft: false
tags:
  - generative-ai
  - video-generation
  - diffusion-models
  - sora
  - computer-vision
  - machine-learning
description: 'Deep dive into state-of-the-art video generation models: Sora, Veo 3, and Open-Sora. Plus motion modeling techniques using optical flow, geometry, and diffusion fields.'
track: GenAI Systems
difficulty: Advanced
interview_relevance:
  - Theory
  - System Design
  - ML-Infra
estimated_read_time: 22
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
    <a href="/posts/pre-training-post-training-video-diffusion" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 5: Pre-Training & Post-Training</a>
    <a href="/posts/diffusion-for-action-trajectories-policy" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 6: Diffusion for Action</a>
    <a href="/posts/modern-video-models-sora-veo-opensora" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Part 7: Modern Models & Motion</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Part 7: Modern Models & Motion</strong> — State-of-the-art video generation</div>
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
      <li><a href="#modern-models">Modern Models: Sora, Veo 3, and Open-Sora</a></li>
      <li><a href="#sora">Sora: World Simulators</a></li>
      <li><a href="#veo">Veo 3: Audio and Motion Control</a></li>
      <li><a href="#open-sora">Open-Sora: Open-Source Alternative</a></li>
      <li><a href="#model-comparison">Model Comparison</a></li>
      <li><a href="#motion-modeling">Motion Modeling: Geometry, Optical Flow, and Diffusion Fields</a></li>
      <li><a href="#flowvid">FlowVid 2.0</a></li>
      <li><a href="#dynamicrafter">DynamiCrafter 2</a></li>
      <li><a href="#diffusion-fields">Diffusion Video Fields</a></li>
      <li><a href="#putting-together">Putting It All Together</a></li>
    </ul>
  </nav>
</div>

---

<a id="modern-models"></a>
## Modern Models: Sora, Veo 3, and Open-Sora

These are not just research toys — they are *world simulators* that demonstrate the state-of-the-art in video generation.

<a id="sora"></a>
### Sora: World Simulators

**OpenAI's Sora** represents a breakthrough in unified video generation:

**Key Innovations:**
* **Spacetime Patches**: Diffusion transformer operates on patches that span both space and time, enabling variable duration and aspect ratios
* **Unified Representation**: Same model handles images and video, with variable resolutions and durations
* **Recaptioning Technique**: Uses DALL·E 3's recaptioning to improve caption quality in training data

**Architecture:**
* Transformer-based denoiser on spacetime patches
* Large-scale pre-training on video + image data
* Post-alignment for prompt-following and safety

**Capabilities:**
* Generate videos up to 60 seconds
* Variable aspect ratios and resolutions
* Strong temporal coherence and object permanence

**Limitations:**
* Physics & causality limitations (OpenAI acknowledges)
* Artifact issues (addressed in research like Sugiyama & Kataoka, 2025)
* Watermarking and copyright concerns
* Access constraints (not fully public)

**On-Device Sora:**
* Research variant optimized for mobile/low compute
* Techniques: Linear Proportional Leap (LPL) to reduce denoising steps
* Temporal Dimension Token Merging (TDTM) for efficiency
* Does not require retraining — works by optimizing inference

<a id="veo"></a>
### Veo 3: Audio and Motion Control

**Google/DeepMind's Veo 3** focuses on high-fidelity video with integrated audio:

**Key Innovations:**
* **Integrated Audio Generation**: Includes lip-sync, environmental sound, and dialogue
* **Motion Control**: Fine-grained control over camera movement, scene dynamics, and object motion
* **Cinematic Quality**: High-fidelity output with professional filmmaking aesthetics

**Architecture:**
* Large-scale pre-training (Google-scale compute and data)
* Post-training for alignment, realism, and audio-visual synchronization
* Motion control through conditioning mechanisms

**Capabilities:**
* High-resolution video generation
* Synchronized audio generation
* Controllable camera and object motion
* Professional-grade cinematic output

**Limitations:**
* Black box: not fully open research
* Access and cost constraints for users
* Balancing audio, visuals, and motion control is computationally intensive

<a id="open-sora"></a>
### Open-Sora: Open-Source Alternative

**Open-Sora** (PKU-Alignment) provides a fully open-source alternative:

**Key Innovations:**
* **Spatial-Temporal Diffusion Transformer (STDiT)**: Decouples spatial and temporal attention for efficiency
* **3D Autoencoder**: Compact video representation in latent space
* **Open Weights + Code**: Fully reproducible pipeline

**Architecture:**
* STDiT backbone with separate spatial/temporal attention
* 3D VAE for latent compression
* Multi-stage training pipeline

**Capabilities:**
* 720p video generation
* ~15 second clip generation
* Open-source and reproducible
* Detailed technical reports

**Limitations:**
* Lower resolution/quality vs. Sora/Veo
* Generation cost for long/high-res video
* Less advanced audio (depending on version)
* Fewer resources for post-training compared to big organizations

<a id="model-comparison"></a>
### Model Comparison

| Model              | Key Strength                          | Pre-Training Scale | Post-Training | Open Source | Best For                          |
| ------------------ | ------------------------------------- | ------------------ | ------------- | ----------- | --------------------------------- |
| **Sora**           | Unified representation, long videos    | Very large         | Extensive     | No          | Research, high-quality generation |
| **Veo 3**          | Audio sync, motion control            | Very large         | Extensive     | No          | Cinematic content, audio-visual   |
| **Open-Sora**      | Reproducibility, open access          | Large (open)       | Limited       | Yes         | Research, education, development   |
| **Mochi 1**        | AsymmDiT architecture                 | Large              | Limited       | Yes (Apache 2.0) | Open-source video generation      |
| **HunyuanVideo**   | Large-scale open model                | ~1B frames         | Limited       | Yes         | Open-source baseline              |

---

<a id="motion-modeling"></a>
## Motion Modeling: Geometry, Optical Flow, and Diffusion Fields

Recent research (2024–2025) shows a shift:

**Models now explicitly learn motion fields, not just pixels.**

Good video models either learn motion explicitly (flow/fields) or implicitly (attention across time). When motion is modeled explicitly, the model gets a physics anchor to hang frames on.

<a id="flowvid"></a>
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

<a id="dynamicrafter"></a>
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

<a id="diffusion-fields"></a>
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

<a id="putting-together"></a>
## Putting It All Together

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

**Modern Models**
* **Sora (OpenAI)**: Video generation models as world simulators. [OpenAI Research](https://openai.com/research/video-generation-models-as-world-simulators)
* **On-device Sora**: Training-free diffusion-based text-to-video for mobile devices. [arXiv](https://arxiv.org/abs/2503.23796)
* **Veo 3**: Google/DeepMind's high-fidelity video generation with audio. [Veo 3](https://3-veo.com/)
* **Open-Sora**: Democratizing efficient video production for all. [arXiv](https://arxiv.org/abs/2412.20404) [GitHub](https://github.com/PKU-Alignment/Open-Sora-2)
* **Simple Visual Artifact Detection in Sora-generated Videos**: Research on detecting artifacts in Sora outputs. [arXiv](https://arxiv.org/abs/2504.21334)
* **Mora**: Enabling generalist video generation via multi-agent framework. [arXiv](https://arxiv.org/abs/2403.13248)

**Motion Learning & Optical Flow**
* **FlowVid 2.0** (CVPR 2024): Explicit optical flow priors for temporal coherence. [arXiv](https://arxiv.org/abs/2403.12934)
* **DynamiCrafter 2** (NeurIPS 2024): Learning temporal scene geometry and non-rigid motion. [arXiv](https://arxiv.org/abs/2405.21060)

---

## Further Reading

* **Part 6**: [Diffusion for Action](/posts/diffusion-for-action-trajectories-policy)
* **Diffusion Models Series Part 1**: [From Molecules to Machines](/posts/diffusion-from-molecules-to-machines)
* **Vision-Language Models**: [Vision-Language Models Explained](/posts/vision-language-models-explained)
* **Physics-Aware Video**: [Physics-Aware Video Diffusion Models](/posts/physics-aware-video-diffusion-models)

---

*This is Part 7 of the Diffusion Models Series, concluding our exploration of image and video diffusion models. The series covers foundations, architectures, training pipelines, robotics applications, and state-of-the-art models.*

