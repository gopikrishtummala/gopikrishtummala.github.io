---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-25T00:00:00Z
modDatetime: 2025-01-25T00:00:00Z
title: 'Diffusion for Action: Trajectories and Policy'
slug: diffusion-for-action-trajectories-policy
featured: true
draft: false
tags:
  - generative-ai
  - diffusion-models
  - robotics
  - autonomous-vehicles
  - reinforcement-learning
  - embodied-ai
description: 'How diffusion models predict action sequences instead of pixels. Covers Diffusion Policy, world models for robotics, and connecting diffusion to reinforcement learning for autonomous systems.'
track: Robotics
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
    <a href="/posts/pre-training-post-training-video-diffusion" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 5: Pre-Training & Post-Training</a>
    <a href="/posts/diffusion-for-action-trajectories-policy" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Part 6: Diffusion for Action</a>
    <a href="/posts/modern-video-models-sora-veo-opensora" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 7: Modern Models & Motion</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Part 6: Diffusion for Action</strong> — Trajectories and policy</div>
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
      <li><a href="#from-pixels-to-actions">From Pixels to Actions: The Bridge</a></li>
      <li><a href="#diffusion-policy">Diffusion Policy: Predicting Action Sequences</a></li>
      <li><a href="#world-models">World Models: Generating Future States</a></li>
      <li><a href="#planning">Diffusion for Planning and Prediction</a></li>
      <li><a href="#robotics-applications">Robotics Applications</a></li>
      <li><a href="#autonomous-vehicles">Autonomous Vehicles and L4/L5 Systems</a></li>
    </ul>
  </nav>
</div>

---

<a id="from-pixels-to-actions"></a>
## From Pixels to Actions: The Bridge

So far, we've seen diffusion models generate **images** and **videos** — visual content. But what if we want to generate **actions** — sequences of motor commands for a robot or autonomous vehicle?

This is where diffusion models connect to **robotics, autonomous systems, and embodied AI**.

**The Key Insight:**

Instead of generating pixels $x \in \mathbb{R}^{H \times W \times 3}$, we generate **action trajectories**:

$$
\tau = [a_1, a_2, \ldots, a_T]
$$

Where each $a_t$ is an action (e.g., joint angles, velocities, steering commands).

The same diffusion process that learns to reverse noise into images can learn to reverse noise into **plausible action sequences**.

---

<a id="diffusion-policy"></a>
## Diffusion Policy: Predicting Action Sequences

**Diffusion Policy** (Chi et al., 2023) applies diffusion models to robot control:

### The Problem

Traditional robot policies output a single action $a_t$ given the current state $s_t$. But many tasks require **multi-step reasoning** — the robot must plan a sequence of actions.

**Example:** A robot arm picking up a cup needs to:
1. Move toward the cup
2. Open gripper
3. Position around cup
4. Close gripper
5. Lift

A single-action policy struggles with this; it needs to see the full sequence.

### The Solution: Diffusion over Action Sequences

Diffusion Policy generates **action sequences** (trajectories) instead of single actions:

$$
\tau = [a_{t+1}, a_{t+2}, \ldots, a_{t+H}]
$$

Where $H$ is the **horizon** (e.g., 16 steps into the future).

**The Diffusion Process:**

1. **Forward**: Add noise to a demonstration trajectory until it becomes random actions
2. **Reverse**: Learn to denoise random actions back into a valid trajectory
3. **Conditioning**: Condition on the current observation (camera image, sensor data)

**Mathematical Formulation:**

The model learns:

$$
p(\tau | o_t) = \text{DiffusionModel}(\tau, o_t)
$$

Where:
* $\tau$ is the action trajectory
* $o_t$ is the current observation (image, state)

**Key Advantage:** Diffusion naturally handles **multi-modal action distributions**. If there are multiple valid ways to complete a task, diffusion can generate diverse trajectories, unlike deterministic policies.

### Why Diffusion Works for Actions

**1. Multi-Modality:**
* There are often multiple valid action sequences for a task
* Diffusion models excel at capturing multi-modal distributions
* Unlike deterministic policies, they can explore diverse solutions

**2. Smoothness:**
* Actions should change smoothly over time (no sudden jerky movements)
* Diffusion's iterative denoising naturally produces smooth trajectories
* The noise schedule enforces temporal smoothness

**3. Constraint Satisfaction:**
* Robot actions must satisfy physical constraints (joint limits, collision avoidance)
* Diffusion can be guided to satisfy constraints through conditioning

---

<a id="world-models"></a>
## World Models: Generating Future States

**World Models** use diffusion to predict **future states** of the environment, not just actions.

### The Concept

Instead of generating pixels or actions, generate **future observations**:

$$
p(o_{t+1}, o_{t+2}, \ldots, o_{t+H} | o_t, a_t)
$$

Where:
* $o_t$ is the current observation (camera image, LiDAR, etc.)
* $a_t$ is the action taken
* $o_{t+1:H}$ are predicted future observations

### Application: Training Planning Agents

**Use Case:** Train a planning agent for autonomous driving:

1. **Collect data**: Record driving videos with actions (steering, acceleration)
2. **Train world model**: Use a small video diffusion model to predict the next 5 seconds of driving
3. **Train planner**: Use the world model to simulate future scenarios, train a planner that avoids collisions

**Why This Works:**

* The world model learns the **dynamics** of the environment
* The planner can "imagine" consequences of actions without real-world trial-and-error
* This is **model-based reinforcement learning** with learned dynamics

### Mathematical Formulation

The world model learns:

$$
p(o_{t+1:H} | o_t, a_{t:H}) = \text{VideoDiffusion}(o_{t+1:H}, o_t, a_{t:H})
$$

This is essentially a **conditional video diffusion model** where:
* The condition is the current observation $o_t$ and actions $a_{t:H}$
* The output is future observations $o_{t+1:H}$

**Key Insight:** Video diffusion models implicitly learn physics. By training on real driving data, the model learns how scenes evolve — cars move, pedestrians cross, lights change. This learned physics can be used for planning.

---

<a id="planning"></a>
## Diffusion for Planning and Prediction

### Trajectory Prediction

In autonomous vehicles, predicting other agents' trajectories is critical:

**Problem:** Given the current scene, predict where other vehicles/pedestrians will be in the next 5 seconds.

**Solution:** Use diffusion to generate **multiple plausible trajectories**:

$$
p(\tau_{\text{other}} | o_t) = \text{DiffusionModel}(\tau_{\text{other}}, o_t)
$$

Where $\tau_{\text{other}}$ is the future trajectory of another agent.

**Why Diffusion:**
* Multiple plausible futures (agent could turn left, right, or go straight)
* Diffusion captures this multi-modality naturally
* Can generate diverse, realistic trajectories

### Motion Planning

For the ego vehicle, diffusion can generate **candidate trajectories**:

1. Generate $N$ diverse trajectories using diffusion
2. Score each trajectory (safety, comfort, goal progress)
3. Select the best trajectory
4. Execute the first action, replan

**Advantage over traditional planners:**
* Naturally handles multi-modal scenarios
* Learns from data (doesn't require hand-coded rules)
* Can adapt to complex, real-world situations

---

<a id="robotics-applications"></a>
## Robotics Applications

### Manipulation Tasks

**Diffusion Policy** has shown strong performance on:
* **Pick and place**: Grasping objects and moving them
* **Assembly**: Putting parts together
* **Kitchen tasks**: Opening drawers, using tools

**Why it works:**
* Manipulation requires multi-step sequences
* Diffusion naturally handles the sequential nature
* Can learn from diverse demonstration data

### Mobile Robotics

**Navigation and path planning:**
* Generate diverse paths to a goal
* Avoid obstacles while maintaining smooth motion
* Handle uncertainty in the environment

### Human-Robot Interaction

**Predicting human intent:**
* Use diffusion to predict where a human will move
* Plan robot actions that avoid collisions
* Generate natural, human-like robot motions

---

<a id="autonomous-vehicles"></a>
## Autonomous Vehicles and L4/L5 Systems

### Behavior Prediction

**Critical for L4/L5 autonomy:** Predicting other agents' behavior.

**Diffusion-based prediction:**
* Generate multiple plausible futures for each agent
* Each trajectory has an associated probability
* Planner uses these predictions to make safe decisions

**Mathematical Formulation:**

For each agent $i$:

$$
p(\tau_i | o_t) = \text{DiffusionModel}(\tau_i, o_t, \text{context})
$$

Where context includes:
* Agent type (car, pedestrian, cyclist)
* Road structure
* Traffic rules
* Historical behavior

### Scene Prediction

**Predict future scenes** (not just agent trajectories):

* Use video diffusion to predict the next 5-10 seconds of the scene
* Includes all agents, road structure, lighting changes
* Planner can "simulate" consequences of actions

**Connection to World Models:**
* The scene prediction model is a **world model** for driving
* It learns the dynamics of traffic scenes
* Can be used for planning and safety validation

### Closed-Loop Planning

**The Complete Pipeline:**

1. **Perception**: Process sensor data (cameras, LiDAR) → current state $s_t$
2. **Prediction**: Use diffusion to predict other agents' trajectories
3. **Planning**: Generate candidate ego trajectories using diffusion
4. **Selection**: Score and select best trajectory
5. **Control**: Execute first action, repeat

**Why Diffusion Fits:**
* Handles multi-modal scenarios (multiple valid plans)
* Learns from data (doesn't require hand-coded rules)
* Naturally produces smooth, realistic trajectories

---

## The Connection to Reinforcement Learning

Diffusion models and RL are complementary:

**Traditional RL:**
* Learns policies through trial-and-error
* Requires many interactions with the environment
* Can be sample-inefficient

**Diffusion + RL:**
* **Diffusion Policy**: Learns from demonstrations (imitation learning)
* **World Models**: Learns environment dynamics, enables model-based RL
* **Planning**: Uses diffusion to generate candidate actions

**Hybrid Approach:**
1. Pre-train diffusion policy on demonstrations
2. Fine-tune with RL for specific tasks
3. Use world models for efficient exploration

This combines the **data efficiency** of imitation learning with the **adaptability** of RL.

---

## Summary: Diffusion Beyond Visual Generation

Diffusion models aren't just for images and video — they're a powerful framework for:

* **Action Generation**: Diffusion Policy for robot control
* **State Prediction**: World models for planning
* **Trajectory Prediction**: Multi-modal future prediction
* **Planning**: Generating diverse, realistic action sequences

**The Common Thread:**

All these applications use the same core idea: **learn to reverse a noise process to generate structured sequences** — whether those sequences are pixels, actions, or states.

**Interview Relevance:**

* **Robotics**: How do you plan multi-step actions? → Diffusion Policy
* **Autonomous Vehicles**: How do you predict other agents? → Diffusion-based trajectory prediction
* **System Design**: How do you handle multi-modal scenarios? → Diffusion captures diversity naturally

---

## References

**Diffusion Policy**
* Chi, C., et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. RSS. [arXiv](https://arxiv.org/abs/2303.04137)

**World Models**
* Ha, D., & Schmidhuber, J. (2018). *World Models*. NeurIPS. [arXiv](https://arxiv.org/abs/1803.10122)
* Recent work applying video diffusion to world models for robotics

**Trajectory Prediction**
* Recent research on diffusion-based trajectory prediction for autonomous vehicles (2024-2025)

---

## Further Reading

* **Part 5**: [Pre-Training & Post-Training](/posts/pre-training-post-training-video-diffusion)
* **Part 7**: [Modern Models & Motion](/posts/modern-video-models-sora-veo-opensora)
* **Behavior Prediction**: [Behavior Prediction for Closed-Loop Driving](/posts/behavior-prediction-closed-loop-driving)

---

*This is Part 6 of the Diffusion Models Series. Part 5 covered pre-training and post-training. Part 7 will explore modern models like Sora, Veo, and Open-Sora.*

