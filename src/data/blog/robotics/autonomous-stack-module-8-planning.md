---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-25T00:00:00Z
modDatetime: 2025-01-25T00:00:00Z
title: 'Module 08: The Chess Master — The Art of Planning'
slug: autonomous-stack-module-8-planning
featured: true
draft: false
tags:
  - autonomous-vehicles
  - robotics
  - planning
  - game-theory
  - control-systems
description: 'From perception to action: How autonomous vehicles make decisions. Covers cost functions, game-theoretic planning, and the modular vs. end-to-end debate.'
track: Robotics
difficulty: Advanced
interview_relevance:
  - System Design
  - Theory
  - ML-Infra
estimated_read_time: 30
---

*By Gopi Krishna Tummala*

---

<div class="series-nav" style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">The Ghost in the Machine — Building an Autonomous Stack</div>
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;">
    <a href="/posts/autonomous-stack-module-1-architecture" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 1: Architecture</a>
    <a href="/posts/autonomous-stack-module-2-sensors" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 2: Sensors</a>
    <a href="/posts/autonomous-stack-module-3-calibration" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 3: Calibration</a>
    <a href="/posts/autonomous-stack-module-4-localization" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 4: Localization</a>
    <a href="/posts/autonomous-stack-module-5-mapping" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 5: Mapping</a>
    <a href="/posts/autonomous-stack-module-6-perception" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 6: Perception</a>
    <a href="/posts/autonomous-stack-module-7-prediction" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 7: Prediction</a>
    <a href="/posts/autonomous-stack-module-8-planning" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Module 8: Planning</a>
    <a href="/posts/autonomous-stack-module-9-foundation-models" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 9: Foundation Models</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Module 8: The Chess Master</strong> — The Art of Planning</div>
</div>

---

### The Story: From Seeing to Acting

So far, our car has **Eyes** (Cameras), **Ears** (Radar), and a sense of **Truth** (LiDAR). It knows exactly where the world is.

But "knowing" isn't driving.

* **Perception** says: *"There is a cyclist 12 meters ahead, moving left."*

* **Prediction** says: *"He will likely cross our lane in 2.5 seconds."*

* **The Planner** has to decide: *"Do I brake? Do I swerve? Do I nudge forward to signal intent?"*

If Perception is the visual cortex, the **Planner** is the frontal lobe. It is the module that weighs *Risk* vs. *Reward* 10 times per second.

This post explores how autonomous vehicles transform raw sensor data and predictions into safe, comfortable, and legal driving decisions.

---

### Act I: The Cost Function (How Robots Think)

Humans drive on intuition. Robots drive on math. Specifically, they drive by minimizing a **Cost Function**.

Imagine a transparent "surface" floating over the road.

* **High Cost (Hills):** Hitting a car, driving off-road, jerking the steering wheel (uncomfortable).

* **Low Cost (Valleys):** Staying in the lane center, maintaining 30 mph, driving smoothly.

The Planner's job is to roll a marble through this landscape so it stays in the deepest valley.

$$J_{total} = w_1(\text{Safety}) + w_2(\text{Comfort}) + w_3(\text{Progress}) + w_4(\text{Legality})$$

Where:

* $w_1$: Safety weight (collision avoidance, staying on road)

* $w_2$: Comfort weight (smooth acceleration, gentle turns)

* $w_3$: Progress weight (reaching the destination)

* $w_4$: Legality weight (following traffic rules, speed limits)

**The Tuning Challenge:**

* If $w_1$ is too high, the car never moves (safest option).

* If $w_3$ is too high, the car drives like a maniac to get there faster.

* **The Art of Planning** is tuning these weights ($w$) so the car feels "human."

---

### Act II: The Hard Part (Interactive Planning & Game Theory)

Driving would be easy if no one else was on the road. The problem is **Other Humans**.

#### The "Frozen Robot" Problem

Early self-driving cars treated other cars like moving rocks. They predicted a path and tried to avoid it.

* *Result:* The "Frozen Robot" problem. The car waits endlessly at a 4-way stop because it's terrified of cutting someone off.

* *Why it happens:* The planner sees infinite risk in any action that involves uncertainty about other drivers' intentions.

#### Game Theoretic Planning

Modern planners (like those in Waymo's latest stack) use **Game Theoretic Planning**.

They treat driving as a **Multi-Agent Game**.

* *"If I nudge forward 1 foot, will that other driver slow down?"*

* *"He is driving aggressively (high risk acceptance); I should yield."*

* *"He is hesitant; I should take the right of way."*

This isn't just physics; it's negotiation. The car simulates thousands of "I do X, you do Y" futures every millisecond to find the **Nash Equilibrium**—the move where everyone is least unhappy.

#### Recent Breakthroughs (Waymo Open Dataset Challenge)

The industry focus has shifted heavily here. The **2025 Waymo Open Dataset Challenges** explicitly added an **"Interaction Prediction"** track. The goal isn't just predicting where a car *will* go, but predicting how it will *react* to us.

Winners of the 2024 challenge (like the ModeSeq team) used advanced Transformer models to handle this "social uncertainty."

Here is a video that perfectly illustrates the "Game Theoretic" nature of modern planning—watch how the Waymo vehicle negotiates space with human drivers in a dense, unstructured environment.

[Waymo driverless car navigating chaos in SF](https://www.youtube.com/watch?v=B8R148hFxGw)

This video shows the "Frozen Robot" problem being solved in real-time: notice how the car nudges, waits, and negotiates with pedestrians and aggressive drivers rather than just stopping passively.

---

### Act III: The Great Divergence (Modular vs. End-to-End)

Right now, there is a massive philosophical war in the industry.

#### Team Modular (Waymo, Aurora, Mobileye)

**Philosophy:** "Divide and Conquer."

**How it works:** You have a distinct Perception box, a distinct Prediction box, and a distinct Planner box.

**Pros:**

* **Explainable:** If the car crashes, you know exactly why ("The Planner chose a bad path" vs "The Camera didn't see the truck").

* **Debuggable:** You can test each module independently.

* **Regulatory Friendly:** Easier to certify and validate individual components.

**Cons:**

* **Brittle:** You have to hand-code rules for every weird edge case (e.g., "If a man in a chicken suit is holding a sign, do X").

* **Integration Complexity:** Errors compound across module boundaries.

* **Limited Generalization:** Struggles with scenarios not explicitly programmed.

#### Team End-to-End (Tesla FSD v12, Wayve, Comma.ai)

**Philosophy:** "Muscle Memory."

**How it works:** You replace the entire stack with one giant Neural Network.

* **Input:** Photons (Video). **Output:** Steering control.

**The Shift:** Tesla recently deleted 300,000+ lines of C++ planning code for FSD v12. Instead of writing rules, they feed the network millions of hours of human driving video. The car doesn't "calculate" a cost function; it simply *imitates* what a good human driver would do in that visual context.

**Pros:**

* **Natural Feel:** It feels incredibly natural. It can handle weird situations (like a parade) that no programmer ever wrote a rule for.

* **Generalization:** Learns from data, not explicit rules.

* **Simpler Architecture:** One model instead of many modules.

**Cons:**

* **Black Box:** If it crashes, you don't know why. You can't just "fix the code"; you have to retrain the brain.

* **Data Hungry:** Requires massive amounts of high-quality driving data.

* **Safety Validation:** Harder to prove the system is safe in all scenarios.

#### The Future: Hybrid World Models

The cutting edge (papers like **UniAD** and **VAD** from CVPR 2024) is trying to merge these approaches.

They use deep learning to build a "World Model"—a neural simulation of the future—and then run a planner inside that learned hallucination.

**The Promise:**

* **Neural Intuition:** The world model captures complex, hard-to-code behaviors.

* **Planner Safety:** The planner ensures hard constraints (don't hit things, follow rules).

It is the best of both worlds: the intuition of a neural network with the safety constraints of a planner.

---

### Summary: The Planning Challenge

Planning in autonomous vehicles requires:

1. **Cost Function Design:** Balancing safety, comfort, progress, and legality.

2. **Game Theoretic Reasoning:** Understanding how other agents will react to your actions.

3. **Architectural Choice:** Modular (explainable) vs. End-to-End (natural) vs. Hybrid (best of both).

**The Path Forward:**

* **Better Interaction Models:** Predicting how agents react to ego vehicle actions.

* **Hybrid Architectures:** Combining neural world models with classical planners.

* **Safety Guarantees:** Formal verification methods for end-to-end systems.

---

### Graduate Assignment: The Cost Function Tuning Problem

**Task:**

Design a simple 1D planner for a car approaching a stop sign.

1. **Scenario:** A car is 50m from a stop sign, traveling at 15 m/s.

2. **Cost Function:** $J = w_1 \cdot (\text{distance to stop line})^2 + w_2 \cdot (\text{deceleration})^2 + w_3 \cdot (\text{time to destination})$

3. **Experiment:**
   * Set $w_1 = 1, w_2 = 0.1, w_3 = 0.01$ (safety-focused). What happens?
   * Set $w_1 = 0.1, w_2 = 1, w_3 = 0.01$ (comfort-focused). What happens?
   * Set $w_1 = 0.1, w_2 = 0.1, w_3 = 1$ (progress-focused). What happens?

4. **Analysis:** Which combination feels most "human"? Why?

**Further Reading:**

* *UniAD: Planning-oriented Autonomous Driving (CVPR 2024)*
* *VAD: Vectorized Scene Representation for End-to-End Autonomous Driving (CVPR 2024)*
* *Waymo Open Dataset: Interaction Prediction Challenge*

**Next Up:** [Module 9 — The Unified Brain (Foundation Models)](/posts/autonomous-stack-module-9-foundation-models)

