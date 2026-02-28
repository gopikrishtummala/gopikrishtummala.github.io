---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-25T00:00:00Z
modDatetime: 2025-01-25T00:00:00Z
title: 'Module 06: Perception — Seeing the World'
slug: autonomous-stack-module-6-perception
featured: true
draft: false
tags:
  - autonomous-vehicles
  - robotics
  - perception
  - object-detection
  - tracking
  - deep-learning
  - computer-vision
description: 'From pixels to objects: How autonomous vehicles understand their environment. Covers 2D/3D detection, multi-object tracking, semantic segmentation, BEV perception, and the long-tail challenge.'
track: Robotics
difficulty: Advanced
interview_relevance:
  - System Design
  - Theory
  - ML-Infra
estimated_read_time: 35
---

*By Gopi Krishna Tummala*

---

<div class="series-nav" style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">The Ghost in the Machine — Building an Autonomous Stack</div>
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;">
    <a href="/posts/robotics/autonomous-stack-module-1-architecture" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 1: Architecture</a>
    <a href="/posts/robotics/autonomous-stack-module-2-sensors" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 2: Sensors</a>
    <a href="/posts/robotics/autonomous-stack-module-3-calibration" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 3: Calibration</a>
    <a href="/posts/robotics/autonomous-stack-module-4-localization" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 4: Localization</a>
    <a href="/posts/robotics/autonomous-stack-module-5-mapping" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 5: Mapping</a>
    <a href="/posts/robotics/autonomous-stack-module-6-perception" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Module 6: Perception</a>
    <a href="/posts/robotics/autonomous-stack-module-7-prediction" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 7: Prediction</a>
    <a href="/posts/robotics/autonomous-stack-module-8-planning" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 8: Planning</a>
    <a href="/posts/robotics/autonomous-stack-module-9-foundation-models" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 9: Foundation Models</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Module 6: Perception</strong> — Seeing the World</div>
</div>

---

### The Story: From Pixels to Understanding

So far, we've built the car's **body** (sensors), **proprioception** (calibration), **spatial awareness** (localization), and **memory** (mapping).

Now comes the hard part: **Perception**—the ability to understand what's actually happening in the world.

Perception is where raw sensor data becomes *meaning*. It's the difference between:

* "There are 50,000 LiDAR points in front of me"
* "There is a pedestrian 15 meters ahead, walking left at 1.2 m/s"

This transformation—from photons and laser pulses to semantic objects with positions, velocities, and classes—is arguably the most challenging problem in autonomous driving.

---

### Act I: The Perception Pipeline

A modern perception system answers three questions in sequence:

1. **Detection:** What objects exist? Where are they?
2. **Classification:** What *kind* of object is each one?
3. **Tracking:** How do objects move over time?

#### The Output: The Object List

The output of perception is an **Object List**—a structured representation of every relevant entity in the scene:

```
Object 1:
  ID: 42
  Class: Vehicle
  Position: (15.2, -3.1, 0.0) meters
  Dimensions: (4.5, 1.8, 1.5) meters (L, W, H)
  Orientation: 12° from ego heading
  Velocity: (8.2, 0.3, 0.0) m/s
  Confidence: 0.94
  
Object 2:
  ID: 17
  Class: Pedestrian
  Position: (8.1, 5.2, 0.0) meters
  ...
```

This object list feeds directly into [Module 7 (Prediction)](/posts/robotics/autonomous-stack-module-7-prediction) and [Module 8 (Planning)](/posts/robotics/autonomous-stack-module-8-planning).

---

### Act II: Object Detection (Finding Things)

Detection is the first step: locate objects in the scene and draw boxes around them.

#### 2D Detection (Images)

In 2D, we detect objects in camera images. The output is a set of **bounding boxes**: rectangles that tightly enclose each object.

**The Evolution:**

| Era | Method | Speed | Accuracy |
|-----|--------|-------|----------|
| 2012 | Sliding Window + HOG | ~1 FPS | Low |
| 2015 | R-CNN (Regions + CNN) | ~0.1 FPS | High |
| 2016 | YOLO (You Only Look Once) | 45 FPS | Medium |
| 2020 | DETR (Transformers) | 30 FPS | High |
| 2024 | RT-DETR (Real-Time) | 100+ FPS | High |

**YOLO: The Speed Revolution**

YOLO treats detection as a single regression problem. Instead of scanning the image with sliding windows, it:

1. Divides the image into an $S \times S$ grid
2. Each cell predicts $B$ bounding boxes with confidence
3. Each cell predicts $C$ class probabilities
4. One forward pass = all detections

$$\text{Output} = S \times S \times (B \times 5 + C)$$

Where each box has 5 values: $(x, y, w, h, \text{confidence})$.

**The Trade-off:** YOLO is fast but struggles with small objects (each grid cell only predicts a few boxes). Modern variants (YOLOv8, YOLO-World) address this with multi-scale features.

#### 3D Detection (LiDAR)

Cameras give you 2D boxes. For driving, you need 3D: where is the object in the world?

LiDAR point clouds are inherently 3D. The challenge: point clouds are **sparse and unordered**.

**PointPillars: The AV Workhorse**

PointPillars (2019) became the dominant LiDAR detector because it's fast and accurate:

1. **Pillarization:** Divide the ground plane into a grid. Stack all points in each cell into a "pillar."
2. **PointNet Encoding:** Use a small neural network to encode each pillar into a feature vector.
3. **Pseudo-Image:** Arrange pillar features into a 2D BEV image.
4. **2D Detection:** Apply standard 2D detection heads (like YOLO) to the pseudo-image.

$$\text{Point Cloud} \xrightarrow{\text{Pillarize}} \text{Pillars} \xrightarrow{\text{PointNet}} \text{BEV Features} \xrightarrow{\text{CNN}} \text{3D Boxes}$$

**Why It Works:** By converting 3D to pseudo-2D, you can use mature 2D convolutions instead of expensive 3D operations.

#### Camera-LiDAR Fusion

Neither sensor is perfect alone:

* **Camera:** Rich semantics (color, texture), poor depth
* **LiDAR:** Precise depth, sparse appearance

**Fusion Strategy 1: Late Fusion**

Run separate detectors on each modality. Merge detections at the box level.

```
Camera → 2D Boxes → Project to 3D
LiDAR → 3D Boxes
Merge (IoU matching, confidence weighting)
```

**Fusion Strategy 2: Early/Mid Fusion**

Combine features before detection. This is the approach used in Waymo's Sensor Fusion Encoder ([Module 9](/posts/robotics/autonomous-stack-module-9-foundation-models)).

```
Camera → CNN Features
LiDAR → PointNet Features
Fuse in BEV space → Unified 3D Boxes
```

Mid-fusion yields higher accuracy but is harder to debug (see traceability discussion in Module 9).

---

### Act III: Multi-Object Tracking (Connecting Detections Over Time)

Detection gives you objects in a single frame. But driving requires **temporal consistency**: you need to know that "Object 42" in frame $t$ is the same car as "Object 42" in frame $t-1$.

This is **Multi-Object Tracking (MOT)**.

#### The Association Problem

At each timestep, you have:
* **Tracks:** Objects you've been tracking (with IDs, histories)
* **Detections:** New observations (no IDs)

You must decide: Which detection matches which track?

#### SORT: Simple Online and Realtime Tracking

SORT (2016) is elegant in its simplicity:

1. **Predict:** Use a Kalman Filter to predict where each track will be.
2. **Associate:** Match predictions to detections using Hungarian algorithm (minimize total IoU cost).
3. **Update:** Update matched tracks with new observations.
4. **Create/Delete:** Start new tracks for unmatched detections; delete tracks that go unmatched too long.

$$\text{Cost}(i, j) = 1 - \text{IoU}(\text{Track}_i, \text{Detection}_j)$$

**Limitation:** Pure SORT uses only position. If two cars cross paths, it can swap their IDs (the "ID switch" problem).

#### DeepSORT: Adding Appearance

DeepSORT (2017) adds a **Re-ID network**—a CNN that extracts an appearance embedding for each detection.

$$\text{Cost}(i, j) = \lambda \cdot d_{\text{motion}}(i, j) + (1 - \lambda) \cdot d_{\text{appearance}}(i, j)$$

Now, even if two cars have similar positions, their different appearances prevent ID swaps.

#### ByteTrack: Using Low-Confidence Detections

ByteTrack (2021) improved tracking by a simple insight: don't throw away low-confidence detections.

Standard trackers only use detections above a threshold (e.g., confidence > 0.5). ByteTrack:

1. First associates high-confidence detections
2. Then associates remaining tracks with low-confidence detections

This recovers partially occluded objects that detectors are uncertain about.

---

### Act IV: Semantic Segmentation (Understanding Every Pixel)

Bounding boxes tell you where objects are. But what about the *background*?

**Semantic Segmentation** labels every pixel with a class:

* Road
* Sidewalk
* Building
* Vegetation
* Sky
* Vehicle
* Pedestrian
* ...

#### Why It Matters for Driving

The planner needs to know:
* Where can I drive? (road, not sidewalk)
* Where is the curb? (boundary between drivable and not)
* What's ahead? (construction barrier vs. shadow)

#### The Architecture: Encoder-Decoder

Segmentation networks use an **encoder-decoder** structure:

1. **Encoder:** Downsample the image, extract features (ResNet, EfficientNet)
2. **Decoder:** Upsample back to full resolution, predict class per pixel

**U-Net** added **skip connections**—direct links from encoder to decoder at each scale—preserving fine details crucial for precise boundaries.

$$\text{Output} = H \times W \times C$$

Where $C$ is the number of classes.

#### BEV Semantic Segmentation

For driving, **Bird's Eye View (BEV)** segmentation is more useful than camera-view:

* Distances are metric (not distorted by perspective)
* Directly usable by planner (no projection needed)
* Handles occlusions better (can reason about occluded areas)

**BEVFormer** (2022) and similar models learn to project camera features into BEV space using transformers, then segment in BEV directly.

---

### Act V: The Long Tail Problem

Here's the uncomfortable truth about perception: **the easy cases are easy, and the hard cases are nearly impossible**.

#### The 99-1 Split

* **99% of the time:** Detection works great. Cars, pedestrians, cyclists—all clearly visible, correctly classified.
* **1% of the time:** Edge cases. A mattress on the highway. A person in a dinosaur costume. A wheelchair user with a flag. A car covered in mirrors.

That 1% is where crashes happen.

#### Why Is the Long Tail So Hard?

**Data Imbalance:** Your training set has millions of "normal car" examples but maybe 3 examples of "mattress on highway." The model learns the common cases.

**Distribution Shift:** The real world has infinite diversity. No matter how much data you collect, you'll encounter something new.

**Confidence Miscalibration:** Models are often *confidently wrong* on edge cases. They don't say "I don't know"—they say "That's definitely a tumbleweed" when it's actually a small child.

#### Mitigation Strategies

1. **Data Augmentation:** Synthetic generation of rare scenarios (simulation, neural rendering).

2. **Open-Vocabulary Detection:** Models like YOLO-World can detect objects from text descriptions, even if never seen in training.

3. **Uncertainty Estimation:** Train models to output calibrated confidence. Flag low-confidence detections for special handling.

4. **Foundation Models:** Vision-language models ([Module 9](/posts/robotics/autonomous-stack-module-9-foundation-models)) bring world knowledge from pre-training, helping with never-seen objects.

5. **Conservative Fallbacks:** When uncertain, assume the worst. Slow down, increase following distance, prepare to stop.

---

### Act VI: The Perception → Prediction Interface

Perception doesn't exist in isolation. Its output feeds [Module 7 (Prediction)](/posts/robotics/autonomous-stack-module-7-prediction).

#### What Prediction Needs

For each object, prediction requires:

| Field | Why It Matters |
|-------|----------------|
| **Position** | Starting point for trajectory forecasting |
| **Velocity** | Constant-velocity baseline |
| **Heading** | Direction of motion |
| **Class** | Different classes move differently (cars vs. pedestrians) |
| **Track History** | Past trajectory constrains future possibilities |
| **Uncertainty** | High-uncertainty objects need more conservative handling |

#### The Handoff

```
Perception Output (per object):
  - 3D bounding box (x, y, z, l, w, h, θ)
  - Velocity (vx, vy, vz)
  - Classification + confidence
  - Track ID
  - Track history (past N positions)
  - Covariance matrix (uncertainty)

Prediction Input:
  - Object list
  - HD Map (lane graph, semantics)
  - Ego state (position, velocity, intent)
```

**The Critical Point:** Perception errors propagate. If you misclassify a pedestrian as a cyclist, prediction will use the wrong motion model. If your position estimate is off by 1 meter, prediction starts from the wrong place.

This is why perception accuracy is non-negotiable for safety.

---

### Summary: The Perception Stack

| Component | Input | Output |
|-----------|-------|--------|
| **2D Detection** | Camera images | 2D bounding boxes |
| **3D Detection** | LiDAR point clouds | 3D bounding boxes |
| **Fusion** | Camera + LiDAR features | Unified 3D detections |
| **Tracking** | Detections over time | Object tracks with IDs |
| **Segmentation** | Images or BEV | Per-pixel class labels |

**The Pipeline:**

```
Raw Sensors
    │
    ▼
Detection (Where are objects?)
    │
    ▼
Classification (What are they?)
    │
    ▼
Tracking (How do they move?)
    │
    ▼
Object List → Prediction → Planning
```

---

### Graduate Assignment: Tracking Under Occlusion

**Task:**

Design a tracking strategy for handling temporary occlusions.

1. **Scenario:** You're tracking a pedestrian (Track ID: 17) who walks behind a parked van. For 2 seconds, no detector sees them. Then they emerge on the other side.

2. **Question 1:** Using a Kalman Filter with constant velocity model, predict where the pedestrian will be after 2 seconds of occlusion. Initial position: $(5.0, 3.0)$ m, velocity: $(0.0, 1.2)$ m/s.

3. **Question 2:** How does prediction uncertainty grow during occlusion? If initial position variance is $\sigma^2 = 0.1 m^2$ and process noise is $Q = 0.05 m^2/s^2$, what is the variance after 2 seconds?

4. **Question 3:** When the pedestrian reappears at $(5.1, 5.3)$ m, should you:
   - (a) Associate with Track 17?
   - (b) Create a new Track 18?
   - Use Mahalanobis distance to decide. Show your work.

5. **Analysis:** What happens if another pedestrian (Track 23) also emerges near the same location? How would you handle ambiguous associations?

**Further Reading:**

* *PointPillars: Fast Encoders for Object Detection from Point Clouds (CVPR 2019)*
* *DETR: End-to-End Object Detection with Transformers (ECCV 2020)*
* *ByteTrack: Multi-Object Tracking by Associating Every Detection Box (ECCV 2022)*
* *BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images (ECCV 2022)*

---

**Previous:** [Module 5 — Mapping](/posts/robotics/autonomous-stack-module-5-mapping)

**Next:** [Module 7 — Prediction: The Fortune Teller](/posts/robotics/autonomous-stack-module-7-prediction)
