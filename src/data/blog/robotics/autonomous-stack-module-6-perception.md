---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-25T00:00:00Z
modDatetime: 2025-02-21T00:00:00Z
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
  - sensor-fusion
  - radar
  - ultrasonics
  - acoustic-sensing
description: 'From pixels to objects: How autonomous vehicles understand their environment. Covers the full sensor suite (camera, LiDAR, radar, ultrasonics, microphones), multi-modal fusion, 2D/3D detection, multi-object tracking, semantic segmentation, BEV perception, and the long-tail challenge.'
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

In [Module 2](/posts/robotics/autonomous-stack-module-2-sensors), we introduced the car's raw senses—cameras, LiDAR, radar, ultrasonics, and microphones. Each provides a different window into reality: pixels, point clouds, Doppler returns, proximity readings, and audio waveforms.

But raw data isn't enough. **Perception** transforms those signals into *meaning*.

The difference:

* "There are 50,000 LiDAR points in front of me" → **raw data**
* "There is a pedestrian 15 meters ahead, walking left at 1.2 m/s" → **perception output**

This transformation—from photons, laser pulses, radio waves, sound waves, and acoustic signals to semantic objects with positions, velocities, and classes—is arguably the most challenging problem in autonomous driving.

---

### Act I: The Perception Pipeline

A modern perception system answers three questions in sequence:

1. **Detection:** What objects exist? Where are they?
2. **Classification:** What *kind* of object is each one?
3. **Tracking:** How do objects move over time?

#### The Sensor Suite

Each sensor modality answers a different question:

| Modality | Primary Contribution | Strengths | Weaknesses | Range |
|----------|---------------------|-----------|------------|-------|
| **Camera** | *"What is it?"* | Rich semantics, texture, color | No native depth, weather-sensitive | Long/near |
| **LiDAR** | *"Where exactly?"* | Precise 3D geometry | Costly, degraded in rain/fog | Long to short |
| **Radar** | *"How fast?"* | Direct velocity, all-weather | Lower spatial resolution | Mid to long |
| **Ultrasonic** | *"How close?"* | Precise at close range, cheap, weather-robust | Very limited range, no velocity | <5–8m |
| **Microphone** | *"What's coming?"* | Non-line-of-sight, weather-agnostic | Limited to audible events | Up to 600m |

While cameras deliver rich semantics and LiDAR provides precise geometry, **radar** answers the critical dynamic question: *How fast is it moving?* Its Doppler-based velocity measurements are instantaneous and reliable even in heavy rain, fog, or dust—conditions where cameras lose contrast and LiDAR points scatter.

For **ultra-short-range tasks**—parking, curb detection, low-speed maneuvering—**ultrasonic sensors** (USS) offer direct, low-cost proximity measurements using high-frequency sound waves. While not central to high-speed perception, they provide reliable close-in distance when depth estimation is unreliable at bumper level.

Beyond vision and ranging, **acoustic sensors (microphones)** add an auditory dimension: detecting and localizing emergency vehicle sirens *before* they enter the camera/LiDAR field of view. Sirens provide non-line-of-sight, weather-agnostic cues—critical for yielding safely around corners or in dense traffic. Production systems (e.g., Waymo's 6th-gen driver with external audio) use microphone arrays with AI to classify siren types and estimate direction.

True perception therefore requires fusing all modalities to build a complete, weather-robust scene understanding (as we'll see in [Module 9](/posts/robotics/autonomous-stack-module-9-foundation-models)'s Sensor Fusion Encoder).

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

Detection is the first step: locate objects in the scene and draw boxes around them. But before we detect, we need to understand how to get sensor data into a common representation—**Bird's Eye View (BEV)**.

#### From Perspective to Bird's Eye View

Cameras see the world in **perspective**: parallel lines converge, objects shrink with distance. This makes planning and fusion difficult. We want a **Bird's Eye View** where distances are metric and geometry is undistorted.

**Inverse Perspective Mapping (IPM): The Geometric Approach**

If we know the camera's height and angle, we can "unwarp" the image onto a ground plane:

$$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} \propto H^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

Where $H$ is the homography matrix encoding the camera's pose relative to a flat ground plane ($Z=0$).

**The Trap:** IPM assumes flat ground. Hills, dips, and speed bumps break the geometry—a hill "looks" farther away than it is.

**Neural BEV: Learning to Project**

Modern systems use learned depth estimation to handle non-flat terrain:

| Approach | How It Works | Pros | Cons |
|----------|--------------|------|------|
| **LSS (Lift, Splat, Shoot)** | Predict depth distribution per pixel, lift to 3D, splat to BEV | Handles arbitrary geometry | Requires depth supervision |
| **BEVFormer** | Transformer queries BEV grid, attends to camera features | No explicit depth needed | Compute-heavy |
| **BEVDet** | Depth-aware feature lifting | Balanced speed/accuracy | Sensitive to depth errors |

**The Result:** A unified BEV representation where all sensors can contribute—cameras provide semantics, LiDAR provides geometry, radar provides velocity—all in the same metric coordinate frame.

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

#### Radar-Based Detection and Velocity Estimation

Radar is often overlooked in perception tutorials, but it's **non-negotiable for production systems**. Here's why:

**What Radar Provides:**

1. **Direct Radial Velocity:** Via Doppler shift, radar measures how fast an object approaches or recedes—no multi-frame differencing needed.
2. **Long-Range Detection:** Up to 200–250 m, ideal for highway scenarios where you need early warning of fast-closing vehicles.
3. **All-Weather Operation:** Radio waves penetrate rain, fog, snow, and dust where optical sensors degrade.

**Modern Radar Formats:**

| Type | Output | Resolution | Use Case |
|------|--------|------------|----------|
| **Traditional** | Object list (position, velocity, RCS) | ~2° azimuth | ADAS, simple tracking |
| **4D Imaging** | Dense point cloud + velocity | ~1° azimuth, elevation | Full 3D perception, approaching LiDAR-like |

**4D imaging radars** (with elevation resolution and dense returns) are narrowing the gap with LiDAR. Some stacks (e.g., Tesla's camera-radar approach) use radar pillarization similar to PointPillars—applying PointNet-style encoders to radar tensors.

**Challenges:**

* **Lower Spatial Resolution:** Traditional radars can't resolve fine object shapes; guardrails and road signs create clutter.
* **Multi-Path Reflections:** Radio waves bounce off surfaces, creating "ghost" detections.
* **Deep Learning on Radar:** Less mature than camera/LiDAR; radar-specific networks (RadarNet, CRAFT) are emerging but not yet standard.

**Mitigation:** Radar works best as a *complementary* modality—its velocity and all-weather strengths compensate for camera/LiDAR weaknesses rather than replacing them.

#### Multi-Modal Fusion: The Full Picture

Production systems go beyond camera-LiDAR: **all modalities are fused** into a unified scene representation.

**Why Fusion Matters**

| Sensor | What It Contributes | What It Lacks |
|--------|---------------------|---------------|
| **Camera** | Rich semantics (class, color, text) | Depth, velocity |
| **LiDAR** | Precise 3D geometry | Semantics, velocity |
| **Radar** | Direct velocity, all-weather | Resolution, classification |

No single sensor provides everything. Fusion combines their strengths.

**The Data Association Problem**

Before fusing, you must **match observations across modalities**. The camera sees a car; the radar sees a moving object—are they the same?

Techniques:
- **Frustum Association:** Project camera detection into 3D cone, find radar/LiDAR points inside
- **Hungarian Matching:** Optimize global assignment minimizing distance/appearance cost
- **Learned Association:** Train networks to predict matches from features

**Fusion Architecture Comparison**

| Approach | Where Fusion Happens | Synergy | Traceability | Use Case |
|----------|---------------------|---------|--------------|----------|
| **Late Fusion** | Merge detection boxes | Low | Easy to debug | Validation, fallback |
| **Mid Fusion** | Merge encoded features | High | Requires tooling | Primary production path |
| **Early Fusion** | Merge raw sensor data | Highest | Very hard | Research, rarely deployed |

**Mid-Fusion Architecture (Production Standard)**

Waymo's **Sensor Fusion Encoder** is the canonical example:

```
Camera Images → CNN → Camera Features
LiDAR Points → PointNet → LiDAR Features  
Radar Returns → RadarNet → Radar Features
                    │
                    ▼
            Project to BEV Space
                    │
                    ▼
         Cross-Modal Attention
    (dynamically weight by confidence)
                    │
                    ▼
         Unified Detection Head
                    │
                    ▼
    3D Boxes + Velocity + Class + Uncertainty
```

**Dynamic Weighting by Conditions:**
- In fog: Upweight radar (penetrates), downweight LiDAR (scatters)
- In darkness: Upweight LiDAR/radar, downweight cameras
- In clear weather: All contribute equally

**The Traceability Trade-off:** Mid-fusion entangles sensor contributions. When something goes wrong, tracing the error to a specific sensor requires XAI tooling (attention maps, gradient attribution). But the accuracy gains justify the debugging complexity.

See [Module 9](/posts/robotics/autonomous-stack-module-9-foundation-models) for how foundation models extend this with semantic reasoning.

#### Ultrasonic Sensors: The Close-Range Specialists

Ultrasonic sensors (USS) emit high-frequency sound pulses and measure time-of-flight for distance. They excel at **very short ranges (0.2–5m)** with cm-level accuracy and robustness to lighting/weather.

**Production Use Cases:**

* **Parking assist:** Detecting poles, curbs, parked cars during low-speed maneuvers
* **Collision avoidance:** Final-meter proximity alerts during tight navigation
* **Curb detection:** Precise distance for parallel parking and valet operations

**Historical Context:** Tesla's vehicles (pre-2023) used 12 ultrasonic sensors around bumpers for parking assist. The transition to pure **Tesla Vision** (camera-only) demonstrates that deep learning can replace USS for some tasks—but many stacks retain ultrasonics for redundancy in tight spaces where cameras struggle with close proximity distortion.

**Fusion Tip:** Ultrasonic readings are often projected into BEV or fed directly to a low-speed planner, bypassing full perception for simple proximity alerts. This creates a fast, safety-critical path for stopping before contact.

**Limitations:** Very limited range (<8m), no velocity information, narrow beam patterns. USS are **complementary** to the primary perception stack, not replacements.

#### Acoustic Detection: Hearing the Unseen

Microphones enable perception of events that vision *cannot* detect—most critically, **emergency vehicle sirens**.

**Why Acoustics Matter:**

* **Non-line-of-sight:** Sirens are audible around corners, behind buildings, or through dense traffic—often seconds before visual contact
* **Weather-agnostic:** Sound waves penetrate fog, rain, and dust where cameras/LiDAR degrade
* **Legal requirement:** Vehicles must yield to emergency responders; early detection enables safer pull-over maneuvers

**System Configurations:**

| Configuration | Capability | Example |
|---------------|------------|---------|
| **Single in-cabin mic** | Siren presence detection | Consumer ADAS (Cerence EVD) |
| **3–4 external mics** | Direction + distance estimation | Waymo 6th-gen, Tensor Robocar |
| **Full array (roof/corners)** | Precise bearing, multi-sound classification | Fraunhofer "Hearing Car" |

**Processing Pipeline:**

1. **Spectral Analysis:** Convert audio to spectrograms (time-frequency representation)
2. **Classification:** CNN or transformer identifies siren patterns (>99% accuracy with noise filtering)
3. **Localization:** Time-difference-of-arrival (TDOA) across multiple mics estimates bearing
4. **Fusion:** Output as high-priority transient agent: "Siren approaching from rear-left at ~200m"

**Production Examples:**

* **Waymo:** External audio detection integrated into 6th-generation driver for siren recognition
* **Cerence EVD:** Deployed in BMW Level 3 vehicles; detects 1,500+ siren variants globally, up to 600m range
* **Fraunhofer Hearing Car (2025):** Extends to horns, pedestrian voices, brake squeals for broader hazard awareness

**Fusion Role:** Audio detections fuse into the object list as transient high-priority agents, prompting the planner to pull over *even if the emergency vehicle isn't yet visible*.

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

#### Radar's Role in Tracking

Radar provides strong motion cues that improve tracking:

* **Direct Velocity Measurement:** Reduces association ambiguity (e.g., distinguishing crossing vehicles by their different Doppler signatures).
* **Kalman Filter Enhancement:** Many trackers incorporate radar velocity directly into the state vector $(x, y, \dot{x}, \dot{y})$, lowering prediction uncertainty.
* **Occlusion Resilience:** Radar can "see through" certain visual occlusions (e.g., detecting a vehicle behind another via different Doppler returns).

In practice, radar measurements are fused with camera/LiDAR detections before or during tracking, providing more stable tracks in high-speed scenarios.

#### Specialized Modality Contributions

**Ultrasonics in Tracking:** For low-speed scenarios, ultrasonic measurements can stabilize tracks of nearby static objects (e.g., reducing drift in bumper proximity), especially when visual/LiDAR tracks are occluded or low-confidence.

**Acoustics in Tracking:** Audio cues from sirens can initialize or stabilize tracks for transient high-priority objects (e.g., fast-approaching emergency vehicles), reducing reliance on delayed visual confirmation. The bearing estimate from microphone arrays provides directional priors before the object enters the camera FOV.

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

5. **Radar as Safety Net:** Many long-tail scenarios involve adverse weather (heavy fog occluding a pedestrian, rain degrading camera visibility). Radar maintains velocity estimates when vision fails—if something is moving toward you, radar will see it even if cameras don't.

6. **Ultrasonics for Short-Range Tails:** Tight parking garages, construction cones at 1m, or unseen curbs benefit from ultrasonic fallback—providing conservative distance when vision or LiDAR confidence drops in cluttered/low-light environments.

7. **Acoustic Perception for Occluded Events:** Audio shines in long-tail scenarios like occluded emergency vehicles (siren behind buildings/traffic) or dense urban fog. Systems like Fraunhofer's Hearing Car extend this to horns, pedestrian voices, or brake squeals—hazards that are *heard before seen*.

8. **Conservative Fallbacks:** When uncertain, assume the worst. Slow down, increase following distance, prepare to stop.

---

### Act VI: The Perception → Prediction Interface

Perception doesn't exist in isolation. Its output feeds [Module 7 (Prediction)](/posts/robotics/autonomous-stack-module-7-prediction).

#### What Prediction Needs

For each object, prediction requires:

| Field | Why It Matters |
|-------|----------------|
| **Position** | Starting point for trajectory forecasting |
| **Velocity** | Constant-velocity baseline (often fused from radar Doppler + multi-frame optical flow) |
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

Special Outputs:
  - Ultrasonic proximity alerts (low-speed planner bypass)
  - Audio-derived siren events (bearing, priority flag)

Prediction Input:
  - Object list
  - HD Map (lane graph, semantics)
  - Ego state (position, velocity, intent)
  - Emergency vehicle alerts (from acoustic detection)
```

**The Critical Point:** Perception errors propagate. If you misclassify a pedestrian as a cyclist, prediction will use the wrong motion model. If your position estimate is off by 1 meter, prediction starts from the wrong place.

For audio-derived events (like approaching sirens), the prediction module must anticipate yielding behavior—pulling over even if the emergency vehicle track is incomplete or uncertain.

This is why perception accuracy—across *all* modalities—is non-negotiable for safety.

---

### Summary: The Perception Stack

| Component | Input | Output |
|-----------|-------|--------|
| **2D Detection** | Camera images | 2D bounding boxes |
| **3D Detection** | LiDAR point clouds | 3D bounding boxes |
| **Radar Detection** | Radar returns | Object list with velocity |
| **Ultrasonic Detection** | USS time-of-flight | Close-range proximity alerts |
| **Acoustic Detection** | Microphone arrays | Siren events with bearing |
| **Fusion** | All sensor features | Unified 3D detections with velocity |
| **Tracking** | Fused detections over time | Object tracks with IDs |
| **Segmentation** | Images or BEV | Per-pixel class labels |

**The Pipeline:**

```
Raw Sensors (Camera + LiDAR + Radar + USS + Mics)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
Multi-Modal Fusion              Special Paths
    │                           (USS → Low-Speed Planner)
    ▼                           (Audio → Emergency Override)
Detection (Where? How fast?)           │
    │                                  │
    ▼                                  │
Classification (What?)                 │
    │                                  │
    ▼                                  │
Tracking (Connect over time)           │
    │                                  │
    ▼                                  ▼
Object List + Alerts → Prediction → Planning
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
* *CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection (WACV 2021)*
* *RadarNet: Exploiting Radar for Robust Perception of Dynamic Objects (ECCV 2020)*
* *Cerence EVD: Emergency Vehicle Detection for ADAS (Production System, 2024)*
* *Fraunhofer "Hearing Car": Acoustic Perception for Autonomous Driving (2025)*

---

**Previous:** [Module 5 — Mapping](/posts/robotics/autonomous-stack-module-5-mapping)

**Next:** [Module 7 — Prediction: The Fortune Teller](/posts/robotics/autonomous-stack-module-7-prediction)
