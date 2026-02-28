---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-25T00:00:00Z
modDatetime: 2025-01-25T00:00:00Z
title: 'Module 05: Mapping — The Memory of the Road'
slug: autonomous-stack-module-5-mapping
featured: true
draft: false
tags:
  - autonomous-vehicles
  - robotics
  - mapping
  - hd-maps
  - slam
  - localization
description: 'How autonomous vehicles remember the world. Covers HD maps, lane graphs, semantic layers, offline vs. online mapping, SLAM, and the map-heavy vs. map-light debate.'
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
    <a href="/posts/autonomous-stack-module-5-mapping" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Module 5: Mapping</a>
    <a href="/posts/autonomous-stack-module-6-perception" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 6: Perception</a>
    <a href="/posts/autonomous-stack-module-7-prediction" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 7: Prediction</a>
    <a href="/posts/autonomous-stack-module-8-planning" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 8: Planning</a>
    <a href="/posts/autonomous-stack-module-9-foundation-models" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 9: Foundation Models</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Module 5: Mapping</strong> — The Memory of the Road</div>
</div>

---

### The Story: Why Maps Matter

In [Module 4](/posts/autonomous-stack-module-4-localization), we solved the "Where am I?" problem. The car knows its position to within centimeters using the Kalman Filter's "Blue Line."

But knowing *where* you are is useless without knowing *what's there*.

Imagine waking up in a dark room. You know you're exactly 3.2 meters from the corner—but is that corner a wall, a door, or a cliff? You need a **Map**: a structured memory of the world that tells you what to expect before you even look.

For autonomous vehicles, maps are not just navigation aids. They are **a priori knowledge**—the rules of the game encoded before the game begins.

---

### Act I: What HD Maps Contain

A standard navigation map (Google Maps, Apple Maps) tells you: "Turn left in 300 meters onto Main Street."

An **HD Map** (High-Definition Map) tells you:

* The exact curvature of the turn (spline coefficients)
* The number of lanes and their widths (to 10cm precision)
* Where the stop line is painted
* Which lanes you're legally allowed to drive in
* The height of the curb
* The location of every traffic light, sign, and crosswalk

HD maps are **centimeter-accurate, semantically rich representations** of the driving environment.

#### The Three Layers

HD maps are typically organized into layers:

| Layer | Contents | Resolution | Update Frequency |
|-------|----------|------------|------------------|
| **Geometric** | 3D point clouds, ground surface mesh, curb heights | ~10cm | Months |
| **Semantic** | Lane boundaries, traffic signs, crosswalks, speed limits | ~10cm | Weeks |
| **Topological** | Lane graph (connectivity), allowed maneuvers, traffic rules | Logical | Days |

**The Geometric Layer** is the "shape" of the world—what the LiDAR would see if you drove through with no traffic.

**The Semantic Layer** adds *meaning*—this line is a lane boundary, that pole is a traffic light.

**The Topological Layer** encodes *rules*—from this lane, you can go straight or turn right, but not left.

---

### Act II: The Lane Graph (The Road's Skeleton)

The most critical structure in an HD map is the **Lane Graph**.

Think of it as the road's skeleton: a directed graph where:

* **Nodes** represent decision points (intersections, lane splits/merges)
* **Edges** represent lane segments with properties (width, curvature, speed limit)
* **Connectivity** encodes legal transitions (can I change from lane 1 to lane 2 here?)

#### The Math: Representing Lanes

Lanes are typically represented as **splines**—smooth mathematical curves.

A common choice is the **Cubic Bézier Spline**:

$$\mathbf{B}(t) = (1-t)^3 \mathbf{P}_0 + 3(1-t)^2 t \mathbf{P}_1 + 3(1-t) t^2 \mathbf{P}_2 + t^3 \mathbf{P}_3$$

Where $t \in [0,1]$ and $\mathbf{P}_0, \mathbf{P}_1, \mathbf{P}_2, \mathbf{P}_3$ are control points.

**Why splines?**

* Compact storage (4 points instead of thousands of coordinates)
* Smooth derivatives (curvature is continuous—important for planning)
* Easy queries ("Where is the lane center 50m ahead?")

For a lane segment, we store:

* Left boundary spline
* Right boundary spline
* Center line spline
* Predecessor/successor lane IDs
* Speed limit, lane type (driving, bike, parking)

#### Querying the Lane Graph

The planner constantly asks:

* "What lane am I in?" → Point-in-polygon test against lane boundaries
* "What's the curvature ahead?" → Evaluate spline derivative
* "Can I change lanes here?" → Check connectivity in the graph
* "What's the speed limit?" → Look up lane attributes

Without the lane graph, the planner would have to infer all of this from raw perception—slow, noisy, and dangerous.

---

### Act III: How Maps Are Made

#### Offline Mapping (The Traditional Approach)

Companies like Waymo, Cruise, and TomTom build maps using **dedicated mapping vehicles**.

**The Process:**

1. **Data Collection:** Drive every road with a survey-grade sensor suite (RTK GPS, multiple LiDARs, cameras). Collect terabytes per city.

2. **Point Cloud Registration:** Align all scans into a unified coordinate frame using scan matching (ICP, NDT). This creates a dense 3D model.

3. **Semantic Annotation:** Human labelers (or ML models) identify lanes, signs, and rules. This is expensive—often $1,000+ per mile.

4. **Quality Assurance:** Verify against ground truth, fix errors, validate topology.

5. **Distribution:** Push maps to vehicles via OTA updates.

**The Math: Point Cloud Registration**

When you drive the same road twice, the two LiDAR scans won't align perfectly (GPS drift, sensor noise). You use **Iterative Closest Point (ICP)** or **Normal Distributions Transform (NDT)** to find the transformation $\mathbf{T}$ that aligns them:

$$\mathbf{T}^* = \arg\min_{\mathbf{T}} \sum_{i} \| \mathbf{T} \cdot \mathbf{p}_i - \mathbf{q}_{\text{nearest}(i)} \|^2$$

This is the same algorithm used for localization (Module 4), but here it's used to *build* the map, not just *use* it.

#### Online Mapping (The Emerging Approach)

What if you can't afford mapping vehicles for every road? What if the road changes?

**Online mapping** builds maps on-the-fly using the vehicle's own sensors.

**Tesla's Approach:** Use the fleet. Every Tesla with FSD collects data. When millions of cars see the same intersection, you can aggregate their observations into a map—without dedicated survey vehicles.

**Key Insight:** Crowd-sourced mapping trades precision for coverage. You might not get 10cm accuracy, but you can map every road on Earth.

**The Math: Map Aggregation**

Multiple observations of the same feature (e.g., a lane line) are fused using **weighted averaging**:

$$\hat{\mathbf{x}} = \frac{\sum_i w_i \mathbf{x}_i}{\sum_i w_i}$$

Where $w_i$ is the confidence of observation $i$ (based on sensor quality, GPS accuracy, etc.).

---

### Act IV: SLAM — Building Maps Without Maps

What happens when you drive somewhere that hasn't been mapped?

This is the domain of **SLAM: Simultaneous Localization and Mapping**.

#### The Chicken-and-Egg Problem

* To localize, you need a map (to compare against).
* To build a map, you need to know where you are (to place observations correctly).

SLAM solves both problems simultaneously.

#### The Intuition: Loop Closure

Imagine exploring a dark cave with a flashlight. You walk forward, sketching the walls as you go. After 10 minutes, you realize you've returned to your starting point.

**The Problem:** Your sketch doesn't close. Due to accumulated drift, your drawn path doesn't connect back to the origin.

**The Solution:** You recognize a landmark you saw earlier ("That's the same rock formation!"). This **loop closure** tells you: "This point in my current map is the same as that point from earlier." You can now correct your entire path and map.

#### The Math: Graph SLAM

Modern SLAM represents the problem as a **factor graph**:

* **Variable nodes:** Robot poses at each timestep $(x_1, x_2, ..., x_n)$, landmark positions $(\ell_1, \ell_2, ...)$
* **Factor nodes:** Constraints from odometry (pose-to-pose), observations (pose-to-landmark), and loop closures

The goal is to find the configuration that minimizes total error:

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_{\text{factors}} \| f(\mathbf{x}) - z \|^2_{\Sigma}$$

This is a large nonlinear least-squares problem, solved using techniques like **Gauss-Newton** or **Levenberg-Marquardt**.

#### When Do You Need SLAM?

| Scenario | Use HD Map | Use SLAM |
|----------|-----------|----------|
| Mapped urban area | ✓ | |
| Construction zone (new layout) | | ✓ |
| Parking garage (no GPS) | | ✓ |
| Rural road (never mapped) | | ✓ |
| Post-disaster (roads changed) | | ✓ |

In practice, production systems use **hybrid approaches**: HD maps where available, SLAM for unmapped regions, and continuous map updates from fleet data.

---

### Act V: The Map Freshness Problem

The world changes. Roads get repaved. New construction appears. Traffic patterns shift.

**The Challenge:** Your map was accurate last month. Is it still accurate today?

#### Sources of Map Staleness

1. **Construction:** Lanes shift, barriers appear, detours are added.
2. **Seasonal Changes:** Snow covers lane lines, foliage obscures signs.
3. **Temporary Events:** Accidents, road closures, special events.
4. **Infrastructure Updates:** New signs, repainted markings, signal timing changes.

#### Detection: Is My Map Wrong?

The vehicle can detect map discrepancies by comparing expectations to observations:

* **Expected:** Lane boundary at $y = 3.5m$
* **Observed:** Lane boundary at $y = 4.2m$
* **Discrepancy:** 70cm—too large for sensor noise

When discrepancies exceed a threshold, the system:

1. Flags the area as potentially changed
2. Increases uncertainty in localization
3. Falls back to perception-only mode (treat map as unreliable)
4. Reports the discrepancy for map update

#### The Math: Change Detection

Using a hypothesis test:

$$d = \| \mathbf{z}_{\text{observed}} - \mathbf{z}_{\text{expected}} \|_{\Sigma^{-1}}$$

If $d > \chi^2_{\alpha, n}$ (chi-squared threshold), reject the null hypothesis that the map is correct.

---

### Act VI: Map-Heavy vs. Map-Light (The Industry Debate)

There's a fundamental philosophical divide in the industry:

#### Team Map-Heavy (Waymo, Cruise, Mobileye)

**Philosophy:** "Pre-compute everything you can."

**Argument:**
* HD maps offload computation from real-time to offline
* More reliable than perception in edge cases (faded lane lines, occlusions)
* Enables centimeter-accurate localization
* Safety: You know the rules before you arrive

**Drawbacks:**
* Expensive to create and maintain ($millions per city)
* Doesn't scale to rural or international roads
* Brittle when maps are stale

#### Team Map-Light (Tesla, Wayve, Comma.ai)

**Philosophy:** "Learn to see, don't memorize."

**Argument:**
* Human drivers don't need HD maps—neither should cars
* Perception + reasoning should be sufficient
* Scales to anywhere cameras can see
* More robust to changes (no stale map problem)

**Drawbacks:**
* Harder perception problem (must infer everything real-time)
* Less reliable in edge cases (ambiguous markings)
* Requires more compute onboard

#### The Emerging Consensus: Hybrid

The leading systems are converging on a **hybrid approach**:

* Use HD maps where available and fresh
* Fall back to learned perception where maps are unavailable or stale
* Use fleet data to keep maps updated
* Foundation models (Module 9) that can reason about both

Waymo's 6th-gen Driver uses HD maps for structure but foundation models for semantic understanding—getting the best of both worlds.

---

### Summary: The Map as Prior Knowledge

| Concept | What It Provides |
|---------|------------------|
| **HD Map** | Pre-computed, high-accuracy world model |
| **Lane Graph** | Road topology, rules, connectivity |
| **Semantic Layer** | Meaning (signs, markings, zones) |
| **SLAM** | Map building for unknown environments |
| **Map Freshness** | Handling a changing world |

**The Key Insight:** Maps are not just navigation aids. They are **compressed world knowledge** that dramatically simplifies perception, prediction, and planning.

Without a map, the planner must ask: "What are the lanes? Where are they? What are the rules?"

With a map, the planner asks: "Am I in the lane I think I am? Is the map still correct?"

The second question is much easier to answer.

---

### Graduate Assignment: Map Discrepancy Detection

**Task:**

Design a simple map discrepancy detector.

1. **Setup:** You have an HD map with a lane boundary at $y = 3.5m$ (in vehicle frame). Your camera detects a lane boundary at $y = 4.1m$ with standard deviation $\sigma = 0.2m$.

2. **Question 1:** Calculate the Mahalanobis distance between expected and observed positions.

3. **Question 2:** Using a chi-squared test with $\alpha = 0.05$ (one degree of freedom, threshold = 3.84), should you flag this as a map discrepancy?

4. **Question 3:** If you detect a discrepancy, what should the vehicle do? List three possible responses in order of conservatism.

5. **Analysis:** Why is it dangerous to immediately trust perception over the map? When might you be wrong?

**Further Reading:**

* *LaneGraph2Seq: Lane Topology Extraction from LiDAR Point Clouds (CVPR 2023)*
* *MapLite: Autonomous Intersection Navigation Without a Prior Map (ICRA 2018)*
* *Tesla AI Day 2021: Occupancy Networks and Online Mapping*
* *Waymo Open Dataset: Motion Forecasting with Lane Graph*

---

**Previous:** [Module 4 — Localization](/posts/autonomous-stack-module-4-localization)

**Next:** [Module 6 — Perception: Seeing the World](/posts/autonomous-stack-module-6-perception)
