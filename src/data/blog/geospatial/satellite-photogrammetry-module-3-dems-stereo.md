---
author: Gopi Krishna Tummala
pubDatetime: 2025-11-25T00:00:00Z
modDatetime: 2025-11-25T00:00:00Z
title: 'Module 3: Digital Elevation Models (DEMs) and Stereo Imaging'
slug: satellite-photogrammetry-module-3-dems-stereo
featured: true
draft: false
tags:
  - geospatial
  - photogrammetry
  - stereo-imaging
  - dem
description: 'Extracting 3D information (height) using stereo pairs. Learn how taking two pictures from different angles allows us to calculate height using geometric shift (parallax).'
track: Geospatial
difficulty: Intermediate
interview_relevance:
  - Theory
estimated_read_time: 40
---

*By Gopi Krishna Tummala*

---

<div class="series-nav" style="background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Satellite Photogrammetry Course</div>
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;">
    <a href="/posts/geospatial/satellite-photogrammetry-module-1-core-principles" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 1</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-2-geometry" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 2</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-3-dems-stereo" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Module 3: DEMs & Stereo</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-4-radiometric-correction" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 4</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-5-orthorectification" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 5</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-6-ai-automation" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 6</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-7-multi-source" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 7</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-8-applications" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 8</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Module 3: Digital Elevation Models (DEMs) and Stereo Imaging</strong></div>
</div>

---

## 3.1 Stereo Viewing and Parallax

**The Visual Perception of Depth:**

Just like human eyes, stereo imaging uses two viewpoints to perceive depth. When you view the same object from two different positions, it appears to shift—this shift is called **parallax**.

**Key Concepts:**
- **Parallax**: The apparent displacement of an object when viewed from different positions
- **Stereo Pair**: Two images of the same area taken from different viewpoints
- **Conjugate Points**: The same ground point visible in both images

**How It Works:**
1. A satellite captures an image
2. The same area is captured again from a slightly different angle (on a different orbit pass)
3. By measuring the shift (parallax) of points between the two images, we can calculate their height

---

## 3.2 Generating Stereo Pairs

**Overlap and Sidelap Requirements:**

For successful stereo extraction, satellite images need:

- **Forward Overlap**: 60-80% overlap between consecutive images along the flight path
- **Sidelap**: 20-30% overlap between adjacent flight lines
- **Base-to-Height Ratio (B/H)**: The ratio of the distance between camera positions to the height above ground

**Base-to-Height Ratio:**
- Optimal B/H: 0.6 to 1.0
- Too small (< 0.3): Insufficient parallax, poor height accuracy
- Too large (> 1.5): Difficult matching, geometric distortions

**Satellite Collection Strategies:**
- **Same-orbit stereo**: Captured on consecutive passes (days apart)
- **Along-track stereo**: Captured on the same pass using forward/backward pointing sensors
- **Cross-track stereo**: Captured from different orbital paths

---

## 3.3 Differential Parallax

**The Shift Used to Calculate Height:**

Differential parallax (Δp) is the difference in parallax between a point and a reference elevation.

**Height Calculation from Parallax:**

The height of a point above a reference plane can be calculated as:

$$h = \frac{\Delta p \cdot H}{B + \Delta p}$$

Where:
- h: Height above reference plane
- Δp: Differential parallax
- H: Height of camera above reference plane
- B: Baseline (distance between camera positions)

**The Process:**
1. Identify conjugate points in both images
2. Measure the parallax (shift) between corresponding points
3. Apply the height equation to calculate elevation
4. Generate a dense elevation model (DEM) by processing all pixels

---

## 3.4 Image Matching Techniques

**Automating the Matching Process:**

Manually matching points is tedious. Automated algorithms find corresponding points in stereo pairs.

**Area-Based Matching (ABM):**

- Compares small image patches (windows) between images
- Uses correlation or normalized cross-correlation
- Works well for textured areas
- Struggles with repetitive patterns or low texture

**Feature-Based Matching (FBM):**

- Detects distinctive features (corners, edges) first
- Matches features using descriptors (SIFT, SURF, ORB)
- More robust to illumination changes
- Can handle larger geometric distortions

**Modern Approaches:**

- **Semi-Global Matching (SGM)**: Combines local and global optimization
- **Deep Learning**: CNNs trained for dense stereo matching
- **Multi-image matching**: Uses more than two images for better accuracy

**Challenges:**
- Occlusions (objects hidden in one view)
- Illumination differences
- Geometric distortions
- Textureless areas (water, snow, sand)

---

*Stereo imaging is the foundation of DEM generation. In the next module, we'll learn how to clean up the image data for reliable measurements.*

