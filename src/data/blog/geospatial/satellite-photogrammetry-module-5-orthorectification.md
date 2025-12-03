---
author: Gopi Krishna Tummala
pubDatetime: 2025-11-25T00:00:00Z
modDatetime: 2025-11-25T00:00:00Z
title: 'Module 5: Orthorectification and Mosaicking'
slug: satellite-photogrammetry-module-5-orthorectification
featured: true
draft: false
tags:
  - geospatial
  - photogrammetry
  - orthorectification
  - mosaicking
description: 'Creating geometrically correct and seamless maps. Learn how to remove relief displacement and stitch multiple corrected images into one perfect map.'
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
    <a href="/posts/geospatial/satellite-photogrammetry-module-3-dems-stereo" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 3</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-4-radiometric-correction" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 4</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-5-orthorectification" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Module 5: Orthorectification</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-6-ai-automation" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 6</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-7-multi-source" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 7</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-8-applications" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 8</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Module 5: Orthorectification and Mosaicking</strong></div>
</div>

---

## 5.1 Geometric Errors

**The Necessity of Orthorectification:**

Raw satellite images have geometric distortions, especially **relief displacement**—where tall objects (buildings, mountains) appear to lean away from the image center.

**Types of Geometric Errors:**

1. **Relief Displacement**:
   - Objects at different elevations appear shifted
   - Most significant in hilly/mountainous terrain
   - Increases with distance from nadir (image center)

2. **Sensor Distortions**:
   - Lens distortion
   - Detector misalignment
   - Scan line timing errors

3. **Earth Curvature**:
   - Earth's curvature causes scale variations
   - More significant for wide-swath sensors

4. **Atmospheric Refraction**:
   - Light bends as it passes through atmosphere
   - Small but measurable effect

**Why Orthorectification?**
- Creates a **planimetrically correct** map
- Every pixel represents a true ground position
- Enables accurate distance and area measurements
- Essential for overlaying with other geospatial data

---

## 5.2 The Orthorectification Process

**Combining Raw Image, DEM, and Sensor Model:**

Orthorectification requires three inputs:

1. **Raw Image**: The distorted satellite image
2. **DEM (Digital Elevation Model)**: Height information for every pixel
3. **Sensor Model**: Camera geometry and orientation

**The Process:**

1. **Forward Transformation**:
   - For each pixel in the output orthoimage
   - Calculate its ground coordinates (X, Y) using the DEM height (Z)
   - Use sensor model to find corresponding pixel in raw image

2. **Resampling**:
   - Raw image pixels rarely align perfectly with output grid
   - Interpolate pixel values (nearest neighbor, bilinear, cubic)

3. **Output**:
   - Georeferenced orthoimage
   - Every pixel has accurate ground coordinates
   - Uniform scale across the image

**Mathematical Foundation:**
Uses the collinearity equations from Module 2, but now solving for image coordinates given ground coordinates and elevation.

---

## 5.3 Image Mosaicking

**Blending Multiple Orthophotos:**

A single satellite image covers a limited area. Mosaicking combines multiple orthorectified images into one seamless map.

**Key Steps:**

1. **Seam-line Generation**:
   - Find optimal boundaries between overlapping images
   - Avoid cutting through important features (buildings, roads)
   - Minimize color differences along seams

2. **Radiometric Balancing**:
   - Images captured at different times have different brightness/color
   - Adjust histogram matching or gain/offset correction
   - Create visually seamless appearance

3. **Blending**:
   - **Feathering**: Gradual transition along seam lines
   - **Cutline**: Sharp boundary (requires careful balancing)
   - **Multi-image blending**: Weighted average of overlapping pixels

**Challenges:**
- Color differences between images
- Cloud shadows and artifacts
- Temporal changes (crops, construction)
- Maintaining geometric accuracy across seams

---

## 5.4 Accuracy Assessment

**Evaluating the Final Map's Accuracy:**

**Root Mean Square Error (RMSE):**

The standard measure of geometric accuracy:

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2}$$

Where:
- (x_i, y_i): Measured coordinates from the map
- (x̂_i, ŷ_i): True coordinates from ground control
- n: Number of check points

**Accuracy Metrics:**

1. **Absolute Accuracy**: Error relative to true ground coordinates
   - Typically reported as RMSE in meters
   - Example: "RMSE = 2.5 meters"

2. **Relative Accuracy**: Error between features within the map
   - Important for feature extraction
   - Usually better than absolute accuracy

3. **CE90/LE90**: Circular/Linear Error at 90% confidence
   - 90% of points have error less than this value
   - Common military/mapping standard

**Validation Process:**
- Use independent check points (not used in georeferencing)
- Measure errors in X, Y, and Z (if DEM available)
- Report statistics: mean, RMSE, maximum error

---

*Orthorectification creates accurate, usable maps. In the next module, we'll explore how AI automates feature extraction from these maps.*

