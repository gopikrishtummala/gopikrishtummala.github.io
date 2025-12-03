---
author: Gopi Krishna Tummala
pubDatetime: 2025-11-25T00:00:00Z
modDatetime: 2025-11-25T00:00:00Z
title: 'Module 2: Turning a Photo into a Blueprint: The Perspective Problem'
slug: satellite-photogrammetry-module-2-geometry
featured: true
draft: false
tags:
  - geospatial
  - photogrammetry
  - geometry
  - coordinate-systems
description: 'A camera takes a 3D world and squishes it onto a flat sensor. Learn the math needed to perfectly reverse this squishing and create accurate maps.'
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
    <a href="/posts/geospatial/satellite-photogrammetry-module-2-geometry" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Module 2: Geometry</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-3-dems-stereo" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 3</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-4-radiometric-correction" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 4</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-5-orthorectification" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 5</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-6-ai-automation" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 6</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-7-multi-source" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 7</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-8-applications" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 8</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Module 2: Turning a Photo into a Blueprint: The Perspective Problem</strong></div>
</div>

---

## The Geometry Challenge

When you take a photo with your phone, the closer things appear bigger, and parallel lines seem to converge. This is **perspective projection**. A satellite image is the same—it's a 2D representation of a 3D reality, and it's full of distortions. If you want to use the image to measure the exact length of a fence on the ground, you can't just use a ruler on the screen.

To create a true map (a **blueprint**), we must mathematically **reverse** the perspective projection. We need to know three things with extreme precision: 1) The exact **location** of the satellite in space, 2) the exact **angle** the camera was pointing, and 3) the internal **geometry** of the camera itself.

---

## 💡 The Math Hook: The Collinearity Equations

This reversal process is governed by the **Collinearity Equations**. Think of them as the mathematical recipe for connecting a 3D point on the Earth $(X, Y, Z)$ to its precise 2D location on the sensor $(x, y)$.

$$
\begin{aligned}
x - x_0 &= -f \cdot \frac{m_{11}(X - X_C) + m_{12}(Y - Y_C) + m_{13}(Z - Z_C)}{m_{31}(X - X_C) + m_{32}(Y - Y_C) + m_{33}(Z - Z_C)} \\[1em]
y - y_0 &= -f \cdot \frac{m_{21}(X - X_C) + m_{22}(Y - Y_C) + m_{23}(Z - Z_C)}{m_{31}(X - X_C) + m_{32}(Y - Y_C) + m_{33}(Z - Z_C)}
\end{aligned}
$$

These two complex-looking formulas are based on simple trigonometry and the **pinhole camera model**. They represent the lines of sight: *The point on the ground, the center of the camera lens, and the point on the sensor must all lie on a single straight line (i.e., they are collinear).* Mastering these equations is mastering the geometry of every map ever made.

**Where:**
- $(x, y)$: Image coordinates
- $(x_0, y_0)$: Principal point (image center)
- $f$: Focal length
- $(X_C, Y_C, Z_C)$: Camera position in 3D space
- $m_{ij}$: Rotation matrix elements (orientation angles)
- $(X, Y, Z)$: Ground coordinates

---

## Key Topics

### The Pinhole Camera Model

The simplest way a camera works. Light from a point in the 3D world passes through a tiny hole (or lens) and projects onto a flat sensor. This creates a **perspective projection** where:

- Objects closer to the camera appear larger
- Parallel lines converge to a vanishing point
- The relationship between 3D and 2D is non-linear

### Why the Earth Isn't Flat: Geoid vs. WGS84 Datum

**Geoid:**
- The true shape of Earth, following mean sea level
- Irregular, "potato-shaped" surface
- Represents the actual gravitational field

**WGS84 Ellipsoid:**
- A mathematical approximation (smooth, oblate spheroid)
- Used as a reference surface for mapping
- Standardized coordinate system for GPS and satellite imagery

**Geographic vs. Projected Coordinates:**

- **Geographic (Latitude/Longitude)**:
  - Latitude: Angle from equator (-90° to +90°)
  - Longitude: Angle from Prime Meridian (-180° to +180°)
  - Units: Degrees, minutes, seconds (DMS) or decimal degrees
  
- **Projected Coordinates (UTM, State Plane)**:
  - Flatten Earth's curved surface onto a plane
  - Preserves distances, angles, or areas (but not all simultaneously)
  - UTM: Universal Transverse Mercator, divides Earth into 60 zones

### Defining the 3D-to-2D Relationship

The collinearity equations establish the fundamental relationship:

1. **Forward Transformation**: Given a 3D point $(X, Y, Z)$ and camera parameters, predict where it appears in the image $(x, y)$
2. **Inverse Transformation**: Given an image point $(x, y)$ and camera parameters, solve for the 3D location $(X, Y, Z)$ (requires additional constraints like height)

---

## Sensor Model for Satellites

**Rigorous Sensor Models (RSM):**

Unlike frame cameras, satellites use **push-broom scanners** that capture one line at a time as the satellite moves.

**Key Differences:**
- Each scan line has its own perspective center
- Orientation changes continuously along the orbit
- Requires time-dependent sensor models

**Sensor Model Components:**
1. **Orbital parameters**: Position and velocity vectors
2. **Attitude angles**: Roll, pitch, yaw (how the sensor is oriented)
3. **Focal length and sensor geometry**: Internal camera parameters

**Rational Polynomial Coefficients (RPC):**
- Alternative to rigorous models
- Polynomial approximation of the sensor geometry
- Faster computation, widely used in commercial software

---

## Ground Control Points (GCPs)

**Why We Need GCPs:**

Ground Control Points are known locations on Earth with precisely measured coordinates. They're essential for:

1. **Georeferencing**: Linking image coordinates to real-world locations
2. **Bundle Adjustment**: Refining camera positions and orientations
3. **Accuracy Assessment**: Validating the final map accuracy

**GCP Requirements:**
- Clearly visible in the image (road intersections, building corners)
- Accurately surveyed coordinates (GPS, survey-grade)
- Distributed across the image (not clustered)
- Typically need 4-6 GCPs minimum for basic georeferencing

**The Process:**
1. Identify GCPs in the image
2. Measure their image coordinates
3. Use known ground coordinates to solve for camera parameters
4. Apply corrections to all other points

---

*Understanding the geometry is crucial for accurate mapping. In the next module, we'll use this geometry to extract 3D height information using stereo pairs.*
