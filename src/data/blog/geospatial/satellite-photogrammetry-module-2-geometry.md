---
author: Gopi Krishna Tummala
pubDatetime: 2025-11-25T00:00:00Z
modDatetime: 2025-11-25T00:00:00Z
title: 'Module 2: The Geometry of a Satellite Image'
slug: satellite-photogrammetry-module-2-geometry
featured: true
draft: false
tags:
  - geospatial
  - photogrammetry
  - geometry
  - coordinate-systems
description: 'Image Formation, Camera Models, and Coordinate Systems. Learn how a satellite image is a perspective projection and the math needed to reverse this process to get true 3D coordinates.'
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
    <a href="/posts/geospatial/satellite-photogrammetry-module-1-core-principles" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 1: Core Principles</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-2-geometry" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Module 2: Geometry</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-3-dems-stereo" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 3: DEMs & Stereo</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-4-radiometric-correction" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 4: Radiometric Correction</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-5-orthorectification" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 5: Orthorectification</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-6-ai-automation" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 6: AI & Automation</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-7-multi-source" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 7: Multi-Source</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-8-applications" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 8: Applications</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Module 2: The Geometry of a Satellite Image</strong></div>
</div>

---

## 2.1 Earth's Shape and Coordinate Systems

**Geoid vs. Ellipsoid:**

- **Geoid**: The true shape of Earth, following mean sea level (irregular, "potato-shaped")
- **Ellipsoid**: A mathematical approximation (smooth, oblate spheroid)
- For mapping, we use ellipsoids (WGS84, GRS80) as reference surfaces

**Coordinate Systems:**

- **Geographic Coordinates (Latitude/Longitude)**:
  - Latitude: Angle from equator (-90° to +90°)
  - Longitude: Angle from Prime Meridian (-180° to +180°)
  - Units: Degrees, minutes, seconds (DMS) or decimal degrees
  
- **Projected Coordinates (UTM, State Plane)**:
  - Flatten Earth's curved surface onto a plane
  - Preserves distances, angles, or areas (but not all simultaneously)
  - UTM: Universal Transverse Mercator, divides Earth into 60 zones

---

## 2.2 The Pinhole Camera Model

The fundamental geometric relationship in photogrammetry. A satellite image is a **perspective projection**—a 3D scene projected onto a 2D plane.

**The Collinearity Equations:**

These equations relate a 3D point on Earth (X, Y, Z) to its 2D position in the image (x, y):

$$
\begin{aligned}
x - x_0 &= -f \cdot \frac{m_{11}(X - X_C) + m_{12}(Y - Y_C) + m_{13}(Z - Z_C)}{m_{31}(X - X_C) + m_{32}(Y - Y_C) + m_{33}(Z - Z_C)} \\[1em]
y - y_0 &= -f \cdot \frac{m_{21}(X - X_C) + m_{22}(Y - Y_C) + m_{23}(Z - Z_C)}{m_{31}(X - X_C) + m_{32}(Y - Y_C) + m_{33}(Z - Z_C)}
\end{aligned}
$$

Where:
- (x, y): Image coordinates
- (x₀, y₀): Principal point (image center)
- f: Focal length
- (X_C, Y_C, Z_C): Camera position in 3D space
- m_ij: Rotation matrix elements (orientation angles)
- (X, Y, Z): Ground coordinates

**What This Means:**
- Given camera position, orientation, and focal length, we can predict where a ground point appears in the image
- Given image coordinates, we can solve for ground coordinates (with additional constraints)

---

## 2.3 Sensor Model for Satellites

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

## 2.4 Ground Control Points (GCPs)

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

