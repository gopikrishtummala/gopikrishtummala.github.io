---
author: Gopi Krishna Tummala
pubDatetime: 2025-11-25T00:00:00Z
modDatetime: 2025-11-25T00:00:00Z
title: 'Module 4: Radiometric and Atmospheric Correction'
slug: satellite-photogrammetry-module-4-radiometric-correction
featured: true
draft: false
tags:
  - geospatial
  - photogrammetry
  - radiometric-correction
  - atmospheric-correction
description: 'Cleaning up image data for reliable measurements. Learn how to remove atmospheric effects and sensor distortions to get the true brightness value of objects on the ground.'
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
    <a href="/posts/geospatial/satellite-photogrammetry-module-4-radiometric-correction" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Module 4: Radiometric Correction</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-5-orthorectification" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 5</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-6-ai-automation" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 6</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-7-multi-source" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 7</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-8-applications" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 8</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Module 4: Radiometric and Atmospheric Correction</strong></div>
</div>

---

## 4.1 Sources of Image Distortion

**Geometric vs. Radiometric Distortions:**

- **Geometric Distortions**: Affect the position of features (covered in Module 5)
- **Radiometric Distortions**: Affect the brightness/color values of pixels

**Sources of Radiometric Distortion:**

1. **Atmospheric Effects**:
   - Scattering (Rayleigh, Mie)
   - Absorption by gases (water vapor, ozone)
   - Haze and aerosols

2. **Sensor Effects**:
   - Detector sensitivity variations
   - Calibration drift over time
   - Electronic noise

3. **Illumination Effects**:
   - Sun angle variations
   - Topographic shading
   - Cloud shadows

4. **Surface Effects**:
   - Bidirectional reflectance (BRDF)
   - Viewing angle dependencies

---

## 4.2 Radiometric Calibration

**Converting Raw Digital Numbers to Physical Units:**

Satellite sensors record **Digital Numbers (DN)**—raw pixel values (0-255 for 8-bit, 0-4095 for 12-bit). For scientific analysis, we need physical units.

**The Calibration Chain:**

1. **DN to Radiance (L)**:
   $$L = \text{gain} \times \text{DN} + \text{offset}$$
   - Radiance: Energy per unit area per unit solid angle (W/m²/sr/μm)
   - Gain and offset provided in image metadata

2. **Radiance to Reflectance (ρ)**:
   $$\rho = \frac{\pi \times L \times d^2}{E_{\text{sun}} \times \cos(\theta_s)}$$
   Where:
   - d: Earth-Sun distance (astronomical units)
   - E_sun: Exoatmospheric solar irradiance
   - θ_s: Solar zenith angle

**Why Reflectance?**
- Reflectance is a property of the surface (0-1 or 0-100%)
- Independent of illumination conditions
- Enables comparison across different images and dates

---

## 4.3 Atmospheric Correction

**Removing Atmospheric Effects:**

The atmosphere scatters and absorbs light before it reaches the sensor. Atmospheric correction removes these effects to get true surface reflectance.

**Types of Scattering:**

1. **Rayleigh Scattering**:
   - Caused by molecules (N₂, O₂)
   - Stronger at shorter wavelengths (blue)
   - Creates blue sky and haze

2. **Mie Scattering**:
   - Caused by aerosols (dust, smoke, water droplets)
   - Affects all wavelengths similarly
   - Creates haze and reduces contrast

**Atmospheric Correction Methods:**

1. **Dark Object Subtraction (DOS)**:
   - Simple, image-based method
   - Assumes dark objects (deep water, shadows) should have zero reflectance
   - Subtracts the minimum DN value from all pixels

2. **Radiative Transfer Models**:
   - MODTRAN, 6S, FLAASH
   - Physics-based, requires atmospheric parameters
   - More accurate but computationally intensive

3. **Machine Learning Approaches**:
   - Train models to predict atmospheric effects
   - Faster than radiative transfer models

---

## 4.4 Cloud and Shadow Removal

**Essential Pre-processing for Time-Series Analysis:**

Clouds and their shadows contaminate images and must be removed or masked.

**Cloud Detection:**

- **Threshold Methods**: Simple brightness/NDVI thresholds
- **Machine Learning**: CNNs trained to detect clouds
- **Multi-temporal**: Compare with cloud-free reference images

**Shadow Detection:**

- Shadows are dark areas adjacent to clouds
- Geometric projection: shadow location depends on sun angle
- Often detected together with clouds

**Handling Strategies:**

1. **Masking**: Mark pixels as invalid, exclude from analysis
2. **Interpolation**: Fill gaps using neighboring pixels or temporal data
3. **Composite Images**: Combine multiple dates to create cloud-free mosaics
4. **Cloud-Free Composite**: Select best pixels from time series

**Quality Assessment:**
- Cloud cover percentage
- Shadow coverage
- Data availability for time-series analysis

---

*Radiometric correction ensures we're measuring true surface properties. In the next module, we'll fix geometric distortions to create accurate maps.*

