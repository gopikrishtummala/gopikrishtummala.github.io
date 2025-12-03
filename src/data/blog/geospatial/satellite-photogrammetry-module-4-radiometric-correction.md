---
author: Gopi Krishna Tummala
pubDatetime: 2025-11-25T00:00:00Z
modDatetime: 2025-11-25T00:00:00Z
title: 'Module 4: The Haze Filter: Cleaning Up the Light Mess'
slug: satellite-photogrammetry-module-4-radiometric-correction
featured: true
draft: false
tags:
  - geospatial
  - photogrammetry
  - radiometric-correction
  - atmospheric-correction
description: 'Every photon of light has a story, but the atmosphere tries to change it. Learn how to un-do the effects of clouds, dust, and haze to get the true measurement.'
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
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Module 4: The Haze Filter: Cleaning Up the Light Mess</strong></div>
</div>

---

## The Atmosphere Problem

When a satellite takes a picture, the light has traveled an enormous distance: from the Sun, through space, through Earth's atmosphere, bouncing off the ground, and then traveling all the way back up through the atmosphere to the sensor. At every stage, the atmosphere interferes.

It **absorbs** some light (like UV rays) and **scatters** other light (like blue light, which is why the sky is blue). This scattering creates **haze**, making the image brighter than it should be and washing out the true colors. If you want to compare how green a forest was this year versus last year, you must first remove the effects of the atmosphere and the sensor itself. This is **Radiometric Correction**.

---

## 💡 The Math Hook: From DN to Reflectance

The ultimate goal is to convert the raw **Digital Number (DN)** into **Reflectance**—the actual percentage of light the ground object bounced back. This requires a series of conversions and subtractions to account for atmospheric path radiance.

One part of the math involves understanding **Rayleigh Scattering**: the inverse fourth power relationship between scattering and wavelength. This math tells us *exactly* how much blue light contamination we should expect in a clear image, allowing us to subtract it and reveal the true colors of the Earth below.

**Rayleigh Scattering Intensity:**

$$I \propto \frac{1}{\lambda^4}$$

Where:
- $I$: Scattering intensity
- $\lambda$: Wavelength

This explains why blue light (shorter wavelength) scatters more than red light (longer wavelength), making the sky appear blue and creating haze in satellite images.

---

## Key Topics

### Sources of Noise

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

### Radiance vs. Reflectance

**Radiance (L)**:
- Light measured at the satellite sensor
- Energy per unit area per unit solid angle (W/m²/sr/μm)
- Depends on illumination conditions
- Varies with time of day, season, atmosphere

**Reflectance (ρ)**:
- Light reflected by the ground surface
- Property of the surface itself (0-1 or 0-100%)
- Independent of illumination conditions
- Enables comparison across different images and dates

**The Calibration Chain:**

1. **DN to Radiance (L)**:
   $$L = \text{gain} \times \text{DN} + \text{offset}$$
   - Gain and offset provided in image metadata

2. **Radiance to Reflectance (ρ)**:
   $$\rho = \frac{\pi \times L \times d^2}{E_{\text{sun}} \times \cos(\theta_s)}$$
   Where:
   - $d$: Earth-Sun distance (astronomical units)
   - $E_{\text{sun}}$: Exoatmospheric solar irradiance
   - $\theta_s$: Solar zenith angle

### Atmospheric Scattering and Absorption

**Types of Scattering:**

1. **Rayleigh Scattering**:
   - Caused by molecules (N₂, O₂)
   - Stronger at shorter wavelengths (blue)
   - Creates blue sky and haze
   - Intensity proportional to $1/\lambda^4$

2. **Mie Scattering**:
   - Caused by aerosols (dust, smoke, water droplets)
   - Affects all wavelengths similarly
   - Creates haze and reduces contrast

**Absorption:**
- Water vapor absorbs infrared radiation
- Ozone absorbs UV radiation
- Carbon dioxide absorbs thermal infrared

### Atmospheric Correction Methods

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

### Cloud and Shadow Removal

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

---

*Radiometric correction ensures we're measuring true surface properties. In the next module, we'll fix geometric distortions to create accurate maps.*
