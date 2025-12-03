---
author: Gopi Krishna Tummala
pubDatetime: 2025-11-25T00:00:00Z
modDatetime: 2025-11-25T00:00:00Z
title: 'Module 1: The Core Principle of Photogrammetry'
slug: satellite-photogrammetry-module-1-core-principles
featured: true
draft: false
tags:
  - geospatial
  - photogrammetry
  - remote-sensing
  - satellite-imagery
description: 'Introduction to Photogrammetry, Remote Sensing, and Satellite Systems. Learn how we measure things on Earth by taking pictures from space, just like how our two eyes help us perceive depth.'
track: Geospatial
difficulty: Beginner
interview_relevance:
  - Theory
estimated_read_time: 40
---

*By Gopi Krishna Tummala*

---

<div class="series-nav" style="background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Satellite Photogrammetry Course</div>
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;">
    <a href="/posts/geospatial/satellite-photogrammetry-module-1-core-principles" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Module 1: Core Principles</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-2-geometry" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 2: Geometry</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-3-dems-stereo" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 3: DEMs & Stereo</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-4-radiometric-correction" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 4: Radiometric Correction</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-5-orthorectification" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 5: Orthorectification</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-6-ai-automation" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 6: AI & Automation</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-7-multi-source" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 7: Multi-Source</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-8-applications" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 8: Applications</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Module 1: The Core Principle of Photogrammetry</strong></div>
</div>

---

## 1.1 What is Photogrammetry?

Photogrammetry is the science and technology of obtaining reliable information about physical objects and the environment through the process of recording, measuring, and interpreting photographic images and patterns of electromagnetic radiant imagery and other phenomena.

**History and Evolution:**
- Early photogrammetry used aerial photography from balloons and airplanes
- Modern photogrammetry has shifted to satellite platforms for global coverage
- The fundamental principle remains: extracting 3D information from 2D images

---

## 1.2 The Electromagnetic Spectrum (EMS) and Remote Sensing

How satellites "see" the Earth. Satellites capture electromagnetic radiation across different wavelengths, each revealing different information about the Earth's surface.

**Key Concepts:**
- **Wavelength (λ)**: Distance between wave peaks, measured in meters or micrometers
- **Frequency (ν)**: Number of wave cycles per second, measured in Hertz
- **Energy (E)**: Related to frequency by Planck's Law: E = hν

**Planck's Law and Wien's Displacement Law:**
- Planck's Law describes the spectral radiance of electromagnetic radiation
- Wien's Displacement Law: λ_max = b/T, where b is Wien's constant and T is temperature
- These laws explain how satellites detect both visible light and thermal energy

**Remote Sensing Bands:**
- **Visible**: 0.4-0.7 μm (what human eyes see)
- **Near-Infrared (NIR)**: 0.7-1.3 μm (vegetation health)
- **Shortwave Infrared (SWIR)**: 1.3-3.0 μm (mineral identification)
- **Thermal Infrared**: 8-14 μm (temperature measurement)

---

## 1.3 Satellite Orbits and Sensor Types

**Low Earth Orbit (LEO) vs. Geostationary Orbit (GEO):**

- **LEO (160-2000 km altitude)**: 
  - Closer to Earth = higher resolution
  - Orbits Earth multiple times per day
  - Examples: Landsat, Sentinel, WorldView
  
- **GEO (35,786 km altitude)**:
  - Stays fixed over one location
  - Lower resolution but continuous monitoring
  - Examples: Weather satellites

**Sensor Types:**

- **Panchromatic (Pan)**: 
  - Single broad band, black and white
  - High spatial resolution (0.3-1m)
  - Captures fine details
  
- **Multi-spectral**: 
  - Multiple narrow bands (Red, Green, Blue, NIR)
  - Lower spatial resolution (2-30m)
  - Enables color and spectral analysis

---

## 1.4 Resolution: The Trade-off Problem

Resolution in remote sensing has four dimensions, and improving one often means sacrificing another:

**1. Spatial Resolution:**
- Size of the smallest object that can be detected
- Measured in meters per pixel
- Trade-off: Higher resolution = smaller coverage area

**2. Spectral Resolution:**
- Number and width of spectral bands
- Trade-off: More bands = lower signal-to-noise ratio

**3. Temporal Resolution:**
- How often the same area is imaged (revisit time)
- Trade-off: More frequent = lower spatial resolution typically

**4. Radiometric Resolution:**
- Number of brightness levels (bit depth)
- Trade-off: Higher bit depth = larger file sizes

**The Fundamental Trade-off:**
You cannot maximize all four simultaneously. Mission design requires balancing these based on application needs.

---

*This module provides the foundation for understanding how satellites capture Earth's imagery. In the next module, we'll explore the geometric principles that allow us to convert these 2D images into accurate 3D maps.*

