---
author: Gopi Krishna Tummala
pubDatetime: 2025-11-25T00:00:00Z
modDatetime: 2025-11-25T00:00:00Z
title: 'Module 7: Advanced Map Generation II - Multi-Source & Time-Series'
slug: satellite-photogrammetry-module-7-multi-source
featured: true
draft: false
tags:
  - geospatial
  - photogrammetry
  - time-series
  - data-fusion
description: 'Integrating different data types and tracking changes over time. Learn how to combine optical and radar data, and monitor the same area over days, months, or years to see how the world is changing.'
track: Geospatial
difficulty: Advanced
interview_relevance:
  - Theory
  - System Design
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
    <a href="/posts/geospatial/satellite-photogrammetry-module-5-orthorectification" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 5</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-6-ai-automation" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 6</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-7-multi-source" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Module 7: Multi-Source</a>
    <a href="/posts/geospatial/satellite-photogrammetry-module-8-applications" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Module 8</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Module 7: Advanced Map Generation II - Multi-Source & Time-Series</strong></div>
</div>

---

## 7.1 Multi-Spectral Indices

**Using Multiple Color Bands to Highlight Specific Features:**

Multi-spectral indices combine different wavelength bands to highlight specific surface properties that aren't visible in individual bands.

**Normalized Difference Vegetation Index (NDVI):**

The most widely used index for vegetation monitoring:

$$NDVI = \frac{NIR - Red}{NIR + Red}$$

Where:
- NIR: Near-Infrared reflectance
- Red: Red band reflectance
- Values range from -1 to +1
- Healthy vegetation: 0.3 to 0.8
- Water/clouds: < 0
- Soil: 0.1 to 0.3

**Applications:**
- Crop health monitoring
- Deforestation detection
- Drought assessment
- Vegetation phenology (seasonal changes)

**Other Important Indices:**

1. **NDWI (Normalized Difference Water Index)**:
   $$NDWI = \frac{Green - NIR}{Green + NIR}$$
   - Highlights water bodies

2. **NDBI (Normalized Difference Built-up Index)**:
   $$NDBI = \frac{SWIR - NIR}{SWIR + NIR}$$
   - Identifies urban/built-up areas

3. **EVI (Enhanced Vegetation Index)**:
   - Improved version of NDVI
   - Better in high-biomass areas
   - Reduces atmospheric effects

---

## 7.2 Data Fusion (Sensor Synergy)

**Combining Optical Imagery with Synthetic Aperture Radar (SAR):**

Different sensors provide complementary information. Fusion combines their strengths.

**Optical vs. SAR:**

**Optical (Visible/Infrared):**
- ✅ Rich spectral information
- ✅ Intuitive interpretation
- ❌ Blocked by clouds
- ❌ Requires daylight

**SAR (Radar):**
- ✅ Works day/night
- ✅ Penetrates clouds
- ✅ Sensitive to surface structure and moisture
- ❌ More complex interpretation
- ❌ Speckle noise

**Fusion Strategies:**

1. **Pixel-Level Fusion**:
   - Combine at raw pixel level
   - Example: Pan-sharpening (merge high-res pan with multi-spectral)

2. **Feature-Level Fusion**:
   - Extract features from each sensor
   - Combine features for classification
   - Example: Optical texture + SAR backscatter

3. **Decision-Level Fusion**:
   - Classify each sensor independently
   - Combine classification results
   - Voting or probability fusion

**Applications:**
- Cloud-free mapping (SAR fills optical gaps)
- Crop monitoring (optical for species, SAR for structure)
- Flood mapping (SAR detects water, optical provides context)
- Urban mapping (combine building detection from both)

---

## 7.3 Time-Series Analysis

**Using Platforms Like Landsat and Sentinel for Long-Term Monitoring:**

Time-series analysis tracks changes over days, months, or years to understand dynamic processes.

**Key Platforms:**

1. **Landsat** (NASA/USGS):
   - 16-day revisit (30m resolution)
   - 50+ year archive (since 1972)
   - 11 spectral bands

2. **Sentinel-2** (ESA):
   - 5-day revisit (10-60m resolution)
   - Free and open data
   - 13 spectral bands

3. **MODIS** (NASA):
   - Daily global coverage
   - 250m-1km resolution
   - 36 spectral bands

**Time-Series Applications:**

1. **Deforestation Monitoring**:
   - Track forest loss over time
   - Detect illegal logging
   - Measure carbon sequestration

2. **Urban Growth**:
   - Monitor city expansion
   - Track infrastructure development
   - Plan for future growth

3. **Agricultural Monitoring**:
   - Crop growth stages
   - Yield prediction
   - Irrigation management

4. **Disaster Response**:
   - Before/after comparisons
   - Damage assessment
   - Recovery tracking

**Analysis Techniques:**
- **Change Detection**: Compare images from different dates
- **Phenology Analysis**: Track seasonal vegetation cycles
- **Trend Analysis**: Identify long-term patterns
- **Anomaly Detection**: Find unusual events

---

## 7.4 Advanced DEM Generation

**The Role of Interferometric Synthetic Apertation Radar (InSAR):**

InSAR uses phase differences between two SAR images to measure very precise vertical changes.

**How InSAR Works:**

1. **Two SAR Images**: Captured from slightly different positions
2. **Interferogram**: Phase difference between images
3. **Phase Unwrapping**: Convert phase to height
4. **DEM Generation**: Create elevation model

**Key Advantages:**
- Very high vertical accuracy (cm to mm level)
- Works through clouds
- Large area coverage
- Can measure ground deformation

**Applications:**

1. **Topographic Mapping**:
   - Generate high-accuracy DEMs
   - Fill gaps in optical DEMs
   - Map areas with persistent cloud cover

2. **Ground Deformation Monitoring**:
   - Subsidence (sinking ground)
   - Landslide detection
   - Volcanic activity
   - Earthquake effects

3. **Glacier Monitoring**:
   - Ice flow velocity
   - Thickness changes
   - Climate change impacts

**Differential InSAR (DInSAR):**
- Measures changes between two time periods
- Detects mm-level ground movement
- Critical for infrastructure monitoring

**Challenges:**
- Phase unwrapping complexity
- Atmospheric effects
- Temporal decorrelation
- Requires stable targets

---

*Combining multiple data sources and time-series analysis unlocks powerful monitoring capabilities. In the final module, we'll explore real-world applications and future trends.*

