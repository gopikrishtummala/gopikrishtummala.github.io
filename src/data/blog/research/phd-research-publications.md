---
author: Gopi Krishna Tummala
pubDatetime: 2019-01-01T00:00:00Z
modDatetime: 2024-01-01T00:00:00Z
title: From Bits to Sight - A Journey from Wireless to Autonomous Vision
slug: phd-research-publications
featured: true
draft: false
tags:
  - research
  - publications
  - computer-vision
description: My research journey from wireless communication foundations to solving the camera calibration bottleneck that enables autonomous vehicle vision.
---

My PhD at The Ohio State University spanned wireless communication, collaborative vehicular systems, and computer vision. What began as improving wireless protocols gradually converged on a fundamental problem blocking autonomous vehicle perception: camera calibration. The progression was logical—first establishing how things communicate, then using that communication to see the world accurately.

## The Foundation: Wireless Communication and Sensing

Early research focused on making wireless systems faster, more scalable, and more efficient—foundational work for building complex vehicular applications.

**In-band Wireless Cut-Through (2014)** explored whether wireless networks could achieve full-duplex communication to effectively double spectral efficiency. By investigating whether nodes could forward data while still receiving, this work aimed to fundamentally improve network throughput. It established expertise in high-efficiency system design at the physical layer.

**TagTone (IEEE INFOCOM 2018 Best Paper Award)** addressed scalability in RFID communication through multi-frequency analysis. Winning the Best Paper Award recognized the impact of creating robust, scalable solutions in wireless sensing. This work demonstrated that large-scale identification and tracking systems were possible with the right algorithmic approach.

These early successes in wireless taught lessons about communication channels, scalable data processing, and efficient system design—principles that directly informed the transition to vehicular applications.

## The Pivot: From Communication to Vehicular Identification

The insights from making wireless systems scalable immediately applied to collaborative vehicular systems. The goal shifted from moving data efficiently to using that data to improve perception of the physical world.

**RoadMap (2016)** connected wireless identification concepts to vehicles. By mapping observed vehicles to IP addresses using their motion signatures, the system created a digital identity for every car—establishing the foundation for collaboration between vehicles.

**Soft-Swipe (2016-2017)** provided the localization component, enabling high-accuracy pairing of vehicles to specific lanes using commercial off-the-shelf technology. This patent (US 10,032,370, purchased by Honda) proved the commercial viability of converting raw sensor data into precise positional knowledge.

**RoadView (2017)** realized the first demonstration of the vision: a collaborative system where vehicles shared sensed data to improve road condition awareness by 1.8x compared to individual vehicle sensing. This showed that collaborative perception could significantly outperform isolated systems.

## The Breakthrough: Eliminating the Calibration Bottleneck

Despite the promise of collaborative vision, there was a major blocker: manual camera calibration simply didn't scale. If every camera—whether on a traffic pole or a dashboard—required manual setup, large-scale deployment would be impossible. My PhD thesis focused on automating this fundamental problem.

**AutoCalib (2017-2018)** was the general solution for fixed infrastructure cameras. By using deep learning to extract features from common vehicles and robust statistical filtering to combine uncertain estimates, the system could automatically solve for camera parameters. This work won Best Paper Award and Best Demo Award at ACM BuildSys 2017, then appeared in ACM Transactions on Sensor Networks. It demonstrated that self-calibration was mathematically sound and scalable for city-wide infrastructure.

**SmartDashCam (2019)** took the principle to its most challenging form: moving platforms. By making calibration live and continuous, with surrounding traffic serving as dynamic calibration targets, SmartDashCam enabled hands-free operation of advanced driver assistance systems. This publication at ACM IPSN 2019 showed that the calibration challenge could be solved even under motion.

## The Complete Picture

The research trajectory forms a coherent progression: from foundational **wireless throughput** improvements, to secure **vehicle identification** (RoadMap), to precise **localization** (Soft-Swipe), to collaborative **perception** (RoadView), and finally to automated **vision calibration** (AutoCalib/SmartDashCam). Each phase built on previous insights while addressing the next bottleneck in the path to practical autonomous systems.

This work wasn't just a collection of papers—it was a sustained effort to remove system-level roadblocks. The progression from bits to sight—from improving how data flows to making sure cameras can measure reality accurately—demonstrates a consistent focus on making autonomous transportation actually work at scale.

## Awards and Impact

The research received recognition including IEEE INFOCOM 2018 Best Paper Award (TagTone), ACM BuildSys 2017 Best Paper and Best Demo Awards (AutoCalib), and Honda royalty payments for the Soft-Swipe patent. The work has been cited 185+ times and enabled commercial deployment in automotive systems.

## Current Extensions

These foundational ideas continue to influence current work. The robust filtering approaches from AutoCalib appear in behavior prediction systems. The lessons about handling uncertainty at scale inform large-scale ML infrastructure. And the focus on making systems practical, not just technically impressive, guides work on real-time decision-making in safety-critical applications.

Looking back, the journey from wireless protocols to camera calibration wasn't a series of disconnected projects. It was an evolution from understanding how things communicate to ensuring systems can accurately see and understand the world they're navigating.

---

*PhD Dissertation: "Automatic Camera Calibration Techniques for Collaborative Vehicular Applications," The Ohio State University, 2018.*
