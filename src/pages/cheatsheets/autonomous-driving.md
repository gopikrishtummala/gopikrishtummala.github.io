---
layout: ../../layouts/CheatsheetLayout.astro
title: "Autonomous Driving in 2 Minutes Each"
description: "Intuition cards for the self-driving stack: ODD, sensors, calibration, localization, HD maps, perception, BEV, occupancy, prediction, planning, closed-loop evaluation, end-to-end models and VLAs."
---

Each card summarizes one module of the [autonomous stack series](/posts/robotics/autonomous-stack-module-1-architecture).

## The stack

**In one line:** sense → calibrate → localize → map → perceive → predict → plan → control.

**Remember:** each module answers one question; most hard cases live between modules.

## ODD (operational design domain)

**In one line:** where and when the car is designed to drive: roads, speeds, weather, time of day.

**Remember:** Level 4 means fully driverless, but only within its ODD. Expansion (new city, snow, freeways) means new rare cases and new validation.

## Why it's hard

**In one line:** it must be right almost every time over enormous distances, and rare events dominate.

**Remember:** US roads see about 1.2 deaths per 100 million miles; proving a car beats that by driving alone would take hundreds of millions of miles, so simulation is essential.

## Sensors

| Sensor | Gives | Weak at |
|---|---|---|
| Camera | Color, text, signals, meaning | Distance, glare, darkness |
| Lidar | Precise 3D shape and distance | Heavy rain, fog; can't read signs |
| Radar | Speed directly; works in bad weather | Low resolution, ghost echoes |

**Remember:** they're combined because they fail differently.

**Go deeper:** [Module 2](/posts/robotics/autonomous-stack-module-2-sensors).

## Calibration and time sync

**In one line:** know exactly where each sensor is, how each lens bends light, and when each measurement was taken.

**Remember:** angle errors grow with distance: half a degree is ~17 cm at 20 m. Shared clocks and motion compensation matter as much as position.

**Go deeper:** [Module 3](/posts/robotics/autonomous-stack-module-3-calibration).

## Localization

**In one line:** where exactly the car is, to centimeters.

**Remember:** GPS is ~5 m; combine satellites, IMU, wheel odometry and map matching. Factor graphs can revise the past and reject bad data.

**Go deeper:** [Module 4](/posts/robotics/autonomous-stack-module-4-localization).

## HD maps

**In one line:** a stored memory of the road: lane geometry, stop lines, which lane leads where.

**Remember:** maps are priors, not truth; live observations win for safety. Cars increasingly draw maps online from cameras (MapTR).

**Go deeper:** [Module 5](/posts/robotics/autonomous-stack-module-5-mapping).

## Perception and BEV

**In one line:** turn sensor data into a list of objects with positions, sizes and velocities.

**Remember:** fuse evidence early, not finished answers; convert everything into a top-down bird's-eye view where planning happens.

**Go deeper:** [Module 6](/posts/robotics/autonomous-stack-module-6-perception).

## Occupancy

**In one line:** for each cube of space, is it filled and moving?

**Remember:** catches debris and odd objects that don't fit a box or a category.

## Prediction

**In one line:** what will everyone else do in the next few seconds?

**Remember:** output several futures with probabilities; predict interacting agents jointly; others react to what your car does.

**Go deeper:** [Module 7](/posts/robotics/autonomous-stack-module-7-prediction).

## Planning

**In one line:** choose a safe, legal, comfortable trajectory that makes progress.

**Remember:** replan many times a second; learned planners that copy humans can still fail in closed loop; hybrids add explicit scoring and a safety layer.

**Go deeper:** [Module 8](/posts/robotics/autonomous-stack-module-8-planning).

## Open loop vs closed loop

**In one line:** open loop compares plans with logged human driving; closed loop lets the car's choices change what happens next.

**Remember:** good open-loop scores can hide bad driving. Judge planners in closed-loop simulation.

## End-to-end and VLA models

**In one line:** one learned model from sensors (and language) to trajectories.

**Remember:** brings world knowledge to rare scenes; still needs a separate safety layer and closed-loop evaluation.

**Go deeper:** [Module 9](/posts/robotics/autonomous-stack-module-9-foundation-models), [VLA models](/posts/robotics/vla-models-data-architecture-evaluation).
