---
layout: ../../layouts/CheatsheetLayout.astro
title: "Computer Vision in 2 Minutes Each"
description: "Intuition cards for images and cameras, convolution, CNN architectures, ResNet, vision transformers, data augmentation, detection, IoU and NMS, segmentation, SAM, depth and 3D, NeRF and Gaussian splatting, CLIP and vision-language models."
---

How machines learn to see.

## Images as numbers

**In one line:** a grid of pixels, each with color values.

**Remember:** a 1080p color image is about 6 million numbers; most of vision is learning which patterns among them matter.

## Camera model

**In one line:** a camera projects the 3D world onto a 2D image.

**Remember:** intrinsics describe the lens (focal length, center); extrinsics describe where the camera sits. Needed whenever images meet 3D.

**Go deeper:** [Module 3: Calibration](/posts/robotics/autonomous-stack-module-3-calibration).

## Convolution

**In one line:** slide a small filter across the image to detect a pattern everywhere.

**Picture it:** a stencil for "vertical edge" pressed against every spot on the photo.

**Remember:** early layers learn edges and colors; deeper layers learn textures, parts and objects.

## CNN architectures

| Model | Remember it as |
|---|---|
| AlexNet (2012) | Deep learning's ImageNet breakthrough on GPUs |
| VGG | Just stack small 3×3 filters |
| ResNet | Skip connections make 100+ layers trainable |
| MobileNet / EfficientNet | Cheap convolutions for phones |
| ConvNeXt | A modernized CNN that matches transformers |

## Vision transformer (ViT)

**In one line:** cut the image into patches, treat them like words, run a transformer.

**Remember:** sees the whole image at once; needs lots of data or pretraining, then scales very well.

## Data augmentation

**In one line:** create variations of training images (flips, crops, color changes) so the model learns what doesn't matter.

**Remember:** the cheapest big win in vision; augmentations must preserve the label.

## Object detection

**In one line:** find each object and draw a box around it with a label.

**Remember:** two-stage detectors (Faster R-CNN) are accurate; one-stage (YOLO) are fast; DETR treats detection as predicting a set of objects with a transformer.

## IoU and NMS

**In one line:** IoU measures box overlap; NMS removes duplicate boxes for the same object.

**Remember:** mAP, the main detection metric, counts a box as correct above an IoU threshold.

## Segmentation

| Type | Output |
|---|---|
| Semantic | A class for every pixel ("road," "car") |
| Instance | A separate mask for each object |
| Panoptic | Both at once |

**Remember:** U-Net's encoder-decoder with skip connections is the classic architecture.

## Segment Anything (SAM)

**In one line:** a foundation model that segments whatever you point at, click or box.

**Remember:** promptable segmentation without task-specific training.

## Self-supervised vision

**In one line:** learn visual features without labels.

**Remember:** masked autoencoders hide patches and rebuild them; DINO learns features so good that object boundaries appear without supervision.

## Depth and 3D

**In one line:** recover distance from stereo cameras, motion, or a single image with a learned model.

**Remember:** monocular depth models (Depth Anything) give relative depth; a few known distances fix the scale. Point clouds come from lidar or depth.

## NeRF and Gaussian splatting

**In one line:** reconstruct a 3D scene from photos so you can render new viewpoints.

**Remember:** NeRF stores the scene in a neural network (slow to render); Gaussian splatting uses millions of small blobs (renders in real time).

## Bird's-eye view

**In one line:** turn camera views into a top-down map around a vehicle.

**Go deeper:** [Module 6: Perception](/posts/robotics/autonomous-stack-module-6-perception).

## CLIP

**In one line:** images and captions trained to land close together in one space.

**Remember:** enables zero-shot classification ("a photo of a dog" vs "a photo of a cat") and powers text-to-image models.

## Vision-language models

**In one line:** a vision encoder plus an LLM that can talk about images.

**Remember:** vision encoder → small connector → language model.

**Go deeper:** [Qwen3-VL](/posts/generative-ai/vlm-moe-qwen-train-to-deploy), [When AI Sees and Speaks](/posts/computer-vision/vision-language-models-explained).

## Generating images and video

**Go deeper:** [Generative Models cheatsheet](/cheatsheets/generative-models).
