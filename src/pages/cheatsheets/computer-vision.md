---
layout: ../../layouts/CheatsheetLayout.astro
title: "Computer Vision"
description: "Image basics, classical methods, CNN and transformer architectures, detection, segmentation, 3D vision and multimodal models."
---

## Image basics

- Image = height × width × channels (RGB). Values 0–255 or normalized.
- Color spaces: RGB, HSV (separates color from brightness), grayscale, YUV.
- Pinhole camera: intrinsics $K$ (focal length, principal point) map 3D camera coordinates to pixels; extrinsics (rotation, translation) map world to camera.

## Classical image processing

- Filtering: blur (Gaussian), sharpen, median (salt-and-pepper noise).
- Edges: Sobel, Canny. Corners: Harris.
- Features: SIFT, SURF, ORB (keypoints + descriptors for matching).
- Geometry: homography (planar warps), RANSAC (robust fitting with outliers), epipolar geometry (two views).
- Morphology: erosion, dilation, opening, closing.

## CNN architectures

| Model | Year | Key idea |
|---|---|---|
| LeNet | 1998 | Conv + pool for digits |
| AlexNet | 2012 | Deep CNN on GPUs, ReLU, dropout; won ImageNet |
| VGG | 2014 | Stacks of small 3×3 convolutions |
| GoogLeNet / Inception | 2014 | Parallel filter sizes |
| ResNet | 2015 | Residual connections; 100+ layers |
| MobileNet / EfficientNet | 2017–19 | Depthwise separable convs; compound scaling |
| ConvNeXt | 2022 | Modernized CNN matching transformers |

- Convolution output size: $(W - K + 2P)/S + 1$.
- Receptive field grows with depth, kernel size and stride.

## Vision transformers

- **ViT:** split image into patches (e.g. 16×16), embed, add positions, run a transformer. Needs lots of data or strong pretraining.
- **Swin:** windowed attention with shifting; hierarchical like a CNN.
- Self-supervised pretraining: MAE (masked patches), DINO / DINOv2 (self-distillation).

## Tasks

| Task | Output | Representative models |
|---|---|---|
| Classification | One label | ResNet, ViT |
| Object detection | Boxes + labels | Faster R-CNN (two-stage), YOLO, SSD (one-stage), DETR (transformer, set prediction) |
| Semantic segmentation | Class per pixel | FCN, U-Net, DeepLab |
| Instance segmentation | Mask per object | Mask R-CNN |
| Panoptic segmentation | Both | Mask2Former |
| Promptable segmentation | Mask for any prompt | SAM / SAM 2 |
| Pose estimation | Keypoints | OpenPose, HRNet |
| Tracking | IDs across frames | SORT, DeepSORT, ByteTrack |

## Detection essentials

- **IoU:** overlap / union of two boxes. **NMS:** remove overlapping duplicate boxes.
- **mAP:** mean average precision over classes (and IoU thresholds, e.g. COCO mAP@[.5:.95]).
- Anchors vs anchor-free; one-stage (fast) vs two-stage (accurate).
- Segmentation metrics: mIoU, Dice.

## 3D vision

- Depth: stereo, structure from motion, monocular depth networks (Depth Anything).
- Point clouds: PointNet, voxel / pillar networks (PointPillars).
- Bird's-eye-view perception for driving: BEVFormer, BEVFusion. See [Module 6: Perception](/posts/robotics/autonomous-stack-module-6-perception).
- Scene representation: NeRF (neural radiance fields), 3D Gaussian splatting (fast rendering).

## Generative vision

- GANs (StyleGAN), VAEs, diffusion models (Stable Diffusion, DiT, FLUX). See the [diffusion series](/posts/generative-ai/diffusion-from-molecules-to-machines).

## Multimodal

- **CLIP:** image and text encoders trained contrastively into one space; zero-shot classification.
- **Vision-language models:** vision encoder + connector + LLM (LLaVA, Qwen-VL). See [Qwen3-VL](/posts/generative-ai/vlm-moe-qwen-train-to-deploy).
- Open-vocabulary detection and segmentation (Grounding DINO, OWL-ViT, SAM with text prompts).

## Practical tips

- Use pretrained backbones; fine-tune with strong augmentation (flips, crops, color jitter, mixup / cutmix, RandAugment).
- Match train and test preprocessing (resize, normalization).
- Check label quality; visualize predictions and failure cases.
