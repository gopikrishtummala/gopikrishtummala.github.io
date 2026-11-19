---
layout: ../../layouts/CheatsheetLayout.astro
title: "Generative Models in 2 Minutes Each"
description: "Intuition cards for autoencoders, VAEs, GANs, normalizing flows, autoregressive models, diffusion, guidance, latent diffusion, DiT, flow matching, distillation and video generation."
---

How models learn to create new images, video and audio.

## The big idea

**In one line:** learn the distribution of the data well enough to sample new examples from it.

**Remember:** every family below is a different trick for turning simple random noise into realistic data.

## Autoencoder

**In one line:** compress the input to a small code and reconstruct it.

**Picture it:** describing a face in 32 numbers so someone else can redraw it.

**Remember:** great for compression and anomaly detection; a plain autoencoder's code space has gaps, so random codes don't decode into realistic samples.

## VAE (variational autoencoder)

**In one line:** an autoencoder whose code space is smooth and organized, so you can sample from it.

**How it works:** the encoder outputs a small cloud (mean and spread) rather than a point; a penalty keeps all clouds close to a standard Gaussian, so the space has no holes.

**Remember:** reconstruction quality vs a tidy latent space is the trade-off. VAEs are the compressors inside Stable Diffusion.

**Go deeper:** [Variational Autoencoders](/posts/fundamentals/variational-autoencoders-from-compression-to-creation).

## GAN

**In one line:** a generator makes fakes, a discriminator tries to catch them; both improve.

**Picture it:** a counterfeiter and a detective training each other.

**Remember:** sharp, fast samples; unstable training and mode collapse (producing only a few kinds of output). Still used as a loss to sharpen other models.

## Normalizing flows

**In one line:** a chain of invertible transformations from simple noise to data.

**Remember:** exact likelihoods, but architecture constraints limit quality. Not to be confused with flow *matching*.

## Autoregressive models

**In one line:** generate one piece at a time, each conditioned on everything before.

**Remember:** that's how LLMs generate text. Excellent quality, but sequential and slow for images and long outputs.

## Diffusion models

**In one line:** add noise to data step by step, and train a network to remove a little noise at a time.

**Picture it:** ink dispersing in water, played in reverse.

**How it works:** training: take an image, add a random amount of noise, ask the network to predict the noise (a simple squared error). Generation: start from pure static and denoise repeatedly.

**Remember:** stable training and great quality; the cost is many network passes per sample.

**Go deeper:** [Diffusion Foundations](/posts/generative-ai/diffusion-from-molecules-to-machines).

## Classifier-free guidance (CFG)

**In one line:** run the model with and without the prompt and push further in the prompt's direction.

**Remember:** higher guidance = follows the prompt more, less variety, eventually oversaturated. It doubles the cost per step.

**Go deeper:** [Sampling and Guidance](/posts/generative-ai/sampling-guidance-diffusion-models).

## Samplers

**In one line:** the rule for walking from noise to image; decides how many steps you need.

**Remember:** DDPM ~1000 steps; DDIM and ODE solvers ~20–50; distilled models 1–4.

## Latent diffusion

**In one line:** run diffusion on a compressed version of the image, then decode.

**Remember:** a 512×512 image becomes a 64×64×4 latent, about 48× fewer numbers. That's what made Stable Diffusion cheap enough to run.

## Diffusion transformer (DiT)

**In one line:** replace the U-Net with a transformer over latent patches.

**Remember:** transformers scale predictably with compute, so modern image and video models (SD3, FLUX, Sora-style) use them.

**Go deeper:** [From U-Net to DiT](/posts/generative-ai/image-diffusion-models-unet-to-dit).

## Flow matching

**In one line:** learn a straight-line path from noise to data and predict the direction along it.

**Remember:** straighter paths = fewer sampling steps and easier distillation. The modern default for image and video models.

## Diffusion distillation

**In one line:** train a student to do in 1–4 steps what the teacher does in 50.

**Remember:** three families: follow the teacher's path (progressive, consistency models), match its distribution (DMD), add a critic (adversarial, e.g. SDXL-Turbo).

**Go deeper:** [Diffusion Distillation](/posts/generative-ai/diffusion-distillation-fewer-steps).

## Video generation

**In one line:** diffusion over space and time together, so frames stay consistent and motion is plausible.

**Remember:** video VAEs compress time as well as space; full space-time attention gives the best consistency at high cost. Physics remains the weak spot.

**Go deeper:** [Video Diffusion](/posts/generative-ai/video-diffusion-fundamentals).

## Measuring generative models

| Metric | Measures |
|---|---|
| FID | Realism and variety vs real images |
| CLIP score | Prompt-image match |
| FVD, VBench | Video quality, consistency, motion |
| Human preference | What actually matters; used for final calls |
