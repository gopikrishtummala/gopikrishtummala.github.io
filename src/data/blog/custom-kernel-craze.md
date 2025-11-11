---
author: Gopi Krishna Tummala
pubDatetime: 2025-11-11T00:00:00Z
modDatetime: 2025-11-11T00:00:00Z
title: The Custom Kernel Craze — Why Developers Are Taking the Wheel on GPU Optimization
slug: custom-kernel-craze
featured: true
draft: false
tags:
  - gpu-optimization
  - deep-learning
  - systems
  - performance
description: A Feynman-style exploration of why modern AI teams are handcrafting GPU kernels—from FlashAttention to TPU Pallas code—and how smarter tooling is making silicon-level tuning accessible.
---

# 🚀 The Custom Kernel Craze: Why Developers Are Taking the Wheel on GPU Optimization

Modern AI looks like software, but its limits are set by physics. Today’s breakthrough models are powered by GPUs and TPUs whose behavior is as finicky as a particle accelerator. This chapter explains, in plain language, why an ever-growing group of developers is abandoning “perfectly good” library kernels and diving headfirst into handcrafted GPU code.

---

## 1. The Problem of the Too-Good Tool

Frameworks like PyTorch and TensorFlow are like Michelin-star cookbooks. When you call `torch.matmul`, the runtime fetches a battle-tested recipe from NVIDIA’s cuBLAS library. For years that was enough—matrix multiply kernels were faster than anything we could write by hand.

The trouble began when models became exotic. Sparse mixtures of experts, colossal sequence lengths, and quirky activation patterns meant our requests no longer matched the cookbook’s assumptions. We still called the master chef, but the results were under-seasoned or painfully slow.

The **custom kernel craze** is developers saying, “I want to stand at the stove myself.” Instead of trusting generic kernels, we handcraft instructions that fit the data layout, memory budget, and hardware topology of the exact model we are training.

---

## 2. The Memory Wall: When Math Isn’t the Bottleneck

Think of a GPU as a giant warehouse full of eager workers (CUDA cores or tensor cores). They compute at trillions of operations per second. What slows them down is the delivery truck: moving data from off-chip global memory into the on-chip scratchpads.

This gap between computation speed and memory bandwidth is the **Memory Wall**. Many classical kernels are optimized for arithmetic throughput, assuming data is waiting on the loading dock. In reality, workers idle while the truck hunts for the next pallet of activations.

Custom kernels flip the script. They focus on **data orchestration**—tiling, caching, fusing—to reduce the number of times data crosses slow interconnects. The math becomes almost secondary; the victory comes from rearranging loads and stores so nothing waits.

---

## 3. A Tale of Custom Silicon: Google TPUs and Pallas

Google’s Tensor Processing Units (TPUs) illustrate why hand tuning resurged. TPUs house monstrous Matrix Multiply Units (MXUs) that deliver staggering throughput, but only if fed in very specific shapes and strides.

- **XLA** handles high-level graph compilation, deciding how operations should be grouped.
- **Pallas** (built on JAX) lets engineers write low-level kernels that explicitly program how tiles are loaded, scheduled, and accumulated inside the MXU.

Automatic tools get you 90% utilization. The last 10%—the difference between a good and great TPU workload—comes from a custom Pallas kernel that hand-places data in SRAM, pipelines the loads, and issues MXU instructions in just the right cadence.

---

## 4. The LLM Revolution Forced Everyone’s Hand

Large language models exploded the size of the attention matrix and tossed the Memory Wall into sharp relief. Two pivotal examples show why bespoke kernels became mandatory.

### 4.1 FlashAttention: When Loop Tiling Saves the Day

Attention requires multiplying $QK^T$, applying a softmax, then multiplying by $V$. Naively, this materializes an $n \times n$ matrix in global memory—catastrophic when $n$ is in the thousands.

**FlashAttention** reorganizes the kernel so that:

1. Queries, keys, and values are loaded block by block.
2. Partial products live entirely in fast on-chip SRAM.
3. Softmax normalization is fused into the same tile loop.
4. Results are written back only once.

By keeping data on-chip and tiling the loops, FlashAttention slashes memory traffic and eliminates the global-memory-sized attention matrix. This single kernel made training GPT-class models viable on commodity clusters.

### 4.2 Expert Parallelism and Custom Reduction Kernels

Mixture-of-Experts LLMs route tokens to a subset of experts every layer. Off-the-shelf kernels couldn’t balance the routing cost with compute. Teams resorted to custom all-to-all communication kernels, fused activation packing, and specialized reductions that exploit NVLink topology. Again, the savings came from managing bandwidth, not inventing new math.

---

## 5. Smarter Tools: Triton, Mojo, and AI-Aided Kernel Design

Handwritten CUDA C++ is notoriously brittle. The new wave of DSLs lowers the barrier.

- **Triton** (OpenAI) lets you write kernels in a Pythonic syntax while still controlling tiling and memory hierarchy. The compiler auto-generates warp-level code and vectorized loads.
- **Mojo** (Modular) aims to blend Python ergonomics with zero-cost abstractions, letting you author kernels that compile down to MLIR and LLVM.
- **AI copilots** are entering the loop. Researchers already prototype custom kernels by prompting a large language model with constraints (“tile 128×128, favor shared memory, target AMD ROCm”). The LLM emits candidate code, which is then verified and auto-tuned.

These tools don’t replace human insight; they amplify it. Developers specify *what* should be tiled or fused, and the compiler handles *how* the warp shuffles, predicate masks, and synchronizations are emitted.

---

## 6. Looking Ahead: Physics-Aware Software Engineering

The custom kernel movement is a reminder that the laws of the chip outweigh the abstractions of software. As models push into multimodal reasoning, long-context memories, and low-latency agents, we should expect:

- **More domain-specific kernels** for attention, routing, quantization, and decompression.
- **Cross-vendor portability layers** that let the same kernel target CUDA, ROCm, and custom ASICs.
- **Autotuners driven by reinforcement learning** that search kernel schedules the way AlphaZero searched for Go strategies.
- **Human-in-the-loop design** where developers sketch the dataflow and let synthesis tools generate optimized code.

In short, performance is becoming a co-design problem. Algorithms, compilers, and hardware architects now collaborate at the granularity of cache lines and tensor tiles.

---

## 7. Try It Yourself

Want to feel this for yourself? Start small:

1. Rewrite a `torch.matmul` using Triton, tiling the operands into $64 \times 64$ blocks. Measure the bandwidth savings when you keep tiles in shared memory.
2. Explore Google’s open-source Pallas tutorials. Replace a simple `jax.numpy.dot` with a custom Pallas kernel that prefetches tiles into SRAM.
3. Prompt your favorite code assistant: “Generate a Triton kernel for scaled dot-product attention that avoids materializing QKᵀ.” Analyze the emitted code, then benchmark it.

The lesson is delightfully Feynman-esque: *to truly know what your GPU is doing, write the kernel yourself.* The physics of data movement is unforgiving, but when you align your code with the chip’s rhythms, the performance gains feel like discovering a new law of nature.

— Gopi Krishna Tummala, curious engineer exploring how machines learn to think (and how the silicon underneath keeps up)


