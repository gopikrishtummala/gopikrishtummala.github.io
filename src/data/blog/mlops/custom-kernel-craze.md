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
description: Why modern AI teams are handcrafting GPU kernels—from FlashAttention to TPU Pallas code—and how smarter tooling is making silicon-level tuning accessible.
track: MLOps & Production
difficulty: Advanced
interview_relevance:
  - System Design
  - ML-Infra
estimated_read_time: 25
---

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

The story keeps evolving. **FlashAttention-3** (Tri Dao et al., 2024) rethinks the kernel for NVIDIA Hopper (H100) GPUs by:

- Streaming tiles with the **Tensor Memory Accelerator (TMA)** so global-to-shared transfers overlap with compute.
- Using **warp specialization** and `cp.async` to stage asynchronous shared-memory loads.
- Pushing attention throughput to roughly **70–72% of the H100’s FP16 peak**, roughly 20 percentage points higher than FlashAttention-2 on the same hardware.

It’s still hand-shaped code, but it feels like a compiler wrote it—proof that the standard for “optimized” keeps rising.

### 4.2 Expert Parallelism and Custom Reduction Kernels

Mixture-of-Experts LLMs route tokens to a subset of experts every layer. Off-the-shelf kernels couldn’t balance the routing cost with compute. Teams resorted to custom all-to-all communication kernels, fused activation packing, and specialized reductions that exploit NVLink topology. Again, the savings came from managing bandwidth, not inventing new math.

---

## 5. Smarter Tools: Triton, Mojo, and AI-Aided Kernel Design

Handwritten CUDA C++ is notoriously brittle. The new wave of DSLs lowers the barrier.

- **Triton** (OpenAI) lets you write kernels in a Pythonic syntax while still controlling tiling and memory hierarchy. The compiler auto-generates warp-level code and vectorized loads.
- **Mojo** (Modular) aims to blend Python ergonomics with zero-cost abstractions, letting you author kernels that compile down to MLIR and LLVM.
- **AI copilots** are entering the loop. Researchers already prototype custom kernels by prompting a large language model with constraints (“tile 128×128, favor shared memory, target AMD ROCm”). The LLM emits candidate code, which is then verified and auto-tuned.

These tools don’t replace human insight; they amplify it. Developers specify *what* should be tiled or fused, and the compiler handles *how* the warp shuffles, predicate masks, and synchronizations are emitted.

#### 5.1 Triton 3.0: Autotuning Out of the Box

The October 2025 release of **Triton 3.0** pushes the DSL even closer to a “Python for kernels” experience:

- A built-in autotuner that combines Bayesian optimization with learned cost models to sweep tile shapes automatically.
- MLIR-based fusion that spans kernel boundaries, so patterns like `matmul → layernorm → gelu` can live in a single launch.
- A rapidly maturing **ROCm backend**; large tiles run within 80–85% of CUDA throughput on AMD’s MI300X.

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A, B, C, M, N, K, BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    # Triton 3.0 fuses loads/stores across blocks and plugs into the autotuner.
    ...
```

Callers can wrap the kernel with `triton.autotune` and let the runtime test 64×64, 128×128, or asymmetric tiles, picking the winner per workload.

#### 5.2 AI-Assisted Kernel Drafting

Prompt-based kernel authoring is no longer science fiction. Engineers regularly feed a large language model instructions like:

> “Write a Triton kernel for softmax(Q @ Kᵀ) @ V with sequence length 8192, head dim 128, on H100. Use 128×128 tiles, shared memory for K/V blocks, fuse softmax stats, avoid materializing QKᵀ, and use Tensor Memory Accelerator if available.”

The first draft often lands within 85–90% of FlashAttention-3’s performance. From there, `triton.autotune` or Pallas’ auto-scheduler can close the remaining gap, leaving humans to polish edge cases and numerical stability.

---

## 6. Looking Ahead: Physics-Aware Software Engineering

The custom kernel movement is a reminder that the laws of the chip outweigh the abstractions of software. As models push into multimodal reasoning, long-context memories, and low-latency agents, we should expect:

- **More domain-specific kernels** for attention, routing, quantization, and decompression.
- **Cross-vendor portability layers** that let the same kernel target CUDA, ROCm, and custom ASICs.
- **Autotuners driven by reinforcement learning** that search kernel schedules the way AlphaZero searched for Go strategies.
- **Human-in-the-loop design** where developers sketch the dataflow and let synthesis tools generate optimized code.
- **LLM compilers** that emit Triton, Pallas, or Mojo kernels directly from high-level graphs, possibly co-simulated against hardware tools like NVIDIA’s Atlantis.
- **Curated kernel “zoos”** per model family (Llama, Gemma, DeepSeek) where the community standardizes the best schedules and shares roofline plots.

In short, performance is becoming a co-design problem. Algorithms, compilers, and hardware architects now collaborate at the granularity of cache lines and tensor tiles.

---

## 7. Try It Yourself

Want to feel this for yourself? Start small:

1. Rewrite a `torch.matmul` using Triton, tiling the operands into $64 \times 64$ blocks. Measure the bandwidth savings when you keep tiles in shared memory.
2. Explore Google’s open-source Pallas tutorials. Replace a simple `jax.numpy.dot` with a custom Pallas kernel that prefetches tiles into SRAM.
3. Prompt your favorite code assistant: “Generate a Triton kernel for scaled dot-product attention that avoids materializing QKᵀ.” Analyze the emitted code, then benchmark it.
4. Take the Triton 3.0 autotuner for a spin:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def silly_fast_matmul(a_ptr, b_ptr, c_ptr, M, N, K,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Tile, load into shared memory, accumulate, and write back—left as an exercise.
    ...

autotuned_matmul = triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK})
        for BM in [64, 128]
        for BN in [64, 128]
        for BK in [32, 64]
    ],
    key=['M', 'N', 'K']
)(silly_fast_matmul)
```

Benchmark against `torch.matmul` on A100 or H100; you’ll often see 1.4–1.8× speed-ups just from tiling and fusion.

The lesson is simple: *to truly know what your GPU is doing, write the kernel yourself.* The physics of data movement is unforgiving, but when you align your code with the chip’s rhythms, the performance gains feel like discovering a new law of nature.

— Gopi Krishna Tummala, curious engineer exploring how machines learn to think (and how the silicon underneath keeps up)


