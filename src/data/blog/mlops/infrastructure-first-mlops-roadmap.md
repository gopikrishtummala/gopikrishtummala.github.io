---
author: Gopi Krishna Tummala
pubDatetime: 2025-12-18T00:00:00Z
modDatetime: 2025-12-18T00:00:00Z
title: "The Infrastructure-First MLOps Roadmap: From Data DNA to Custom Kernels"
slug: infrastructure-first-mlops-roadmap
featured: true
draft: false
tags:
  - mlops
  - production
  - infrastructure
  - career
  - roadmap
  - systems-design
description: "Standard MLOps advice tells you to learn Git and Docker. But for the next generation of AI Engineers, that's just the baseline. This roadmap focuses on the Infrastructure Round—deep-diving into how data is structured for speed, how it's fed into models, how those models scale across clusters, and how we squeeze every drop of performance out of the silicon."
track: MLOps & Production
difficulty: Advanced
interview_relevance:
  - System Design
  - ML-Infra
  - Behavioral
estimated_read_time: 20
---

*By Gopi Krishna Tummala*

---

## TL;DR: The Paradigm Shift

**Most MLOps tutorials teach you how to train a model. Real production systems require you to build the engine that runs them.**

This roadmap is designed for engineers who want to master the infrastructure layer that powers modern AI systems—the hidden machinery that moves data at scale, orchestrates thousand-GPU training jobs, serves millions of concurrent requests, and optimizes code that talks directly to silicon.

**The Journey:**
1. **Month 1:** Understand data formats (Parquet/Arrow) — the foundation
2. **Month 2:** Master dataloaders — the intake valve that feeds GPUs
3. **Month 3:** Scale training across clusters — fault tolerance at 1000 GPUs
4. **Month 4:** Serve models efficiently — vLLM and continuous batching
5. **Month 5:** Write custom kernels — squeeze performance from hardware
6. **Month 6:** Design end-to-end systems — bring it all together
7. **Month 7:** Synthesize through mock interviews — prove your mastery

---

## The Problem with "Standard" MLOps

When I first started building production ML systems, I followed the standard playbook:

1. Learn Git, Docker, and Kubernetes
2. Use Flask or FastAPI to serve models
3. Monitor accuracy metrics
4. Done.

But then I hit reality: **a GPU that cost $40,000 was sitting idle 60% of the time** because the data pipeline couldn't keep up. Our "production" model serving system collapsed under 100 concurrent requests. A 512-GPU training job crashed at hour 40, and we had to restart from scratch.

The standard MLOps curriculum teaches you how to *deploy* models. It doesn't teach you how to *build the infrastructure* that makes them actually work at scale.

This roadmap is different. It's designed to take you from "I can train a model" to "I can build the systems that power billion-parameter models in production."

---

## The Infrastructure-First Philosophy

Every production AI system is built on four layers:

```
┌─────────────────────────────────────────────────────────┐
│  Custom Kernels (Month 5)                               │
│  "How do we talk to the hardware directly?"            │
│  FlashAttention, Triton, Quantization                   │
├─────────────────────────────────────────────────────────┤
│  Serving Infrastructure (Month 4)                       │
│  "How do we serve models to millions of users?"        │
│  vLLM, Continuous Batching, PagedAttention             │
├─────────────────────────────────────────────────────────┤
│  Training Frameworks (Month 3)                          │
│  "How do we train across 1000 GPUs reliably?"          │
│  FSDP, DDP, Fault Tolerance, Checkpointing             │
├─────────────────────────────────────────────────────────┤
│  Data Pipeline (Months 1-2)                             │
│  "How do we feed data to GPUs without bottlenecks?"    │
│  Parquet, Arrow, Dataloaders, Streaming                │
└─────────────────────────────────────────────────────────┘
```

Most engineers start at the top (serving) and work their way down. This roadmap starts at the foundation and builds upward. Why? Because **data format choices made in Month 1 determine whether your Month 4 serving system can scale**.

---

## The 6-Month Learning Path

### 📦 Month 1: The DNA of Data (Storage & Pipelines)

**The Question:** "How do you optimize a 1TB data pipeline for a GPU that is idling?"

**The Answer:** Start with how data is structured, not how models are trained.

#### Key Topics:
- **Columnar Formats:** Why Parquet beats CSV for ML workloads
- **In-Memory Speed:** Apache Arrow and zero-copy reads
- **Data Lineage:** Tracking data provenance (DVC, Lakefs)
- **The Pattern:** Store in Parquet → Convert to Arrow → Use for compute → Convert to tensors only at the final step

#### Deep Dive Article:
📖 **[The DNA of Data: Parquet, Arrow, and the Quest for Analytic Speed](/posts/parquet-arrow-quest-for-analytic-speed)**

This article explains why "just use CSV" is the bottleneck you didn't know you had. Learn how modern systems like HuggingFace Datasets handle petabytes of data by understanding the fundamental trade-off between storage efficiency (Parquet) and compute speed (Arrow).

#### Interview Focus:
*"Walk me through why Parquet is better than CSV for training a large language model on 100TB of text data."*

#### Mini-Lab:
Set up a data pipeline that converts 10GB of CSV files to Parquet, loads them with PyArrow, and measures the I/O speedup. Compare row-based vs. column-based access patterns.

---

### 🔄 Month 2: Datasets & Dataloaders (The Intake Valve)

**The Question:** "Walk me through the lifecycle of a batch from S3 to the GPU register."

**The Answer:** Understand the pipeline that moves data from disk to GPU memory without starving your $40,000 accelerator.

#### Key Topics:
- **Prefetching & Multi-processing:** `num_workers` and `pin_memory` in PyTorch
- **Streaming Datasets:** Handling datasets larger than local disk (MosaicML Streaming)
- **Sharding:** Data partitioning for distributed training
- **The Pipeline:** S3 → Parquet → Arrow → Sharded Arrow → Dataloader → GPU

#### Deep Dive Article:
📖 **[The Hidden Engine of AI: Datasets and Dataloaders](/posts/datasets-and-dataloaders)**

This comprehensive guide explains how data flows from storage to model, covering PyTorch DataLoader internals, HuggingFace Datasets architecture, and NVIDIA DALI for GPU-accelerated preprocessing. Learn why "just increase batch size" doesn't solve throughput problems.

#### Interview Focus:
*"Your GPU utilization is at 40% but your dataloader is maxed out. How do you debug this?"*

#### Mini-Lab:
Build a custom DataLoader that streams data from S3, implements proper sharding for distributed training, and uses Arrow for zero-copy column access. Measure the impact of different `num_workers` settings.

---

### ⚡ Month 3: Training Frameworks & Resilience (The Scale-Out)

**The Question:** "Your 512-GPU training job crashed at hour 40. How do you recover without losing progress?"

**The Answer:** Distributed training isn't just about parallelism—it's about building systems that survive failures.

#### Key Topics:
- **Distributed Strategies:** DDP (Distributed Data Parallel) vs. FSDP (Fully Sharded Data Parallel)
- **Fault Tolerance:** Checkpointing strategies and Elastic Training
- **Mixed Precision:** FP16/BF16 training and loss scaling
- **The Challenge:** Coordinating 1000 GPUs across multiple data centers

#### Deep Dive Article:
📖 **[The Hidden Engine of AI: Training Frameworks and Resilience](/posts/hidden-engine-of-ai)**

Dive deep into PyTorch's distributed training ecosystem, covering DDP, FSDP, and the checkpointing strategies that keep large-scale training jobs alive. Learn why "just use more GPUs" requires sophisticated orchestration.

#### Interview Focus:
*"Design a training system that can survive the failure of 10% of nodes without restarting from scratch."*

#### Mini-Lab:
Set up a multi-GPU training job using FSDP, implement checkpointing that saves state every 1000 steps, and simulate node failures to test recovery. Compare checkpoint frequency vs. training speed trade-offs.

---

### 🚀 Month 4: The Serving Infrastructure (vLLM & Inference)

**The Question:** "Explain the memory fragmentation problem in LLM serving and how PagedAttention solves it."

**The Answer:** Serving LLMs isn't like serving traditional models. You need architectures built for variable-length sequences, dynamic batching, and efficient memory management.

#### Key Topics:
- **PagedAttention:** How vLLM manages KV cache memory without fragmentation
- **Continuous Batching:** Why static batching is dead for LLMs
- **Speculative Decoding:** Using small models to speed up large ones
- **Quantization:** Weight-only vs. Activation quantization (AWQ, FP8)

#### Deep Dive Article:
📖 **[vLLM and the Trilogy of Modern LLM Scaling](/posts/vllm-trilogy-of-modern-llm-scaling)**

This article explains how PagedAttention, Continuous Batching, and Speculative Decoding work together to make LLM serving 10x faster than naive implementations. Learn why serving a 70B model to millions of users requires rethinking memory management from first principles.

#### Interview Focus:
*"Design a system to serve 1 million concurrent users with an LLM-based chatbot. What's the bottleneck?"*

#### Mini-Lab:
Set up a vLLM server with a quantized model (AWQ or GPTQ), measure throughput with continuous batching enabled, and compare memory usage vs. a naive HuggingFace Transformers server.

---

### ⚙️ Month 5: Custom Kernels & GPU Optimization (The "Deep Tech" Layer)

**The Question:** "What is the 'Memory Wall' in GPU computing and how do custom kernels bypass it?"

**The Answer:** Standard libraries are optimized for average cases. Production systems require kernels optimized for your specific workload.

#### Key Topics:
- **FlashAttention:** Why IO-awareness matters more than FLOPs
- **Triton & CUDA:** Why teams write custom kernels instead of using standard libraries
- **Quantization:** Going beyond post-training quantization to quantization-aware training
- **The Reality:** Sometimes you need to write code that talks directly to hardware

#### Deep Dive Article:
📖 **[The Custom Kernel Craze: Why Developers Are Taking the Wheel on GPU Optimization](/posts/custom-kernel-craze)**

Explore why teams at OpenAI, Anthropic, and other AI labs write custom CUDA kernels instead of relying on PyTorch's built-in operations. Learn about the Memory Wall, IO-bound operations, and how FlashAttention changed the game for transformer inference.

#### Interview Focus:
*"Why would you write a custom CUDA kernel instead of using PyTorch's attention implementation?"*

#### Mini-Lab:
Implement a simple custom kernel in Triton (e.g., a fused activation function), benchmark it against PyTorch's implementation, and analyze the memory bandwidth improvements.

---

### 🏗️ Month 6: System Design & Production Monitoring (The Full Loop)

**The Question:** "Design a system to serve 1 million concurrent users with an LLM-based chatbot."

**The Answer:** Bring everything together—data pipelines, training infrastructure, serving systems, and monitoring—into a cohesive end-to-end architecture.

#### Key Topics:
- **E2E System Design:** Designing real-time RAG or Recommendation systems
- **Monitoring:** Latency percentiles (P99), throughput, and GPU utilization
- **Drift & Quality:** Detecting "hallucination drift" and context quality in RAG
- **The Checklist:** 10 things you must monitor before going live

#### Interview Focus:
*"Design a production RAG system that serves 10,000 queries per second with sub-200ms latency. What are the bottlenecks?"*

#### Mini-Lab:
Design and document a complete system architecture for a production ML service. Include data ingestion, training pipeline, serving infrastructure, monitoring, and failure modes. Create a monitoring dashboard mockup.

---

### 🎯 Month 7: Review & Mock Interviews

**The Goal:** Synthesize six months of learning into coherent interview performance.

#### Activities:
- **LeetCode-Style Problems:** Solve tensor manipulation and optimization problems
- **System Design Mock Interviews:** Practice designing large-scale ML systems
- **Article Reviews:** Re-read all 5-6 articles and create your own "cheat sheets"
- **Mental Models:** Build connections between concepts across months

#### Focus Areas:
- Connecting data format choices (Month 1) to serving latency (Month 4)
- Explaining why custom kernels (Month 5) matter for your data pipeline (Month 2)
- Designing fault-tolerant systems (Month 3) that serve models efficiently (Month 4)

---

## The Interview Philosophy: "Why" Over "What"

Senior ML infrastructure interviews don't test whether you know what PagedAttention is. They test whether you understand *why* it was necessary.

Throughout this roadmap, you'll notice a pattern:

1. **Month 1:** Why Parquet/Arrow? → Because CSV pipelines waste GPU cycles
2. **Month 2:** Why streaming datasets? → Because datasets don't fit on disk
3. **Month 3:** Why FSDP? → Because DDP doesn't scale to 1000 GPUs
4. **Month 4:** Why PagedAttention? → Because memory fragmentation kills throughput
5. **Month 5:** Why custom kernels? → Because the Memory Wall limits standard libraries

Each article in this series explains the *problem* before diving into the *solution*. This is exactly what interviewers want to hear: engineers who understand trade-offs, not just technologies.

---

## How to Use This Roadmap

### For Learning (Months 1-6):

1. **Start of Month:** Read the corresponding deep-dive article listed above
2. **Week 2:** Build the "Mini-Lab" project to solidify concepts
3. **Week 3:** Write your own explanation (blog post, notes, or teaching a colleague)
4. **Week 4:** Practice the interview questions for that month

### For Interview Prep (Month 7):

1. **Week 1-2:** Review all articles and create synthesis notes
2. **Week 3:** Practice system design problems (focus on ML infrastructure)
3. **Week 4:** Mock interviews with friends or mentors

### The Meta-Learning Principle:

**You don't truly understand something until you can explain it to someone else.**

After reading each article, try explaining the core concept to a colleague (or write a blog post). This forces you to identify gaps in your understanding and build mental models that connect concepts across months.

---

## What Makes This Different?

Most MLOps roadmaps teach you:
- How to use tools (Docker, Kubernetes, MLflow)
- How to deploy models (FastAPI, Flask, AWS SageMaker)
- How to monitor metrics (accuracy, latency, throughput)

This roadmap teaches you:
- **How data formats impact GPU utilization** (Month 1)
- **How dataloader architecture determines training speed** (Month 2)
- **How distributed training frameworks handle failures** (Month 3)
- **How memory management unlocks LLM serving** (Month 4)
- **How custom kernels bypass hardware limitations** (Month 5)
- **How to design systems that scale end-to-end** (Month 6)

The difference? **This roadmap focuses on the infrastructure layer that makes everything else possible.**

---

## The Articles in This Series

This roadmap is built on deep-dive articles that explore each topic in detail:

1. **[The DNA of Data: Parquet, Arrow, and the Quest for Analytic Speed](/posts/parquet-arrow-quest-for-analytic-speed)**  
   *Why columnar formats and zero-copy reads are the foundation of modern ML pipelines.*

2. **[The Hidden Engine of AI: Datasets and Dataloaders](/posts/datasets-and-dataloaders)**  
   *How data flows from storage to GPU, covering PyTorch, HuggingFace, and NVIDIA DALI.*

3. **[The Hidden Engine of AI: Training Frameworks and Resilience](/posts/hidden-engine-of-ai)**  
   *Scaling training across 1000 GPUs with DDP, FSDP, and fault tolerance.*

4. **[vLLM and the Trilogy of Modern LLM Scaling](/posts/vllm-trilogy-of-modern-llm-scaling)**  
   *How PagedAttention, Continuous Batching, and Speculative Decoding make LLM serving 10x faster.*

5. **[The Custom Kernel Craze: Why Developers Are Taking the Wheel on GPU Optimization](/posts/custom-kernel-craze)**  
   *When and why to write custom CUDA/Triton kernels instead of using standard libraries.*

6. **[Beyond Inference: Architecting Infrastructure for Agentic MLOps & The Model Context Protocol (MCP)](/posts/beyond-inference-agentic-mlops-mcp)**  
   *The journey from stateless inference to stateful, tool-augmented AI agents. Learn how MCP, secure sandbox environments, distributed tracing, and holistic versioning enable the next generation of agentic AI systems.*

Each article stands alone, but together they form a coherent curriculum for mastering ML infrastructure.

---

## The End Goal: Infrastructure-First Thinking

After completing this roadmap, you won't just know how to use tools. You'll understand:

- **Why** certain data formats are chosen for specific workloads
- **How** distributed training frameworks coordinate thousands of GPUs
- **When** to write custom kernels vs. using standard libraries
- **What** trade-offs exist in production ML system design
- **How** to architect infrastructure for agentic AI systems that can maintain state, use tools, and execute complex workflows

More importantly, you'll think like an infrastructure engineer: **optimizing the entire stack, not just the model.**

This is the difference between an engineer who can train a model and an engineer who can build the systems that power AI at scale—from data pipelines to autonomous agents.

---

## Ready to Start?

Begin with **[Month 1: The DNA of Data](/posts/parquet-arrow-quest-for-analytic-speed)**.

Understand how data is structured before you try to move it at scale. Master the foundation, and the rest of the stack becomes intuitive.

**The journey from "I can train a model" to "I can build the infrastructure that powers AI at scale" starts here.**

---

*Have questions or want to discuss this roadmap? Reach out on [Twitter/X](https://twitter.com/gopikrishnat) or [LinkedIn](https://linkedin.com/in/gopikrishnat).*

