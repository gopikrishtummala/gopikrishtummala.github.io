# Editorial roadmap

The working todo list for new posts and site work. Check items off as they ship. Posts follow [style.md](style.md): intuition first, one running example, diagrams, verified facts.

Direction: make the site an engineering knowledge hub that connects **research → core idea → production system → trade-offs**. Prefer engineering investigations over generic tutorials.

## Now: top three

- [x] **LLM Evaluation: How Do You Know Your Model Is Actually Better?** ([post](/posts/generative-ai/llm-evaluation-is-it-actually-better))
  - Offline vs online evaluation; LLM-as-a-judge and its biases; pairwise comparisons and Elo/Bradley-Terry; benchmark contamination; error bars; production feedback.
- [ ] **Inside an LLM Inference Engine: From Request to GPU Kernel**
  - Prefill vs decode, KV cache management, continuous batching, tensor and pipeline parallelism, CUDA graphs and kernel fusion. Builds on the vLLM and speculative decoding posts.
- [ ] **RL for LLMs: From PPO to GRPO and Beyond**
  - Reward models and reward hacking, PPO, GRPO, DAPO, verifiable rewards, on- vs off-policy, KL and stability. Goes deeper than the post-training overview.

## Next: core posts

- [ ] How Multimodal LLMs Actually Work (visual and audio tokens, cross-attention vs early fusion, video, native multimodal training). Check overlap with the VLM, multimodal-foundation-models and VLM/MoE posts first.
- [ ] Data Is the Model: Building Training Datasets at Scale (ingestion, dedup, quality filtering, synthetic data, mixtures, lineage).
- [ ] Training Agents to Use Tools: From SFT to Agentic RL (tool-call data, trajectories, execution feedback, agent rewards, browser and coding agents).
- [ ] Mixture of Experts: The Systems Behind Sparse Models (routers, top-k, load balancing, expert parallelism, communication). Check overlap with the VLM/MoE post.
- [ ] From Attention to FlashAttention (complexity, memory bottlenecks, tiling and IO, FlashAttention-2/3).
- [ ] Mechanistic Interpretability: Looking Inside a Network (features, superposition, sparse autoencoders, activation patching, circuits).
- [ ] Quantization: Running Big Models on Less Hardware (BF16/FP16, INT8/INT4, GPTQ, AWQ, GGUF, KV cache quantization).
- [ ] GPU Performance Engineering: Compute-Bound vs Memory-Bound (roofline, bandwidth, occupancy, profilers).
- [ ] Synthetic Data: Can AI Teach Itself? (teacher-student, self-instruct, filtering, synthetic preferences, model collapse).
- [ ] World Models: Predicting Before Acting (latent dynamics, video prediction, JEPA, model-based RL, physical AI).
- [ ] Beyond RAG: GraphRAG, Long Context and Memory.
- [ ] Observability for AI Systems: Tracing Every Token (traces, token latency, GPU utilization, quality regressions, cost).

## Engineering investigations

- [ ] Why GPU utilization lies.
- [ ] What happens when an LLM serves 10,000 concurrent users? (queueing, batching, KV cache math)
- [ ] Building a multimodal feature store.
- [ ] Why AI benchmarks don't predict production quality (pairs with the evaluation post).
- [ ] Building a reliable AI agent runtime (recovery, state, retries, budgets, observability).

## Flagship series: The Anatomy of a Frontier AI Lab

A 10-part path following a frontier model from research to deployment. Each module links out to the deeper posts above.

1. [ ] How frontier models are pretrained
2. [ ] How training data is collected and curated
3. [ ] How distributed training works at scale
4. [ ] How post-training changes model behavior
5. [ ] How RL teaches models to reason
6. [ ] How multimodal models learn to see and hear
7. [ ] How models are evaluated
8. [ ] How inference infrastructure works
9. [ ] How models are monitored in production
10. [ ] How research ideas become products

## Format experiment: engineering notebook

An optional second format for investigation posts: problem → first principles → architecture → small runnable example → measurements → failure modes → papers. It still follows style.md's limits on math and code: label any numbers that weren't measured as illustrative, and keep snippets short.

## Site: navigation and design

The plan is in `~/.claude/plans/greedy-enchanting-backus.md`.

- [ ] Series route map and "You are here" bar on posts; Prev/Next follow series order (today they follow publish date).
- [ ] Track pages drawn as roads; fix the literal `**` in the learning path.
- [ ] Learn menu in the header and Ctrl+K search.
- [ ] Home page track grid (no clipped text) and "Continue reading".
- [ ] Big Map page of all tracks; interview-prep review mode.
