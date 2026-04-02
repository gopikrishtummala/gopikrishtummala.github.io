# next-article

Suggest and draft the next logical article(s) for this blog, based on existing content gaps and series progressions.

## Usage
`/next-article [--track <track>] [--type <type>]`

Examples:
- `/next-article` — scans all tracks and suggests the 3 best next articles
- `/next-article --track "GenAI Systems"` — suggests next for that track specifically
- `/next-article --type series` — focuses on continuing unfinished series

---

## What this skill does

1. **Read** `src/data/blog/` to catalog all existing posts by track
2. **Identify gaps** by comparing against the syllabus in `src/pages/learn/[track]/index.astro`
3. **Detect incomplete series** (e.g., a Part 3 that references a Part 4 not yet written)
4. **Propose** the top 3 candidate articles with rationale
5. **Draft** the chosen article when the user confirms

---

## Known series and gaps to check

### GenAI Systems Track
Current posts (verify in the file system):
- Diffusion: Foundations → Architectures → Sampling → Video → Training Lifecycle → Policy → Frontier → Physics-Aware
- **Gap candidates**: Vision-Language Models (CLIP, LLaVA, Flamingo), Flow Matching vs Diffusion, LoRA / PEFT for generative models

### MLOps & Production Track  
Current posts: Datasets, vLLM Trilogy, Custom Kernels, Inference lifecycle, Agentic MLOps, Parquet/Arrow, Post-Training PEFT
- **Gap candidates**: Quantization deep dive (GPTQ, AWQ, bitsandbytes), Serving benchmarks (TGI vs vLLM vs TensorRT-LLM), Observability for LLMs

### Fundamentals Track
Current posts: XGBoost, Transformers, VAEs, Backprop Math, Python interview hacks
- **Gap candidates**: Reinforcement Learning (mentioned in syllabus but missing), Attention variants (GQA, MQA, MLA), Embedding models and retrieval

### Robotics Track
Current posts: Modules 1–9 of Autonomous Stack series
- **Gap candidates**: Module 10 (End-to-end learning), Module 11 (Sim2Real), or a "Robotics Interview Prep" summary post

### Agentic AI Track
Current posts: Parts 1–5 of Design Patterns series
- **Gap candidates**: Part 6 (Real-world deployments), MCP deep dive, Tool-use reliability patterns

---

## Article Selection Criteria

Rank candidates by:
1. **Series completion** — continuing a series readers are mid-way through (highest priority)
2. **Interview relevance** — topics frequently asked at top ML companies
3. **Uniqueness** — something Gopi has direct experience with (Adobe Firefly, Zoox, Qualcomm)
4. **Syllabus alignment** — fills a gap explicitly listed in the learning path syllabus

---

## When drafting the selected article

Invoke the `/write-article` logic (same style guide applies):
- Voice: practitioner, not textbook
- Structure: Act 0 (hook) → Act 1 (mechanism) → Act 2 (depth) → Act 3 (production) → Interview Angles → Takeaways
- At least one mermaid architecture diagram
- TL;DR bullet points at the top
- Correct series-nav block if part of a series

---

## After suggesting

Present options like:

```
Here are 3 recommended next articles:

1. **[Track] Title** — Why: [reason]. Estimated length: X min read.
2. **[Track] Title** — Why: [reason]. Estimated length: X min read.
3. **[Track] Title** — Why: [reason]. Estimated length: X min read.

Which would you like me to draft? (Reply with 1, 2, 3, or describe something different)
```
