---
author: Gopi Krishna Tummala
pubDatetime: 2025-12-18T00:00:00Z
modDatetime: 2025-12-18T00:00:00Z
title: "Building Production-Grade Multimodal RAG Systems with OpenSearch"
slug: building-production-rag-opensearch
featured: true
draft: false
tags:
  - generative-ai
  - rag
  - opensearch
  - vector-search
  - multimodal
  - production
  - retrieval
description: "A modern, industry-standard approach to building robust RAG systems using OpenSearch as the core engine. Transition from simple vector retrieval to production-grade multimodal systems handling text, images, and video with advanced patterns like hybrid search, query rewriting, parent-document retrieval, and cross-encoder reranking."
track: GenAI Systems
difficulty: Advanced
interview_relevance:
  - System Design
  - ML-Infra
estimated_read_time: 25
---

*By Gopi Krishna Tummala*

---

## Introduction: From Naive RAG to Production Systems

Modern Retrieval-Augmented Generation (RAG) systems have evolved far beyond simple vector similarity search. This article outlines a modern, industry-standard approach to building robust RAG systems using OpenSearch as the core engine. We will explore how to transition from simple vector retrieval to a production-grade multimodal system that handles text, images, and video.

Production RAG requires careful consideration of indexing algorithms, retrieval strategies, precision optimization, and multimodal capabilities. Each decision impacts latency, recall, cost, and user experience. This guide covers the critical patterns and trade-offs you'll encounter when building systems at scale.

---

## 1. Vector Search Fundamentals: HNSW vs. IVF

In OpenSearch, the `k-NN` plugin provides the backbone for vector search. Choosing the right indexing algorithm is a critical system design decision that impacts latency, memory usage, and recall.

### HNSW (Hierarchical Navigable Small World)

**How it works:** HNSW creates a multi-layered graph where the bottom layer contains all vectors and higher layers contain subsets for fast "skipping." At query time, the algorithm starts at the top layer and navigates downward, finding approximate nearest neighbors efficiently.

**When to use:** HNSW is the industry standard for **low-latency** requirements. It provides high recall and is robust for high-dimensional data (e.g., 1536d from OpenAI embeddings). Most production RAG systems start with HNSW.

**Trade-offs:**
- **High RAM usage:** The entire graph structure is typically stored in memory. This can be partially mitigated with the space parameter `m` and `ef_construction`, or by using approximate HNSW with filters.
- **Fast queries:** Sub-millisecond retrieval for datasets in the millions of vectors.

### IVF (Inverted File Index)

**How it works:** IVF clusters the vector space into "Voronoi cells" (regions). At query time, only the closest clusters are searched, dramatically reducing the search space.

**When to use:** Large-scale datasets where memory efficiency is prioritized over absolute speed. OpenSearch also supports IVF-PQ (product quantization) variants for extreme scale (billions of vectors) with further compression.

**Trade-offs:**
- **Lower recall:** Compared to HNSW, especially if clusters aren't well-distributed.
- **Requires periodic re-training:** As data distribution shifts, clusters may need recalculation.
- **Memory efficient:** Significantly lower memory footprint than HNSW.

### When to Choose Which

| Criteria | HNSW | IVF |
|----------|------|-----|
| Latency requirements | ✅ Best choice | ⚠️ Slower |
| Memory constraints | ⚠️ Higher usage | ✅ More efficient |
| Dataset size | ✅ Good up to ~100M vectors | ✅ Better for billions |
| Recall requirements | ✅ High recall | ⚠️ Lower recall |
| Update frequency | ✅ Handles frequent updates | ⚠️ Requires re-indexing |

**Recommendation:** Start with HNSW for most production RAG systems. Switch to IVF or IVF-PQ only when memory constraints become critical or you're dealing with datasets exceeding 100 million vectors.

---

## 2. Advanced RAG Patterns

To move beyond "Naive RAG," production systems implement a multi-stage pipeline that addresses common retrieval failures.

### Hybrid Search (BM25 + Vector)

Pure vector search often fails on specific keywords (e.g., "iPhone 15 Pro Max") because the embedding might group it with "smartphones" generally, losing the specificity. Conversely, keyword search can miss semantic similarity (e.g., "mobile device" vs. "smartphone").

**Solution:** Hybrid search combines the precision of keyword matching with the semantic understanding of vector search.

**Implementation:** OpenSearch supports native hybrid search via the `hybrid` query type (available from OpenSearch 2.9+), which automatically applies **Reciprocal Rank Fusion (RRF)** to merge results from BM25 and vector queries into a single ranked list.

**Reciprocal Rank Fusion (RRF):** This industry-standard algorithm normalizes and combines scores from different retrieval methods:
- BM25 scores are transformed into ranks
- Vector similarity scores are transformed into ranks
- RRF computes a combined score: `score = Σ (1 / (k + rank))` for each document across both result sets
- `k` is a tuning parameter (typically 60) that controls the influence of lower-ranked results

This ensures that documents appearing in both result sets get boosted, while still preserving highly relevant results from either method alone.

### Query Rewriting & Expansion

Users often ask vague questions or use terminology that doesn't match the indexed content. Query rewriting transforms user queries into more effective search terms.

**Technique:** Use a small LLM (like GPT-4o-mini or Claude Haiku) to rewrite the user's query into a more descriptive search term or generate multiple variations to broaden the search (Multi-Query Retrieval).

**Example:**
- **Original:** "How do I fix my phone?"
- **Rewritten variations:** 
  - "troubleshooting smartphone issues"
  - "common mobile device problems"
  - "phone repair guide"
  - "smartphone troubleshooting steps"

Each variation can be used as a separate query, with results aggregated and ranked. This significantly improves recall for ambiguous queries.

**Multi-Query Retrieval:** Instead of a single rewritten query, generate 3-5 query variations, retrieve documents for each, and merge using RRF. This is particularly effective when the original query is ambiguous or domain-specific.

### Parent-Document Retrieval

Standard chunking (e.g., 500 tokens) often loses context. When a small chunk matches a query, the surrounding context that provides the "big picture" is missing.

**Solution:** Index small "child" chunks for high-granularity search, but when a match is found, retrieve the larger "parent" document or surrounding context to provide to the LLM.

**How it works:**
1. During indexing, create small chunks (children) for detailed retrieval
2. Maintain a mapping from child → parent document
3. At query time, retrieve top child chunks
4. Expand to include parent documents (or neighboring chunks)
5. Pass the enriched context to the LLM

**Implementation:** This pattern is natively supported in frameworks like LlamaIndex (via `HierarchicalNodeParser`) and Haystack. In OpenSearch, you can implement this by:
- Storing `parent_id` metadata on child documents
- Using `terms` query to fetch parent documents after initial retrieval
- Merging and deduplicating before passing to the LLM

This ensures the model has both the precise match and the broader context needed for accurate generation.

---

## 3. The Precision Layer: Re-ranking with Cross-Encoders

Initial retrieval is fast but can be "noisy." Even with hybrid search, you might retrieve documents that are topically related but not actually relevant to the specific query intent.

### Two-Stage Retrieval Architecture

Production RAG systems use a two-stage approach to balance speed and precision:

**Stage 1 (Bi-Encoder - Fast Retrieval):**
- Use HNSW to pull the top 50–200 candidate documents
- This is fast (sub-millisecond) but less accurate
- Recall-focused: cast a wide net

**Stage 2 (Cross-Encoder - Precise Re-ranking):**
- Pass the query and the candidate documents through a Cross-Encoder model (e.g., `BAAI/bge-reranker`, `mixedbread-ai/mxbai-rerank`, or sentence-transformers models fine-tuned for reranking)
- Cross-encoders process the query and document together, allowing for full attention between them
- This yields much higher relevance scores but is slower and more expensive

**Why Cross-Encoders are More Accurate:**

Unlike Bi-Encoders (where query and document are embedded separately), Cross-Encoders process them together. This allows the model to:
- Attend to specific word overlaps and interactions
- Understand nuanced relationships (e.g., negation, temporal ordering)
- Score relevance based on the actual query-document pair, not just similarity in embedding space

**Cost Consideration:** Cross-encoders are slower and more expensive (they process each query-document pair separately), so they're only applied to the reduced candidate set from Stage 1. This gives you the best of both worlds: fast retrieval with precise reranking.

**Popular Models:**
- **BAAI/bge-reranker-v2-m3:** Multilingual, supports 30+ languages
- **mixedbread-ai/mxbai-rerank:** Fast and accurate for English
- **Cohere Rerank API:** Managed service option (highest accuracy, but per-query cost)

---

## 4. Multimodal Search: Image & Video

Modern RAG isn't limited to text. By using **CLIP (Contrastive Language-Image Pre-training)**, we can map text, images, and video frames into the same vector space, enabling unified search across modalities.

### Image Search

**How it works:** Store CLIP embeddings of images in a `knn_vector` field in OpenSearch. A user's text query is embedded via CLIP's text encoder and matched against the image vectors.

**Example use case:** "Show me images of sunset over mountains" — the text query is embedded, matched against image vectors, and relevant images are retrieved even if they weren't tagged with those keywords.

**Embedding dimensions:** CLIP dimensions vary by model:
- **ViT-B/32:** 512 dimensions
- **ViT-L/14:** 768 dimensions
- **OpenAI CLIP:** 1024 dimensions (larger models)

Choose the model based on your accuracy vs. latency trade-off.

### Video Retrieval

Video adds complexity: you need to handle temporal information and extract meaningful frames.

**Frame Sampling Strategy:**
- Extract frames at set intervals (e.g., 1 frame per second)
- For long videos, consider adaptive sampling: higher density during action scenes (detected via scene change detection)
- Store frame-level embeddings and timestamps

**Captioning (Visual-to-Text):**
Use models like BLIP-2, LLaVA, or even PaliGemma to generate textual descriptions for each frame. This enables **caption search** using standard BM25, providing an alternative retrieval path.

**Hybrid Video Search:** Combine:
1. **Frame-level CLIP embeddings** (semantic similarity)
2. **Generated captions** (keyword matching via BM25)
3. **Temporal metadata** (timestamps for precise moment retrieval)

**Temporal Search:** Index timestamps alongside vectors so the RAG system can point users to the exact moment in a video. Store `start_time` and `end_time` metadata to enable temporal queries like "show me the scene where the character enters the room."

**Advanced Techniques:**
- **ImageBind:** A newer multimodal model that can embed audio, video, depth, and more into the same space
- **Video-specific encoders:** Use CLIP on sampled frames + temporal pooling (e.g., average or max pooling across frames) to create video-level embeddings
- **Scene segmentation:** Detect scene boundaries and create embeddings per scene rather than per frame

### Example OpenSearch Mapping for Multimodal RAG

```json
PUT /multimodal-index
{
  "settings": { 
    "index.knn": true,
    "index.knn.algo_param.ef_search": 100
  },
  "mappings": {
    "properties": {
      "text_content": { 
        "type": "text",
        "analyzer": "standard"
      },
      "text_vector": { 
        "type": "knn_vector", 
        "dimension": 1536, 
        "method": { 
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "nmslib",
          "parameters": {
            "ef_construction": 128,
            "m": 24
          }
        }
      },
      "image_vector": { 
        "type": "knn_vector", 
        "dimension": 768, 
        "method": { 
          "name": "hnsw",
          "space_type": "cosinesimil",
          "engine": "nmslib"
        }
      },
      "video_frame_vector": {
        "type": "knn_vector",
        "dimension": 768,
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil"
        }
      },
      "caption_text": {
        "type": "text",
        "analyzer": "standard"
      },
      "metadata": { 
        "properties": { 
          "timestamp": { "type": "float" },
          "start_time": { "type": "float" },
          "end_time": { "type": "float" },
          "source_url": { "type": "keyword" },
          "media_type": { "type": "keyword" },
          "parent_id": { "type": "keyword" }
        } 
      }
    }
  }
}
```

**Key features:**
- Separate vector fields for text, images, and video frames (allowing independent optimization)
- `caption_text` field for hybrid search on video frames
- Temporal metadata for video moment retrieval
- `parent_id` for parent-document retrieval pattern

---

## 5. Context Window & Token Management

Even with long-context models (128k+ tokens), passing too much information leads to the **"Lost in the Middle"** phenomenon—where LLMs ignore context in the center of the prompt, focusing on the beginning and end.

This was first documented in research by Liu et al. (2023) and remains relevant even for modern models with extended contexts.

### Strategies for Token Optimization

**1. Prompt Compression:**
Use libraries like `LLMLingua` or `Selective Context` to remove redundant tokens from retrieved context while preserving key information:
- Identifies and removes repetitive phrases
- Maintains named entities and key facts
- Can reduce context by 50-70% with minimal information loss

**2. Summarization:**
For very long documents, retrieve the document and have a secondary LLM summarize the relevant sections before passing it to the final prompt:
- Extract top chunks via retrieval
- Summarize each chunk independently
- Pass summarized chunks to the main LLM
- Reduces token count while preserving high-level information

**3. Selective Context Pruning:**
Advanced techniques like context caching (available in LangChain and LlamaIndex) can:
- Cache frequently accessed context
- Prune context based on relevance scores
- Dynamically adjust context length based on query complexity

**4. Hierarchical Context:**
For extremely long documents:
- Store document summaries at multiple levels (section, chapter, document)
- Retrieve at the appropriate granularity based on query
- Only expand to detailed chunks when needed

### Token Budget Strategy

Establish a token budget for your RAG pipeline:
- **Retrieval stage:** Top K documents (e.g., 10–20)
- **Re-ranking stage:** Top R documents after reranking (e.g., 5–10)
- **Final context:** Compress or summarize to fit within model limits
- **Reserve tokens** for system prompts, user query, and model output

This ensures you're maximizing the value of retrieved information while staying within model constraints.

---

## Conclusion: Building Production RAG Systems

Building production-grade RAG systems requires careful orchestration of multiple components:

1. **Choose the right indexing algorithm** (HNSW for speed, IVF for scale)
2. **Implement hybrid search** to combine keyword and semantic retrieval
3. **Use query rewriting** to handle ambiguous user queries
4. **Apply parent-document retrieval** to preserve context
5. **Re-rank with cross-encoders** for precision
6. **Extend to multimodal** with CLIP and captioning
7. **Manage token budgets** with compression and summarization

Each of these patterns addresses a specific failure mode in naive RAG systems. Implementing them together creates a robust, production-ready system that can handle real-world query complexity and scale to enterprise workloads.

---

## Next Steps

Would you like a deep-dive on:
- **Implementing hybrid queries with built-in RRF in OpenSearch** (code examples)
- **Parent-document retrieval patterns** (full implementation guide)
- **Deploying multimodal embeddings at scale** (infrastructure considerations)
- **Cost optimization strategies** for cross-encoder reranking

---

*This article is part of the [GenAI Systems](/learn/gen-ai) track. For more on production ML infrastructure, explore the [MLOps & Production](/learn/mlops) track.*

