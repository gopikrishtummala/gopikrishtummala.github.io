---
layout: ../../layouts/CheatsheetLayout.astro
title: "Embeddings, Search & RAG in 2 Minutes Each"
description: "Intuition cards for embeddings, similarity, approximate nearest neighbor search, BM25, hybrid search, reranking, chunking, RAG pipelines and their evaluation."
---

How models find the right information and answer from it instead of from memory.

## Embeddings

**In one line:** turn text, images or anything else into a vector so that similar things land close together.

**Picture it:** a map where "dog" and "puppy" are neighbors, "cat" is nearby, and "invoice" is on another continent.

**Remember:** embeddings make "similar meaning" a distance you can compute. Quality depends on what the model was trained to consider similar.

## Cosine similarity

**In one line:** compare the direction of two vectors, ignoring their length.

**Remember:** 1 = same direction, 0 = unrelated, −1 = opposite. With normalized vectors, cosine similarity and dot product are the same.

## Approximate nearest neighbor (ANN) search

**In one line:** find the closest vectors among millions fast, accepting tiny misses.

**How it works:**
- **HNSW:** a layered graph; jump across far links at the top, refine through near links below, like taking highways then local roads. Fast and accurate, memory-hungry.
- **IVF:** cluster the vectors; search only the nearest clusters.
- **Product quantization:** compress vectors to save memory, at some accuracy cost.

**Remember:** ANN trades a little recall for huge speed. Tune it and measure recall against exact search.

## Keyword search (BM25)

**In one line:** score documents by how often query words appear, weighted by how rare those words are.

**Remember:** keyword search nails exact terms (product codes, names, error messages) that embeddings blur. Embeddings catch paraphrases that keywords miss.

## Hybrid search

**In one line:** run keyword and vector search, then merge.

**How it works:** reciprocal rank fusion combines rankings by position, avoiding the problem that the two kinds of score aren't comparable.

**Remember:** hybrid is a strong default for real-world search.

## Reranking

**In one line:** retrieve many candidates cheaply, then rescore the top few with a slower, smarter model.

**Picture it:** a quick skim of a hundred résumés, then careful reading of the best ten.

**How it works:** a **bi-encoder** embeds query and document separately (fast, used for retrieval). A **cross-encoder** reads query and document together (accurate, used for reranking).

**Remember:** retrieve for recall, rerank for precision.

## Chunking

**In one line:** split documents into pieces small enough to retrieve precisely and large enough to make sense.

**Remember:** split on natural boundaries (sections, paragraphs), keep some overlap, attach metadata (title, section, date). Retrieve small chunks, then show the model the surrounding section ("parent document" retrieval).

**Watch out:** bad chunking is the most common cause of bad RAG answers.

## RAG (retrieval-augmented generation)

**In one line:** retrieve relevant passages and give them to the model to answer from.

**Picture it:** an open-book exam instead of answering from memory.

**How it works:** rewrite the question → search (hybrid) → rerank → put the best passages in the prompt → generate an answer with citations.

**Remember:** RAG is how you add fresh, private or verifiable knowledge without retraining. Citations let users check.

**Go deeper:** [Building Production RAG](/posts/generative-ai/building-production-rag-opensearch).

## Query rewriting

**In one line:** turn a vague or conversational question into a good search query.

**Remember:** "what about the second one?" means nothing to a search engine; rewrite it using the conversation first. Multiple rewrites or a hypothetical answer (HyDE) can improve recall.

## Evaluating RAG

**In one line:** check retrieval and generation separately.

| Stage | Ask | Metrics |
|---|---|---|
| Retrieval | Is the right passage in the top k? | Recall@k, MRR, NDCG |
| Generation | Is the answer supported by the passages? | Faithfulness / groundedness |
| Generation | Does it answer the question? | Answer relevance, correctness |

**Remember:** if retrieval misses, no model can answer correctly. Fix retrieval first.

## When RAG isn't enough

- Questions needing reasoning over many documents: agentic search (search, read, search again).
- Structured data: query the database directly (text-to-SQL) rather than embedding rows.
- Relationships between entities: knowledge graphs (GraphRAG).
