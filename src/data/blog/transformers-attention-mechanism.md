---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-15T00:00:00Z
modDatetime: 2025-01-15T00:00:00Z
title: The Transformer — How Machines Pay Attention
slug: transformers-attention-mechanism
featured: true
draft: false
tags:
  - machine-learning
  - transformers
  - neural-networks
  - nlp
description: An intuitive introduction to the Transformer architecture — from the attention mechanism to self-attention and cross-attention, using language translation as a concrete example.
---

Transformers underpin modern AI — GPT, BERT, ChatGPT, AlphaFold.  
They are neural networks built on attention, enabling models to link distant tokens in parallel.  
This post explains how.

---

## 1. The Problem: Reading with Blindfolds

You're reading a detective novel, solving a mystery.  
The author drops clues in chapter 3, then reveals the answer in chapter 20.  
Your brain connects them: chapter 3's clue jumps to the present.

Before Transformers, neural networks couldn't do this. They read like a person with blindfolds on, processing words strictly left-to-right, forgetting earlier words by the end.

Recurrent Neural Networks (RNNs) tried to remember by maintaining a hidden state — a tiny notebook summarising what they'd seen.  
But it was too small. In a long sentence, early words faded.

This **vanishing memory** problem was fundamental. To understand word position 50, they needed to process positions 1–49 sequentially.  
Processing couldn't be parallelized; attention was local.

In 2017, a Google team proposed something new. Rather than sequential processing, let every word attend to every other word at once.  
That is the Transformer.

---

## 2. From Sequences to Parallel Processing

The core idea: drop sequential processing and attend to the full sequence at once.  
Instead of a single reader, imagine a team of linguists: each word is assigned a linguist who can query and compare all words in parallel.

Mathematically: for a sentence of $n$ words, let word $i$ attend to all other $n-1$ words.  
GPUs can compute these $n \times n$ comparisons in parallel.

---

## 3. Attention: The Mechanism

### 3.1 The Core Concept

**Attention** computes relevance-weighted sums of inputs.  
For each position, score all others, normalize to probabilities, and blend.

Consider translating "The cat sat on the mat" to French.  
When producing "chat", we need gender and context (as well as article/verb form).  
Attention gathers that information.

### 3.2 Query, Key, Value

Each word uses three vectors:
- **Query (Q)**: what I’m searching for
- **Key (K)**: what I’m offering
- **Value (V)**: my content

For word $i$:
1. Dot-product similarities: $\text{score}_{i,j} = Q_i \cdot K_j$
2. Softmax: $\alpha_{i,j} = \frac{e^{\text{score}_{i,j}}}{\sum_k e^{\text{score}_{i,k}}}$
3. Weighted sum: $\text{Attention}(i) = \sum_j \alpha_{i,j} V_j$

Each output is a context-aware blend.

---

## 4. Self-Attention: Words Looking at Words

In **self-attention**, the same sequence supplies Q, K, V.  
Each word attends to all others within the sequence.

Consider "The animal didn't cross the street because it was too tired."  
Self-attention lets "it" focus on "animal" (high weight) and not "street" (low weight).  
It captures local syntax and long-range dependencies in one pass.

### 4.1 Multi-Head Attention

Use multiple attention heads in parallel to capture distinct patterns.  
One head may focus on syntax, another on semantics, another on long-range links.  
The model learns these roles.

Stack 6–12 layers; each refines patterns from the previous.

---

## 5. Encoder-Decoder Architecture and Cross-Attention

Machine translation uses two stacks: encoder (read) and decoder (write).

### 5.1 The Encoder

The encoder reads the source with self-attention across the input.  
Stacked layers accumulate richer representations.

### 5.2 The Decoder

The decoder combines two attention mechanisms:
1. Self-attention over generated tokens (with masking)
2. Cross-attention to the encoder

**Cross-attention** lets the decoder query the encoder.  
The decoder’s Q comes from its outputs; K and V come from the encoder’s outputs.  
Each generation step attends to the full source.

### 5.3 Why Two Types Matter

- Self-attention: internal consistency within a sequence
- Cross-attention: alignment across sequences  
For translation, self-attention handles surface form; cross-attention handles correspondence.

---

## 6. Language Translation: A Concrete Example

For "The cat sat on the mat" → "Le chat s'assit sur le tapis":

**Encoder Phase:**
1. Ingest English tokens
2. 6 layers of self-attention
3. Produce context vectors

**Decoder Phase:**
1. Start with `<start>`
2. Self-attention over the current output prefix
3. Cross-attention to encoder outputs
4. Predict the next token
5. Append; repeat until `</end>`

**Cross-Attention in Action:** When generating "chat", the decoder attends across the encoder: "cat" receives high weight; "le" appears; grammatical features align. Cross-attention forges these links.  
Similarly, self-attention within the French sequence ensures agreement: "chat" → "s'assit" (3rd person singular), "Le" → "chat" (masculine article) with the gender flowing through the decoder.

---

## 7. Positional Encoding

Attention is permutation-invariant; word order matters.  
**Positional encoding** adds positional signals.

Common choices: fixed sinusoidal patterns or learned embeddings.  
Add them to word embeddings before the first layer.

---

## 8. Why Transformers Work

Speed: parallelism replaces sequential recurrence.  
Context: full-sequence access.  
Depth: many layers without degradation.  
Scalability: aligned with GPU compute trends.

Together, they support complex, long-range dependencies.

---

## 9. Context Windows and Limitations

Limits:
- Fixed-size positional embeddings
- Quadratic attention cost in length
- Memory pressure on long contexts

Prompts beyond the window are truncated.  
Extending windows is active: sparse attention, sliding windows, hierarchical patterns.

This bounds practical model sizes.

---

## 10. Summary

Transformers use attention: self-attention for context, cross-attention to bridge sequences.  
This enables parallel processing of long sequences and underpins modern language models.

