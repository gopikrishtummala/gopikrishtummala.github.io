---
layout: ../../layouts/CheatsheetLayout.astro
title: "Transformers & LLMs in 2 Minutes Each"
description: "Intuition cards for attention, transformers, tokenization, pretraining, scaling, decoding, KV cache, MoE, long context, reasoning models and hallucination."
---

Attention, the transformer, and what happens when it becomes a large language model.

## Attention

Each word looks at every other word and decides how much each one matters for understanding it.

**Picture it:** in "the animal didn't cross the street because *it* was tired," the word *it* scans the sentence and lights up *animal* strongly and *street* weakly. That weighted look is attention.

**How it works:**
- Every token makes three vectors: a **query** (what am I looking for?), a **key** (what do I contain?) and a **value** (what do I pass on?).
- Compare one token's query with every key; similar pairs get high scores.
- Softmax turns scores into weights that sum to 1; the output is the weighted mix of values.
- Divide scores by $\sqrt{d}$ so they don't get so large that softmax becomes all-or-nothing.

**Remember:** attention is a soft, learned lookup table: query searches keys, returns a blend of values.

**Watch out:** attention weights show where the model looked, not a reliable explanation of why it decided.

**Go deeper:** [The Transformer](/posts/fundamentals/transformers-attention-mechanism).

## Self-attention vs cross-attention

Self-attention looks within one sequence; cross-attention lets one sequence look at another.

**Picture it:** self-attention is a group discussion inside one team. Cross-attention is one team asking another team questions, like a translator's output checking the source sentence, or image patches checking a text prompt.

**Remember:** queries come from "who's asking," keys and values from "who's being asked."

## Multi-head attention

Run several attentions in parallel, each free to learn a different kind of relationship.

**Picture it:** one head tracks grammar (subject ↔ verb), another tracks coreference (*it* ↔ *animal*), another tracks nearby words. Their findings are concatenated.

**Remember:** many heads = many simultaneous "views" of the same sentence.

## The transformer block

Attention (tokens talk to each other) followed by a small feed-forward network (each token thinks on its own), with shortcuts and normalization around both.

**Picture it:** a meeting (attention) then everyone goes back to their desk to process what they heard (feed-forward). Repeat 30–100 times.

**How it works:**
- Residual connections add each sub-layer's output to its input, so information and gradients flow through deep stacks.
- Layer norm (or RMS norm) keeps activations in a stable range; modern models normalize *before* each sub-layer ("pre-norm").
- The feed-forward layers hold most parameters and much of the stored knowledge.

**Remember:** attention mixes information across tokens; feed-forward transforms each token; residuals keep it trainable.

## Positional encoding and RoPE

Attention has no built-in sense of order, so position must be added.

**Picture it:** without positions, "dog bites man" and "man bites dog" look identical, just a bag of words.

**How it works:**
- Original transformer: add fixed sine and cosine patterns to the embeddings.
- **RoPE** (rotary position embedding), used by most LLMs: rotate each query and key by an angle proportional to its position, so attention scores depend on *relative* distance.

**Remember:** RoPE encodes "how far apart," which helps models handle longer texts and is what context-extension tricks adjust.

## Encoder, decoder, encoder-decoder

| Type | Sees | Good for | Examples |
|---|---|---|---|
| Encoder-only | Whole input both ways | Understanding: classification, embeddings, search | BERT |
| Decoder-only | Only earlier tokens (causal mask) | Generation, chat, everything via prompting | GPT, Llama, Claude |
| Encoder-decoder | Encoder sees all input; decoder generates | Translation, summarization | T5, original transformer |

**Remember:** a causal mask is what turns a transformer into a text generator: no peeking at the future.

## Tokenization

Text is split into sub-word pieces before the model sees it.

**Picture it:** "unbelievable" → "un" + "believ" + "able." Common words are one token; rare words are several.

**How it works:** byte-pair encoding (BPE) starts with characters and repeatedly merges the most frequent pairs into new tokens, building a vocabulary of 30k–200k pieces.

**Remember:** roughly 4 characters or 0.75 English words per token. Costs, context limits and many odd model failures are about tokens.

**Watch out:** models struggle with spelling, counting letters and arithmetic partly because they see tokens, not characters. Non-English text often uses more tokens per word.

## Pretraining: next-token prediction

Predict the next token on trillions of tokens of text; everything else emerges from that.

**Picture it:** to predict the next word of a physics textbook, a mystery novel and Python code, you have to learn physics, plot and programming. The simple objective forces broad knowledge.

**Remember:** pretraining gives knowledge; post-training gives behavior (following instructions, being helpful).

**Go deeper:** [Post-Training](/posts/mlops/modern-post-training-peft-2026).

## Scaling laws

Loss falls smoothly and predictably as you increase model size, data and compute.

**Picture it:** a straight line on a log-log plot. Double the compute, get a predictable improvement.

**How it works:** for a fixed compute budget, there's an optimal split between parameters and training tokens. The Chinchilla paper (2022) found many models were undertrained: roughly 20 tokens per parameter was compute-optimal. Modern models train far beyond that because smaller, longer-trained models are cheaper to serve.

**Remember:** scaling made progress predictable enough to justify enormous training runs.

## In-context learning

The model learns a task from examples in the prompt, with no weight updates.

**Picture it:** show three examples of "English → French" and the model continues the pattern on the fourth.

**Remember:** zero-shot = instructions only; few-shot = instructions plus examples. Example choice and order matter.

## Decoding: temperature, top-k, top-p

How the next token is chosen from the model's probabilities.

**Picture it:** the model hands you a ranked list of next words with probabilities. Decoding is how you pick from it.

**How it works:**
- **Greedy:** always the top word. Deterministic, can be repetitive.
- **Temperature:** below 1 sharpens the distribution (safer), above 1 flattens it (more creative, more errors).
- **Top-k:** sample only from the k most likely. **Top-p (nucleus):** sample from the smallest set covering p of the probability.
- **Beam search:** keep several candidate sequences; common in translation, rare for chat.

**Remember:** low temperature for facts and code, higher for brainstorming.

## KV cache

Store the keys and values of earlier tokens so generating each new token doesn't recompute them.

**Picture it:** a running set of notes on everything said so far; each new word only adds one line.

**Remember:** the KV cache, not the weights, is usually what limits how many users a GPU can serve at once. It grows with every token and every concurrent conversation.

**Go deeper:** [LLM Serving with vLLM](/posts/mlops/vllm-trilogy-of-modern-llm-scaling).

## Prefill vs decode

Reading the prompt is fast and parallel; writing the answer is slow and one token at a time.

**Picture it:** skimming a page at once vs writing a reply word by word.

**Remember:** prefill sets time to first token (compute-bound); decode sets speed of the rest (memory-bound).

**Go deeper:** [Life of a Request](/posts/mlops/life-of-a-tensor-production-inference).

## Long context

Extending how much text a model can read at once, from thousands to millions of tokens.

**How it works:** RoPE scaling tricks (position interpolation, YaRN), training on long documents, efficient attention kernels, and memory-saving KV cache techniques.

**Watch out:** "fits in context" isn't "uses it well." Models often recall the beginning and end better than the middle ("lost in the middle"). Test with your own retrieval tasks.

## Mixture of experts (MoE)

Many small expert networks per layer; a router sends each token to only a few.

**Picture it:** a hospital with 128 specialists where each patient sees 8. Huge total expertise, modest cost per patient.

**Remember:** MoE separates capacity (total parameters, must fit in memory) from compute per token (active parameters). Routers must be kept balanced so a few experts don't take all the work.

**Go deeper:** [Qwen3-VL and Mixture of Experts](/posts/generative-ai/vlm-moe-qwen-train-to-deploy).

## FlashAttention

Compute exact attention in tiles that fit in fast on-chip memory, never writing the full attention matrix to slow memory.

**Remember:** same math, far less memory traffic: faster and enables longer contexts.

**Go deeper:** [Custom Kernels](/posts/mlops/custom-kernel-craze).

## State space models (Mamba)

Sequence models that carry a compressed running state instead of attending to every past token.

**Picture it:** a transformer rereads all its notes for every word; an SSM keeps a running summary.

**Remember:** linear cost in sequence length and constant memory per step; often combined with attention in hybrid models. Weaker at exact recall of arbitrary earlier details.

## Reasoning models

Models trained to "think" at length before answering, spending more compute on harder problems.

**Picture it:** showing your work on scratch paper before writing the final answer.

**How it works:** reinforcement learning on problems with checkable answers (math, code) rewards long chains of reasoning that reach correct results; the model learns to explore, check and backtrack.

**Remember:** test-time compute is a second scaling axis: a model can get better by thinking longer, not just by being bigger.

## Hallucination

Fluent, confident output that isn't true.

**Why it happens:** the model is trained to produce plausible continuations, not to know what it doesn't know; rare facts and missing context make guesses look like facts.

**Remember:** reduce it with retrieval (ground answers in sources), asking for citations, lower temperature, letting the model say "I don't know," and verification of claims.

**Go deeper:** [Building Production RAG](/posts/generative-ai/building-production-rag-opensearch).
