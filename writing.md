# Writing guide for cheatsheets

This guide covers the cheatsheets at `/cheatsheets`. Blog posts follow [style.md](style.md); cheatsheets follow this file. Where the two conflict for a cheatsheet, this file wins.

---

## 1. Who it's for

**The reader already works in ML, or knows it well, and wants to refresh.**

They're preparing for an interview, walking into a design review, about to read a paper, or picking a topic back up after a year away. They know what a loss function, a gradient, a softmax and a train/test split are. They don't need ML explained; they need a concept's **intuition, trade-offs and gotchas** back in their head in about two minutes, plus a link to go deeper.

Write for a colleague, not a student:

- **Assume the vocabulary.** Don't define "gradient," "overfitting," "embedding," "logits." Define only the concept the card is about, and only in its opening sentence.
- **Skip what every practitioner knows.** No cards for "what is ML," "classification vs regression," "images are grids of pixels." If a senior engineer would skim past it, cut it.
- **Lead with what's easy to forget.** The reason it works, the knob that matters, the failure mode, the number worth remembering.

## 2. What a cheatsheet is, and isn't

A cheatsheet is a **set of concept cards for one area**, readable top to bottom in 15–25 minutes, or one card at a time in two.

It is **not** a tutorial, a derivation, a survey or a blog post. No narrative, no running example across cards, no "in this sheet we will." Depth lives in the linked posts.

## 3. The card

Every concept is one card under its own `##` heading. The card **opens with one plain sentence saying what the concept is, with no label in front of it**. Labels like "In one line:" add nothing; the first sentence is obviously the definition. After that sentence come these labeled fields, in this order:

| Field | Required? | Purpose |
|---|---|---|
| (opening sentence, unlabeled) | Always | What it is, in one sentence a practitioner would nod at. |
| **Picture it:** | When a genuinely good image exists | One concrete analogy or mental picture. Skip rather than force a weak one. |
| **How it works:** | For mechanisms (algorithms, architectures) | 2–5 short bullets. The steps, not the derivation. |
| **Remember:** | Always | The single thing to carry away: the insight, trade-off or rule of thumb. |
| **Watch out:** | When there's a real, common mistake | The misconception or failure mode people actually hit. |
| **Go deeper:** | When a post or sheet covers it | Link to the blog post (or another cheatsheet). |

A comparison table can replace "How it works" when several variants are the point (e.g. encoder vs decoder vs encoder-decoder).

Example of the right altitude:

> ## KV cache
>
> Store the keys and values of earlier tokens so generating each new token doesn't recompute them.
>
> **Remember:** the KV cache, not the weights, usually limits how many users a GPU can serve. It grows with every token and every concurrent conversation.
>
> **Go deeper:** [LLM Serving with vLLM](/posts/mlops/vllm-trilogy-of-modern-llm-scaling).

It doesn't explain what attention is or what a token is. A practitioner knows. It says the one non-obvious thing.

## 4. Length

- **A card: about 40–120 words.** If it needs more, it's two cards, or it belongs in a post.
- **A sheet: 10–20 cards.** Past 20, split the sheet.
- **"How it works": at most 5 bullets**, each one line.

## 5. What makes a good "Remember"

The "Remember" line is the card's reason to exist. Good ones are one of:

- **The insight:** "MoE separates capacity (total parameters) from compute per token (active parameters)."
- **The trade-off:** "Large C: narrow margin, overfit risk; small C: wider margin, smoother boundary."
- **The rule of thumb, with a number:** "About 16 bytes per parameter for mixed-precision Adam training, before activations."
- **The decision rule:** "Retrieve for recall, rerank for precision."

Bad ones restate the opening sentence, or are vague ("this is an important technique").

## 6. Math

- **Intuition first, math rarely.** Use a formula only when the formula *is* what people remember: attention's softmax of $QK^\top/\sqrt{d}$, the Bellman equation, Bayes' rule.
- **At most one formula per card**, inline where possible, with every symbol obvious from context.
- No derivations, no proofs, no multi-line equations. Link to the post that has them.

## 7. Analogies

- One per card, concrete and accurate: "a hospital with 128 specialists where each patient sees 8" for MoE.
- An analogy must not mislead. If it breaks on the property that matters, drop it.
- Don't reuse the same analogy across cards.

## 8. Numbers and facts

- Prefer **numbers practitioners use**: ~4 characters per token, ~16 bytes/parameter for training, ~48× compression in latent diffusion, ~5 m GPS accuracy.
- Every number must be checkable. Verify against the paper, official docs or the linked post. If unsure, leave it out.
- Date fast-moving facts ("as of 2026") or phrase them so they don't go stale.
- No benchmark leaderboard claims; they age in months.

## 9. Voice and formatting

- Plain, direct, present tense. Short sentences.
- Card headings are the concept's name, as people search for it: "Layer norm," "DPO," "Precision vs recall."
- Bold only the field labels and, rarely, one key term.
- Tables only for real comparisons (3+ rows). Bullets only for lists.
- No emoji, no "TL;DR," no "let's dive in," no marketing words (see the banned list in [style.md](style.md)).
- Em-dashes rare; prefer periods and commas.

## 10. Linking

- **Go deeper** links point to blog posts (`/posts/...`), at a section anchor when one fits.
- Cross-link sheets where a concept lives elsewhere ("Go deeper: [Generative Models cheatsheet](/cheatsheets/generative-models)") rather than duplicating a card.
- Every link must resolve; the publish check verifies blog links and anchors.

## 11. Order within a sheet

- Order cards the way the ideas build: core concept first, then variants, then practical concerns (tuning, failure modes, evaluation).
- For a pipeline topic (RAG, the AV stack), follow the pipeline order.
- End with a "which one when" table or a short gotchas card if the sheet has natural choices.

## 12. Adding a sheet

1. Create `src/pages/cheatsheets/<id>.md` with:

   ```yaml
   ---
   layout: ../../layouts/CheatsheetLayout.astro
   title: "<Topic> in 2 Minutes Each"
   description: "Intuition cards for <list of the main concepts>."
   ---
   ```

2. One short intro line under the title (who/what the sheet covers). No more.
3. Write the cards.
4. Add the sheet to the right group in `src/pages/cheatsheets/index.astro`.
5. Build and check: no KaTeX errors, no stray `$` (escape literal dollars as `\$`), links resolve.

## 13. Checklist

- [ ] Would a working ML engineer learn or relearn something from every card? Cut any that are basics.
- [ ] Every card opens with an unlabeled definition sentence and has a **Remember** line.
- [ ] Analogies are accurate and not repeated.
- [ ] At most one formula per card; none where words suffice.
- [ ] Every number verified or removed.
- [ ] Cards 40–120 words; sheet 10–20 cards.
- [ ] Go-deeper links resolve; overlapping concepts cross-link instead of duplicating.
- [ ] Sheet is listed in the index.
