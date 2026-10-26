# Writing style for gopikrishnatummala.com

This blog is a **map**, not a textbook. Someone should be able to land on any post, spend ten minutes, and walk away knowing what a concept is, why it exists, how it works at the level of intuition, and where it fits next to everything else. If they want depth, we point them to the paper, the docs, or a longer post. We don't try to be the deepest source on the internet. We try to be the clearest.

Every post should read like it was written by one engineer who has done this work, explaining it to a smart colleague over coffee. Not like a course catalog, and not like a model's first draft.

This guide is the contract for new posts and for the cleanup of old ones.

---

## 1. What we learned from posts we admire

I read recent posts from three engineering blogs that people actually finish reading:

- **Netflix Tech Blog:** [measuring the blast radius of library changes](https://netflixtechblog.com/beyond-api-compatibility-building-a-tool-to-measure-the-blast-radius-of-library-changes-9474f2fd4f90).
- **Google Research blog:**
  - [Diffusion Controller](https://research.google/blog/how-diffusion-controller-unifies-and-simplifies-ai-image-generation/), steering image generation.
  - [Provably private federated learning](https://research.google/blog/toward-provably-private-learning-from-federated-data/).
- **Engineering at Meta:**
  - [The Thin Orchestrator](https://engineering.fb.com/2026/10/19/ml-applications/thin-orchestrator-ai-agent-architecture/), on AI agent architecture.
  - [GenRecs Tokenization](https://engineering.fb.com/2026/10/19/ml-applications/genrecs-tokenization-transforming-ads-recommendation-at-meta-with-stable-model-vocabulary/), on semantic IDs for recommendations.

They differ in purpose (tool write-up, research announcement, production story), but they share a lot.

### What all three do

- **Real authors, real roles.** Every post names the people who wrote it and their team. Writing feels human when a person is visibly behind it.
- **A one-sentence promise under the title.** Google and Meta open with a short summary line (a "dek") that says exactly what the post delivers. One sentence, no bullets.
- **The opening is a situation, not a definition.** Netflix opens with who has the problem and what they can't see. Google opens with a concrete prompt that goes wrong ("a lizard wearing sunglasses" comes out without the sunglasses, or with a distorted face). Meta opens with what they built first and how it failed in production.
- **One running example, revisited.** Netflix threads one removed method through the whole post. Google keeps coming back to the lizard. Meta's tokenization post keeps returning to a new season's trail shoe that inherits what the model learned from last season's. The example is introduced early and resolved at the end.
- **Plain, descriptive headings.** Each heading names what the section covers (how an API change maps to affected consumers, how the new design strengthens privacy, why the approach works). No wordplay, no numbering theatrics.
- **Terms are defined in place.** A parenthetical plain-English definition the first time a term appears (for example, explaining "remotely attestable" as "outsiders can check which code is running"). No glossary, no condescension.
- **Before and after, side by side.** "In our earlier system…, in the new system…" is the backbone of the Google privacy post. Contrast is how readers understand why something matters.
- **Small, labeled code and data.** Snippets are marked "simplified," "abridged" or "illustrative," and data points carry a footnote on where and when they were measured.
- **Honest limits.** Netflix spells out what "no references found" does *not* prove. Google notes its guarantees hold only within the limits of current secure-hardware technology. Meta says what the earlier approach got right before explaining why it failed.
- **A forward-looking close.** Each ends with what they're doing next, and most thank the people involved.
- **Roughly 1,500–2,500 words.**

### What each does especially well

- **Netflix: frame the problem as the reader's open questions.** A short list of questions the reader genuinely can't answer yet, which the rest of the post answers. Ends with four lessons, each a bold lead sentence plus two lines.
- **Google: one analogy, carried all the way.** The Diffusion Controller post introduces one image (a small steering damper added to a powerful motorcycle whose engine you don't touch) and uses it consistently. The analogy does real explanatory work and never gets a label.
- **Google: anchor in a real deployment.** The privacy post grounds everything in one shipped product (Gboard) and says what got better there.
- **Meta: tell the failure first.** They describe building the obvious single all-knowing agent, watching it work in demos and then fail in production. Then the fix, then *why* it works, then a "How to apply this to your own system" section of short imperative lessons. This is the most human shape of the three.
- **Meta: tiny inline analogies instead of big setups.** "Similar items share tokens, much as related words share roots." One clause, then move on.
- **Meta: a counterintuitive one-line hook.** One sentence up top that makes you want to read the rest, restated as the final lesson.

### What not to copy

Even these blogs slip into corporate polish: "paradigm shift," "seamlessly," "perfectly," "best of all," "state-of-the-art," and the occasional cute reversal line. Company blogs are partly announcements. This blog is a learning map written by one person, so we keep their structure and honesty and leave the marketing out.

---

## 2. What every post must do

Each post answers five questions, roughly in this order:

1. **What problem does this solve?** Start from a concrete situation where the old way hurt.
2. **What is the core idea?** One or two sentences a reader could repeat to a friend.
3. **How does it work, intuitively?** The mechanism at the level of "what moves where and why," not every detail.
4. **Where does it break or stop being the right tool?** Limits, trade-offs, when to use something simpler.
5. **Where does it fit?** What it connects to on this site and where to go for depth.

If a section doesn't serve one of these five, cut it or move it to its own post.

### Depth rule

Go **one level deeper than the reader expects, then stop.** Explain *why* PagedAttention helps (memory was wasted on reservations), not every scheduler state. If you catch yourself writing the third paragraph of implementation detail, replace it with a link.

- **Math:** at most one or two equations per post, only when the equation *is* the idea (for example, the KV-cache size formula). Always say in words what it means right after.
- **Numbers:** round them, give units, and either cite the source or label them "illustrative." Never invent precise-looking figures (no "1,657% speedup").
- **Code:** only short, runnable-looking snippets that show the shape of an API. Label them "simplified" if they are. Prefer 5 lines to 50.

---

## 3. Structure

### The default shape

```
Title
Byline + date
One-sentence dek: what the reader will understand by the end.
Series line (if part of a series)

Opening (2–4 short paragraphs): the situation, what hurt, what this post covers.
The running example, introduced.

## The idea in one picture
   One diagram or mind map, with a caption that states the takeaway.
   Sections below can each have their own diagram (see "Working with diagrams").

## 3–5 sections, each one concept
   Each: intuition first, mechanism in a few sentences, a return to the running example,
   and one honest limit.

## What I'd remember
   3–5 bold lead sentences with a line or two each.

## Where to go next
   2–5 links: deeper posts on this site, the paper, the docs.
```

### Pick one of three shapes

The default shape works for most concept posts. Two variations, borrowed from the posts above, fit specific topics better:

| Shape | Use when | Section flow |
|---|---|---|
| **Concept map** (default) | Explaining an idea or technique: PagedAttention, LoRA, flow matching | Situation → idea in one picture → 3–5 concepts → limits → what I'd remember → where next |
| **Before and after** (Google style) | Something replaced an older approach: FSDP vs DDP, Parquet vs CSV, continuous vs static batching | How it used to work and where it hurt → the new idea → what changed, side by side → where the old way is still fine |
| **What we tried** (Meta style) | A real system or decision you were part of, or a well-documented public case | What we did first → how it failed → the fix → why it works → how to apply it to your system → what's next |

Whichever shape you pick, keep it for the whole post. Don't stack all three.

### Length

- **Target 1,200–2,500 words** (about 6–12 minutes).
- A post over 3,000 words is two posts. Split it.
- Set `estimated_read_time` honestly: words ÷ 220, rounded.

### Openings

Good openings put the reader in a real situation:

> Before vLLM, an LLM server reserved enough GPU memory for every request's *maximum* length before generating a single token. Most of that memory sat empty, and new users waited in a queue for it. vLLM borrowed an idea from operating systems, handing out memory in small pages as each conversation grows, and the same GPU suddenly served several times more people.

Avoid:

- A TL;DR box of eight bullets before the post starts. If you want a summary, write one plain sentence at the end of the opening: "This post covers X, Y and Z."
- "Imagine…" analogies as the first line of every post.
- Restating the title.

Two openings that work well, from the posts above:

- **The failing example:** show one concrete case where the obvious approach breaks (the prompt that drops the sunglasses), then promise the fix.
- **The honest first attempt:** "The first thing I tried was X. It worked on the small case and fell apart at scale." Only if it's true.

A single counterintuitive hook sentence before the opening is fine (for example, "The component that matters most here is the one that does the least."), once, if the post earns it by the end.

### Endings

End with **What I'd remember** (lessons, Netflix-style: bold lead sentence, then one or two lines) and **Where to go next** (links). Optionally a short line thanking people or naming sources of the story.

For "what we tried" posts, add a short **How to apply this** section before the lessons (Meta style): 3–5 imperative sentences a reader can use on their own system, each with one line of why. For example: "Keep the coordinator dumb. Once it starts making domain decisions, it becomes the thing that fails."

Don't end with "This concludes the series" or a motivational line.

---

## 4. Voice

- **First person, and only true first person.** "I" for my own experience and opinions, "we" for team work I was part of. Never invent anecdotes ("I've seen teams spend weeks…") unless it actually happened. A real, small story beats a vivid fake one.
- **Label opinions as opinions.** "My rule of thumb is…" is honest. "The industry standard is…" needs a source.
- **Plain words.** "Use" not "leverage." "Fast" not "blazing." "Important" only when you can say why.
- **Short sentences, mostly.** Vary the rhythm, but if a sentence needs two em-dashes and a colon, it's two sentences.
- **Calm confidence.** No hype, no superlatives, no "this changes everything." Let the idea be impressive on its own.
- **Respect the reader.** Don't explain what a GPU is in an advanced post. Don't tell them a section "separates senior candidates."

### One analogy, used well

An analogy is a tool for the hardest idea in the post, not decoration for every section.

- **At most one main analogy per post.** Introduce it, use it to explain the hard part, and stay consistent with it (Google's steering damper on a motorcycle shows up in the intro, the method and the conclusion, always meaning the same thing).
- **Prefer tiny inline analogies to big setups.** One clause inside a sentence ("much as related words share roots") usually beats a paragraph that starts with "Imagine…".
- Don't label it ("The Feynman version:"). Just use it.
- If the analogy needs a paragraph of caveats, it's the wrong analogy.

### Define terms where they appear

The first time a term shows up, define it in a parenthesis or a short clause, in plain words: "KV cache (the keys and values of every earlier token, kept so they aren't recomputed)." No glossary, no "for the uninitiated." After that, use the term freely.

### Emphasis lines

A very short sentence on its own line ("It passed every test. Then real traffic arrived.") is a strong human device. Use it at most twice per post, for the turning point of the story. More than that reads as performance.

---

## 5. Formatting

| Element | Rule |
|---|---|
| **Headings** | Descriptive, sentence case: "Why the KV cache limits concurrency." No "Act 0 / Act I.V," no emoji, no "Mature Architecture." |
| **Bold** | At most one bold phrase per section, for the single thing to remember. Lesson leads in "What I'd remember" are the exception. |
| **Bullets** | Only for things that are actually a list (steps, options, questions). Explanations are paragraphs. For a list of named parts, use the "**Term:** one-line definition" form, as the Google and Meta posts do. |
| **Footnotes for data** | When a number comes from a specific measurement, say where and when ("measured on 8×H100, vLLM 0.x, October 2026") in a footnote or caption. |
| **Tables** | Only for real comparisons (at least 3 rows and 3 columns). Never as a substitute for explaining. At most two per post. |
| **Diagrams** | Diagrams are one of the best things on this site; a reader skimming the pictures should get the map. Open with one "idea in one picture" diagram, then put a diagram in any section where a picture explains the mechanism faster than prose. No fixed cap. Each one sits next to the paragraph it explains and gets an italic caption that states the takeaway. See "Working with diagrams" below. |
| **Mind maps** | A good opening picture for a post that is a map of several concepts (series overviews, roadmaps). Section diagrams can still follow it. |
| **Code** | Short, labeled "simplified" or "illustrative" when it is. Use real APIs and check they run or match current docs. |
| **Callouts / blockquotes** | Rare. For a real caveat or a quoted source, not for repeating the paragraph above. |
| **Emoji** | None in headings, diagrams or body text. |
| **Em-dashes** | Rare. Prefer a period or commas. |
| **HTML banners** | None. Series navigation is one plain line (see below). |

### Working with diagrams

When cleaning up a post, **never delete a diagram to hit a style rule.** A diagram usually holds information the prose doesn't, so restyle it instead:

- **Keep every fact.** If a node or edge carries information (a number, a name, a step, an arrow label), it stays, either in the diagram or moved into the caption or the paragraph next to it.
- **Restyle the labels.** Short, plain words; no emoji; sentence case; use `<br/>` to break long labels instead of cutting content.
- **Tie it to the running example** where it fits (relabel a generic "Request" as "the contract question"), but don't bend a diagram that's clearer generic.
- **Place it where it's explained.** Move a diagram next to the section that discusses it, and add a one-line caption that states the takeaway.
- **Merge, don't drop.** If two diagrams show the same thing, combine them into one that keeps the information from both.
- **Split if it's a wall.** A diagram with more than about 15 nodes is usually two diagrams, or one diagram plus a short list.
- **Keep it readable at page width.** A flowchart more than about five boxes wide shrinks its text; switch it to top-to-bottom, or add `direction TB` inside wide subgraphs.
- **Keep it renderable.** Avoid colons and special characters in Gantt task names, quote labels with punctuation, and check it in every site theme (light, dark, cyber, academic).

### Series navigation

Replace the gradient HTML banners with one line under the byline:

> Part 3 of [the MLOps map](/posts/mlops/infrastructure-first-mlops-roadmap). Previous: [Dataloaders](...). Next: [Post-training](...).

And the same Previous/Next line at the end.

---

## 6. Interview material

Interview prep is valuable, but it shouldn't take over concept posts.

- In a concept post, at most **one short "In an interview" section** (a paragraph or 2–3 questions with one-line answers). No "Interview one-liner" in the TL;DR, no "Graduate Assignment," no answer dropdowns in every post.
- Self-check questions, if any, go at the very end: **at most three**, phrased as questions to think about.
- Longer interview drills, scenario banks and cheat sheets live in dedicated posts in the interview-prep section, which link to the concept posts.

---

## 7. Honesty and sourcing

- **Every factual claim that a reader might repeat needs a source or a hedge.** Link the paper, docs or announcement. If you can't find one, soften it or cut it.
- **Date fast-moving facts.** "As of late 2026…" for model names, library versions, pricing, company practices.
- **Say what you don't know** and what a technique doesn't solve. One honest limit per section is a good habit.
- **Distinguish vendor claims from independent results.** "The authors report…" vs "independent benchmarks show…"
- **No private or employer-confidential details**, and no coaching notes addressed to yourself in published posts.

---

## 8. Phrases and patterns to remove

These are the tells that make writing read as machine-generated. Search for them during cleanup.

**Words and phrases:** delve, leverage, unlock, supercharge, game-changer, paradigm shift, seamless(ly), perfectly, best of all, state-of-the-art (unless citing a benchmark), robust (as filler), crucial, landscape, "in today's fast-paced world," "let's dive in," "the secret sauce," "here's the thing," "at the end of the day," "it's worth noting," "in conclusion," "the gold standard," "mature architecture," "the real unlock."

**Patterns:**

- **"It's not X, it's Y"** and **"not just X, but Y"** framing.
- **Colon reveals:** "The answer? Memory."
- **Stacked rhetorical questions.**
- **Triplets by reflex:** "faster, cheaper, and more reliable" when only one is true.
- **Every section ending on a punchline** or a bold one-liner.
- **Labeled devices:** "The Feynman version," "Plain English," "Key insight:" before every point.
- **Fake specificity:** precise numbers with no source.
- **Uniform scaffolding:** the same Act/TL;DR/Mature Architecture/Graduate Assignment skeleton on every post regardless of topic.

---

## 9. Before and after

**Heading**

- Before: `#### Act I.V: Mature Architecture — The "Gold Standard" Pipeline`
- After: `## How a production pipeline is laid out`

**Opening**

- Before: *"Imagine you are running a very busy hotel (The GPU). Every guest (User Request) needs a room to stay in… You can fit 10x more guests in the same building, and nobody has to wait in the lobby."*
- After: the vLLM opening in section 3: a real situation, the real mechanism, a hedged result.

**Claim**

- Before: *"Most AI models spend 90% of their time just waiting for data to arrive."*
- After: *"Big matrix multiplications keep the GPU busy, but most of the other operations in a model move a lot of data and do little math, so for those the cores mostly wait on memory."*

**Section ending**

- Before: *"This is the section that actually differentiates senior candidates."*
- After: no line at all. Let the content do the work.

---

## 10. Checklist before publishing

- [ ] There's a one-sentence dek under the title.
- [ ] The opening describes a real situation, and the post covers the five questions in section 2.
- [ ] The post follows one shape (concept map, before and after, or what we tried).
- [ ] Terms are defined in plain words where they first appear.
- [ ] There's one running example, and it comes back in most sections.
- [ ] One "idea in one picture" diagram (or mind map) with a takeaway caption.
- [ ] Every diagram from the old version is still there (restyled, merged or split, never silently dropped), next to the section it explains, with a caption.
- [ ] 1,200–2,500 words; read time set honestly.
- [ ] Headings are descriptive, with no "Act" numbering and no emoji.
- [ ] At most one analogy, used for the hardest idea.
- [ ] Every number is sourced or labeled illustrative; fast-moving facts are dated.
- [ ] Each major section names at least one limit or trade-off.
- [ ] Bold, bullets and tables are used sparingly and only where they fit (section 5).
- [ ] No phrases or patterns from section 8.
- [ ] Code snippets are short, labeled and checked against current APIs.
- [ ] Ends with "What I'd remember" and "Where to go next."
- [ ] Interview material is one short section at most.
- [ ] Read it out loud. If a sentence sounds like a press release, rewrite it.

---

## 11. Plan for the cleanup

When we clean up existing posts, in this order:

1. **Template-level fixes across all posts:** replace gradient series banners with the plain series line; strip emoji from headings; rename "Act" headings to descriptive ones; remove "Mature Architecture" and "Graduate Assignment" boilerplate (move worthwhile exercises to interview-prep posts).
2. **Trim the longest posts first:** split anything over 3,000 words into a map post plus deep dives, or cut the deep parts and link out.
3. **Rewrite openings and endings** to the shapes in section 3.
4. **Fact pass:** remove or source unsourced numbers; date fast-moving facts.
5. **Voice pass:** search for the section 8 list and rewrite those sentences.
6. **Diagram pass:** keep every existing diagram; restyle labels, add captions, and move each next to the section it explains (see "Working with diagrams").
7. **Interview material:** consolidate scattered self-checks and interview scenarios into dedicated interview-prep posts per series.
