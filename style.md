# Writing style for gopikrishnatummala.com

This blog is a **map**, not a textbook. Someone should be able to land on any post, spend ten minutes, and walk away knowing what a concept is, why it exists, how it works at the level of intuition, and where it fits next to everything else. If they want depth, we point them to the paper, the docs, or a longer post. We don't try to be the deepest source on the internet. We try to be the clearest.

Every post should read like it was written by one engineer who has done this work, explaining it to a smart colleague over coffee. Not like a course catalog, and not like a model's first draft.

This guide is the contract for new posts and for the cleanup of old ones.

---

## 1. What we learned from posts we admire

The reference point is engineering writing like the Netflix Tech Blog's post on measuring the blast radius of library changes ([link](https://netflixtechblog.com/beyond-api-compatibility-building-a-tool-to-measure-the-blast-radius-of-library-changes-9474f2fd4f90)). What makes it read as human and trustworthy:

- **It starts with a real situation.** The first paragraph names who has the problem and what they can't see. No "imagine a hotel," no hype, no summary box.
- **The problem is stated as the reader's open questions.** A short list of the questions the reader genuinely can't answer yet. The rest of the post answers them.
- **One running example carries the whole post.** A single concrete case (removing one deprecated method) is introduced early, revisited in every section, and resolved at the end. Concepts are shown happening *to* that example.
- **Headings say what the section is about.** Plain, descriptive phrases. No numbering theatrics, no emoji.
- **It is honest about limits.** It says what the system does *not* do and what a result does *not* prove. That honesty is most of the credibility.
- **Code and data are small and labeled.** Snippets are marked "simplified" or "illustrative," trimmed to the lines that make the point.
- **Diagrams have captions that state the takeaway.** The caption tells you what to notice, not just what the picture is.
- **It ends with lessons, then what's next.** Four short lessons, each a bold lead sentence plus two lines of explanation. Then where the work is going. Then thanks to the people involved.
- **It's about 2,000 words.** Long enough to be useful, short enough to finish.
- **The voice is calm.** Short declarative sentences. Few adjectives. No superlatives. "We built," "we learned," "we want to."

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
Series line (if part of a series)

Opening (2–4 short paragraphs): the situation, what hurt, what this post covers.
The running example, introduced.

## The idea in one picture
   One diagram or mind map, with a caption that states the takeaway.

## 3–5 sections, each one concept
   Each: intuition first, mechanism in a few sentences, a return to the running example,
   and one honest limit.

## What I'd remember
   3–5 bold lead sentences with a line or two each.

## Where to go next
   2–5 links: deeper posts on this site, the paper, the docs.
```

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

### Endings

End with **What I'd remember** (lessons, Netflix-style: bold lead sentence, then one or two lines) and **Where to go next** (links). Optionally a short line thanking people or naming sources of the story.

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

- **At most one main analogy per post.** Introduce it, use it to explain the hard part, then drop it.
- Don't label it ("The Feynman version:"). Just use it.
- If the analogy needs a paragraph of caveats, it's the wrong analogy.

---

## 5. Formatting

| Element | Rule |
|---|---|
| **Headings** | Descriptive, sentence case: "Why the KV cache limits concurrency." No "Act 0 / Act I.V," no emoji, no "Mature Architecture." |
| **Bold** | At most one bold phrase per section, for the single thing to remember. Lesson leads in "What I'd remember" are the exception. |
| **Bullets** | Only for things that are actually a list (steps, options, questions). Explanations are paragraphs. |
| **Tables** | Only for real comparisons (at least 3 rows and 3 columns). Never as a substitute for explaining. At most two per post. |
| **Diagrams** | One "idea in one picture" diagram per post, plus at most one more if it earns its place. Every diagram gets a caption that states the takeaway. Keep labels short; diagrams must read in every site theme. |
| **Mind maps** | Use one *instead of* the main diagram when the post is a map of several concepts (series overviews, roadmaps). Never both. |
| **Code** | Short, labeled "simplified" or "illustrative" when it is. Use real APIs and check they run or match current docs. |
| **Callouts / blockquotes** | Rare. For a real caveat or a quoted source, not for repeating the paragraph above. |
| **Emoji** | None in headings, diagrams or body text. |
| **Em-dashes** | Rare. Prefer a period or commas. |
| **HTML banners** | None. Series navigation is one plain line (see below). |

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

**Words and phrases:** delve, leverage, unlock, supercharge, game-changer, seamless, robust (as filler), crucial, landscape, "in today's fast-paced world," "let's dive in," "the secret sauce," "here's the thing," "at the end of the day," "it's worth noting," "in conclusion," "the gold standard," "mature architecture," "the real unlock."

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

- [ ] The opening describes a real situation, and the post covers the five questions in section 2.
- [ ] There's one running example, and it comes back in most sections.
- [ ] One "idea in one picture" diagram (or mind map) with a takeaway caption.
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
6. **Interview material:** consolidate scattered self-checks and interview scenarios into dedicated interview-prep posts per series.
