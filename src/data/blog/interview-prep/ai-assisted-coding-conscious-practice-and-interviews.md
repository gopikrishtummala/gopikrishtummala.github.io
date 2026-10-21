---
author: Gopi Krishna Tummala
pubDatetime: 2026-10-21T00:00:00Z
modDatetime: 2026-10-21T00:00:00Z
title: "Coding With AI Without Losing Your Edge: Conscious AI-Assisted Engineering and the New AI-Enabled Interview"
slug: ai-assisted-coding-conscious-practice-and-interviews
featured: true
draft: false
tags:
  - interview-prep
  - ai-coding
  - software-engineering
  - learning
  - productivity
  - coding-interviews
description: "What the research actually says about AI coding tools (faster? slower? worse at learning?), a conscious workflow that keeps you in charge, the mistakes that quietly cost teams, and a concrete plan for the new AI-enabled coding interviews at Meta and elsewhere."
track: Fundamentals
difficulty: Intermediate
interview_relevance:
  - Coding
  - Behavioral
  - System Design
estimated_read_time: 35
---

*By Gopi Krishna Tummala*

---

> **Who this is for.** Engineers who already use Copilot, Cursor, Claude Code or ChatGPT every day and have a nagging feeling: *"Am I getting better, or just getting faster at producing code I don't fully understand?"* And anyone preparing for the new generation of coding interviews where an AI assistant sits in the editor next to you.

**Quick jump:** [The evidence](#act-i-what-the-research-actually-says) · [The conscious loop](#act-ii-the-conscious-loop) · [Daily workflow](#act-iii-a-workflow-that-keeps-you-in-charge) · [Getting better, not worse](#act-iv-using-ai-to-get-better-not-worse) · [Mistakes](#act-v-the-mistakes-that-quietly-cost-you) · [AI-enabled interviews](#act-vi-the-ai-enabled-coding-interview) · [4-week prep plan](#act-vii-a-four-week-preparation-plan) · [ML-specific traps](#act-viii-for-ml-engineers-the-silent-bug-problem) · [Checklists](#checklists) · [Self-check](#check-yourself)

## Table of contents

---

## TL;DR

- **AI is an amplifier, not an equalizer.** Google's DORA 2025 report found that AI magnifies the strengths of good teams *and* the dysfunctions of struggling ones. The same is true for individuals: strong engineers get stronger; engineers who outsource their thinking get weaker.
- **Feeling faster is not being faster.** In METR's 2025 randomized trial, experienced open-source developers were **19% slower** with AI tools on their own repos, yet *believed* they were ~20% faster. Newer tools may well change the number; the perception gap is the lesson.
- **Delegating code can cost you the skill.** Anthropic's 2026 randomized study found developers learning a new library with AI scored **~17 percentage points lower** on a follow-up quiz, with the biggest gap in **debugging**. But how people used AI mattered: those who asked *conceptual* questions or asked for *explanations* scored far better than those who just delegated.
- **"Almost right" is the core failure mode.** In Stack Overflow's 2025 survey, 84% of developers use or plan to use AI, but more distrust its accuracy (46%) than trust it (33%), and the top frustration (66%) is solutions that are *almost right, but not quite*.
- **AI code needs a security and supply-chain review.** Veracode found vulnerabilities in **45%** of AI-generated code samples across 80 tasks; a USENIX Security 2025 study found models invent non-existent package names (5.2% for commercial models, 21.7% for open models), and attackers register those names.
- **The conscious loop:** *Think → Specify → Generate small → Verify → Understand → Own.* You own the problem framing, the design, the verification and the final judgment. AI accelerates everything in between.
- **Interviews changed.** Meta replaced one onsite coding round with a **60-minute AI-enabled round** in CoderPad on a multi-file codebase; other companies allow AI in some rounds, and many still ban it. AI-enabled rounds grade **judgment, verification, communication and ownership**, not prompt magic.
- **Interview one-liner:** *"I use AI like a fast junior pair: I frame the problem and the plan, delegate small well-specified pieces, verify everything with tests and reading, and I never ship code I can't explain."*

---

## Act 0: AI Coding in Plain English

Think about GPS.

Before GPS, you had to *build a map in your head*. You knew that the river is east, the highway runs north, and if you miss the exit you can loop back through downtown. GPS made driving dramatically easier. But people who follow turn-by-turn directions for years often can't navigate their own city without it. And when GPS is confidently wrong ("turn left into the lake"), the driver who never built a mental map just... turns left.

AI coding tools are GPS for code.

- Used **consciously**, they're a superpower: you still know where you're going and why; the tool handles the tedious turns.
- Used **unconsciously**, they slowly erase your mental map of your own system. The code works until the day it doesn't, and then you're debugging something you never understood.

There's a second analogy I like even more: **a brilliant, fast, overconfident junior engineer** who has read every Stack Overflow answer ever written, never gets tired, never says "I don't know," and has never seen *your* codebase, *your* users or *your* production incidents. You would never let that person merge to `main` without review. You'd give them clear tasks, check their work and teach them your conventions. That's exactly the relationship to build with AI.

Keep both pictures in mind. Everything below is about keeping your mental map while still letting the GPS drive the boring parts.

---

## Act I: What the Research Actually Says

There is a lot of hype in both directions. Here is what careful studies found, with the caveats that matter.

| Study | What they did | Finding | Caveat |
|---|---|---|---|
| **METR RCT (2025)** | 16 experienced open-source devs, 246 real tasks on repos they'd maintained ~5 years | **19% slower** with AI (mostly Cursor + Claude 3.5/3.7 Sonnet), while *estimating* a ~20% speedup | Small sample, early-2025 tools, very familiar codebases. METR's 2026 follow-up showed signs of speedups but with measurement problems |
| **Google internal RCT** | 96 Google engineers on an enterprise task | ~21% faster with AI | Wide confidence interval |
| **Anthropic skill-formation RCT (2026)** | 52 mostly junior engineers learning the Trio async library, then a quiz | AI group **~17 points lower** on the quiz (≈50% vs 67%), biggest gap on debugging; time savings not significant | Small sample; immediate comprehension only; chat sidebar, not an agent |
| **Stack Overflow Dev Survey (2025)** | ~49,000 developers | 84% use or plan to use AI; 46% distrust accuracy vs 33% trust; 66% cite "almost right" answers | Self-reported |
| **DORA 2025 (Google Cloud)** | ~5,000 professionals | 90% use AI, >80% feel more productive, ~30% have little or no confidence in AI code; **AI amplifies** existing strengths and dysfunctions | Survey + qualitative |
| **Veracode GenAI Code Security (2025)** | 100+ LLMs, 80 coding tasks | **45%** of cases introduced an OWASP Top 10 vulnerability; XSS and log injection failed most of the time; bigger models weren't meaningfully more secure | Synthetic tasks |
| **Package hallucination (USENIX Security 2025)** | 16 models, 576K code samples | 5.2% (commercial) / 21.7% (open) of suggested packages **don't exist**; 43% of hallucinated names reappear on reruns | Attackers can pre-register them ("slopsquatting") |
| **GitClear code quality (2025)** | 211M changed lines, 2020–2024 | Duplicated 5+ line blocks rose ~8× in 2024; "moved" (refactored) code fell from ~25% to under 10% of changes | Correlational |

### What I take from this

1. **The perception gap is real.** You *feel* fast because typing disappears. But time moves to reading, reviewing, re-prompting and debugging "almost right" code. Measure your own outcomes (cycle time, bugs, rework), not your vibes.
2. **Learning is the hidden cost.** The Anthropic study is the most important one for anyone early in a domain. The *way* you use AI decides whether you learn. More on this in Act IV.
3. **The danger is not dramatic failures, it's quiet ones.** Duplicated code, insecure defaults, hallucinated dependencies and code nobody on the team can explain. None of them fail a demo.
4. **Context matters enormously.** AI helps most on unfamiliar boilerplate and well-specified tasks, and least on deep, familiar, highly-constrained codebases where you already know the answer faster than you can explain it.

---

## Act II: The Conscious Loop

Here is the mental model I use. Six steps; the AI only drives one of them.

```mermaid
flowchart LR
  T["1. Think<br/>What problem? What does<br/>'done' look like?"] --> S["2. Specify<br/>Constraints, interfaces,<br/>tests, examples"]
  S --> G["3. Generate small<br/>One function, one file,<br/>one step"]
  G --> V["4. Verify<br/>Run tests, read the diff,<br/>try edge cases"]
  V --> U["5. Understand<br/>Can I explain every line?<br/>Why this approach?"]
  U --> O["6. Own<br/>Simplify, rename, document,<br/>commit as mine"]
  V -.->|"wrong or 'almost right'"| S
  U -.->|"don't understand"| A["Ask: explain, compare<br/>alternatives, cite docs"]
  A --> U
```

**Who owns what:**

| You own (never delegate) | AI accelerates (delegate freely) |
|---|---|
| Problem framing: what are we actually solving? | Boilerplate, glue code, config, scripts |
| Architecture and interfaces | First drafts of well-specified functions |
| Choosing *which* tests prove correctness | Drafting test cases (which you then review) |
| Security, data and privacy decisions | Explaining unfamiliar code, APIs, error messages |
| Reading and accepting every line | Searching a large codebase, summarizing history |
| The final "this is correct" judgment | Refactors with a clear spec and tests as a safety net |

### The delegation matrix

A simple way to decide how much to hand off: **how risky is the code** × **how well do I understand the domain**.

| | **I know this domain well** | **This is new to me** |
|---|---|---|
| **Low risk** (scripts, tests, internal tools) | Delegate generously; skim and run | Delegate, but ask for explanations; this is a cheap place to learn |
| **High risk** (auth, payments, data pipelines, concurrency, infra) | Write the core yourself or delegate tiny pieces; review line by line | **Slow down.** Learn first (docs + conceptual questions), write it yourself, use AI as a reviewer |

The bottom-right box is where people get hurt: high-risk code in an unfamiliar domain, generated fast, accepted because it looks plausible.

---

## Act III: A Workflow That Keeps You in Charge

These practices come from my own work and line up closely with the [Claude Code best practices](https://code.claude.com/docs/en/best-practices) guide. They apply to any assistant.

### 1. Explore, then plan, then code

Don't let the assistant start typing immediately. First have it **read** the relevant code and answer questions ("how do we handle sessions today? where are secrets loaded?"). Then ask for a **plan**: which files change, the data flow, the edge cases. Edit the plan yourself. Only then implement. If you could describe the change in one sentence, skip the plan.

### 2. Give it a way to verify its own work

The single biggest upgrade. Instead of *"write a function that validates emails"*, say *"write `validate_email`; these should pass: `a@b.co` → True, `a@.com` → False, `''` → False; run the tests after implementing."* A test suite, a type checker, a build or a script that diffs output against a fixture turns "looks done" into "is done." **If you can't verify it, don't ship it.**

### 3. Be specific: scope, sources, patterns, symptoms

| Vague | Specific |
|---|---|
| "add tests for foo.py" | "test foo.py for the logged-out user case; no mocks" |
| "fix the login bug" | "login fails after session timeout; check token refresh in `src/auth/`; write a failing test that reproduces it, then fix" |
| "add a widget" | "follow the pattern in `HotDogWidget`; no new libraries" |
| "the build is failing" | "build fails with [error]; fix the root cause, don't suppress the error" |

### 4. Work in small, reviewable steps

Ask for one function or one file at a time. A 600-line AI diff is unreviewable; you'll skim it and approve it, and that's exactly how bugs and duplicated code slip in.

### 5. Course-correct early; reset when polluted

If you've corrected the same mistake twice, the conversation is full of failed attempts and the model is anchored on them. Start a fresh session with a better prompt that includes what you learned. Long, meandering sessions degrade quality.

### 6. Separate the writer from the reviewer

Have one session write the code and a *fresh* session (or a different model, or a human) review it against the spec: edge cases, race conditions, consistency with the codebase. A model reviewing code it just wrote is biased toward it, just like we are.

### 7. Make your conventions persistent

Keep a short project instructions file (`CLAUDE.md`, `.cursorrules`, Copilot instructions): build and test commands, code-style rules that differ from defaults, architecture decisions, gotchas. Keep it short; long files get ignored. Things that must happen *every* time (lint, format, block edits to migrations) belong in hooks or CI, not in prose.

### 8. Prompts I actually use

```text
Before writing code: restate the problem, list assumptions, and ask me
anything ambiguous. Then propose 2 approaches with trade-offs. Don't code yet.
```

```text
Implement only step 2 of the plan. Keep the public interface unchanged.
Add tests for: empty input, duplicates, and a 1e6-element input.
Run them and show me the output.
```

```text
Review this diff as a skeptical senior engineer. Report only issues that
affect correctness, security, or the stated requirements: line, problem, fix.
No style nits.
```

```text
Explain this function line by line as if I'm new to asyncio. What would
break if I removed the lock on line 14?
```

---

## Act IV: Using AI to Get Better, Not Worse

This is the part most people skip. The Anthropic study is worth re-reading here, because it identified **interaction patterns** that predicted who learned:

| Pattern | What they did | Quiz score |
|---|---|---|
| **AI delegation** | Had AI write all the code | Low (<40%), even though fastest |
| **Progressive reliance** | Started asking questions, then handed everything over | Low |
| **Iterative AI debugging** | Asked AI to fix each error instead of understanding it | Low, and slowest |
| **Generation then comprehension** | Generated code, then asked follow-up questions to understand it | High (~86%) |
| **Hybrid code + explanation** | Asked for code *with* explanations | High (~68%) |
| **Conceptual inquiry** | Asked only conceptual questions, fixed errors themselves | High (~65%), and second-fastest |

(Small groups; association, not causation. But the direction is consistent with everything we know about learning.)

**The Feynman version:** you learn what you *struggle* with. If the AI does the struggling, the AI learns nothing (it's frozen) and you learn nothing either.

### Habits that keep you sharp

1. **Attempt first, then compare.** Give yourself 10–15 minutes on a problem before asking. Then compare your approach to the AI's and ask *why* they differ.
2. **Ask for hints, not answers.** *"Don't give me code. What concept am I missing? Give me a hint toward the next step."*
3. **Explain it back.** After accepting code, explain it out loud (or in the PR description) without looking. If you can't, you don't own it yet.
4. **Predict, then run.** Before running AI code, predict what it will print or which test will fail. Wrong predictions are where the learning is.
5. **Debug it yourself first.** The biggest gap in the Anthropic study was debugging. When a test fails, form a hypothesis and check it before pasting the stack trace into a chat.
6. **Read the docs it cites.** Ask *"which documentation supports this?"* and open it. It's also how you catch hallucinated APIs.
7. **Keep unplugged reps.** A few times a week, solve a problem with AI fully off. This is your "can I still navigate without GPS?" test, and it's exactly what traditional interview rounds still measure.
8. **Keep a mistakes log.** Every time AI code was wrong in a way you almost missed, write down the pattern. That log becomes your personal review checklist.
9. **Use learning modes.** Several assistants now have explanatory or learning modes that explain as they go or leave pieces for you to write. Use them when you're in a new domain.

---

## Act V: The Mistakes That Quietly Cost You

| Mistake | What it looks like | Why it hurts | Do this instead |
|---|---|---|---|
| **Accept without reading** | Tab-tab-tab, merge | "Almost right" code ships; you lose your mental map | Read every line; small diffs only |
| **Vibe coding into production** | Prototype grown by prompting until it "works" | No design, no tests, nobody understands it | Vibe-code to explore; rewrite deliberately for prod |
| **Trusting imports** | `pip install` whatever the model suggests | Hallucinated or typo-squatted packages ("slopsquatting") | Verify every new dependency exists, is maintained and is the one you meant; pin and lock |
| **Insecure defaults** | String-built SQL, unescaped HTML, permissive CORS, logging user input | 45% vulnerability rate in Veracode's tests | Security checklist + SAST in CI; ask explicitly for secure patterns |
| **Tests that test nothing** | AI writes code *and* tests; tests mirror the implementation or mock the thing under test | Green CI, broken behavior | Write or specify the test cases yourself; check that a test fails when you break the code |
| **Symptom suppression** | "Fix the error" → wrapped in `try/except: pass`, test skipped, type ignored | Bug hidden, not fixed | Ask for the root cause; forbid suppression in the prompt |
| **Copy-paste sprawl** | Five near-identical helpers in five files | Duplication up, refactoring down (GitClear) | Ask "does this already exist in the codebase?"; refactor as you go |
| **Over-engineering** | Abstract factories for a 20-line script; defensive code for impossible cases | Complexity you must maintain | Ask for the simplest solution; delete aggressively |
| **Context pollution** | One endless chat for five unrelated tasks | Quality drops; old mistakes leak into new code | One task per session; reset after repeated corrections |
| **Secrets and data leakage** | Pasting `.env`, customer data or proprietary code into an unapproved tool | Security and compliance incidents | Follow your org's approved tools; redact; use enterprise settings |
| **Debugging by slot machine** | Paste error → paste fix → new error → repeat | Slow, and you learn nothing | Form a hypothesis; reproduce with a failing test first |
| **Outsourcing design** | "Build me a scalable system for X" | Generic architecture that ignores your constraints | You design; AI critiques and fills in |
| **Measuring by feel** | "I'm so much faster now" | METR's perception gap | Track cycle time, review rework, escaped bugs |

---

## Act VI: The AI-Enabled Coding Interview

### The landscape (as of late 2026)

- **Meta** started piloting an **AI-enabled coding round** in October 2025: 60 minutes in a CoderPad environment with a built-in AI chat assistant, replacing one of the onsite coding rounds. Instead of two algorithm puzzles, you get a **multi-file codebase** with partially implemented features, failing tests and bugs, and you work through a sequence of tasks (debug this, then extend that). Reports say you can switch between several frontier models, the assistant can't edit your code directly (you write or paste), and senior candidates may see it as their only coding round. Official guidance reportedly says AI use is optional.
- **Other companies** vary a lot: some (reportedly Canva and Shopify, among others) allow or expect AI in some rounds or take-homes and explicitly grade how you verify AI output; many large companies still run classic algorithm rounds with AI disabled; CoderPad's 2026 data suggests about a third of hiring teams still ban AI entirely.
- **Always ask your recruiter** which rounds allow AI, which tools and models, and whether you can use your own IDE. Policies change quickly and vary by team.

### What's actually being evaluated

Across the published guidance and prep material, the rubric converges on:

1. **Problem understanding before prompting.** Do you read the code, the tests and the task first?
2. **Judgment about delegation.** Do you hand off well-defined subtasks and keep design decisions?
3. **Verification.** Do you test, read the output critically and catch the AI's mistakes?
4. **Communication.** Can you narrate what you asked, what you kept, what you changed and why?
5. **Ownership.** Can you explain and defend every line of the final code?

Notice what's *not* on the list: clever prompts. The AI is the same for every candidate. **The differentiator is you.**

### A 60-minute game plan

```mermaid
gantt
  title AI-enabled round: one way to spend 60 minutes
  dateFormat mm
  axisFormat %M
  section Orient
  Read task, code, tests (no AI)     :a1, 00, 8m
  Plan out loud, agree on approach   :a2, after a1, 4m
  section Build
  Task 1: debug / small fix          :b1, after a2, 12m
  Task 2: feature, AI for subtasks   :b2, after b1, 20m
  section Prove
  Run tests, add edge cases          :c1, after b2, 10m
  Walk through and simplify          :c2, after c1, 6m
```

**Minutes 0–12: Orient without AI.** Read the task, skim the file structure, run the existing tests to see what fails. Say out loud what you think the code does. You can ask the AI to *summarize* an unfamiliar file, but form your own picture first. Then state a plan.

**Minutes 12–44: Build in small, verified steps.**
- For debugging: reproduce first (which test fails, what's the actual vs expected). Form a hypothesis. Use the AI to check a hypothesis or explain an API, not to "fix it."
- For features: you decide the design and interfaces; delegate well-defined pieces ("write a function that parses this log line format into this dataclass; here are three example lines").
- Read every line before you paste it. Say what you're changing ("it used a list; I'm switching to a set because we check membership in a loop").

**Minutes 44–60: Prove it and own it.** Run the full test suite. Add edge cases (empty input, duplicates, large input, invalid input). Ask the AI to *suggest* more edge cases, and evaluate them. Simplify anything convoluted. Finish with a short walkthrough: what you built, trade-offs, what you'd do with more time.

### What good narration sounds like

> "Before I prompt anything, let me run the tests... two failures, both in `inventory.py`. My guess is the off-by-one in the window logic. I'll confirm with a print... yes. I'll fix this one by hand since it's one line."

> "For the CSV export I know exactly what I want, so I'll have the assistant draft it. I'm giving it the dataclass and the expected header row... It used `csv.writer` but didn't handle commas inside fields. Let me switch to `QUOTE_MINIMAL`... and add a test for that."

> "The assistant suggested a caching layer here. That's more than we need for this task; I'll keep it simple and mention caching as a follow-up."

### Red flags interviewers notice

- Prompting before reading anything; asking the AI to "solve the whole problem."
- Pasting code you haven't read; being unable to explain a line when asked.
- Long prompt-tweaking loops for trivial things you could type in 20 seconds.
- No tests run, or only the tests that were already there.
- Blaming the AI ("it gave me bad code") instead of fixing the prompt or the code.
- The opposite failure: refusing to use AI at all and running out of time on boilerplate. The round is designed to reward using it *well*.

### Don't forget the classic rounds

Many loops still include traditional algorithm rounds with AI disabled, and even the AI round assumes you can recognize a correct algorithm when you see one. **You can only verify what you could have written.** Keep your pattern fluency: see [The 45-Minute Interview Flow](/posts/interview-prep/interview-problem-solving-flow) and [Python Interview Hacks 2026](/posts/fundamentals/python-interview-hacks-2026).

---

## Act VII: A Four-Week Preparation Plan

### Week 1: Baseline and fundamentals (AI off)

- Solve 8–10 medium problems **without AI** using a structured flow (clarify → examples → approach → complexity → code → test).
- Time yourself. Write down where you get stuck: that's your learning list.
- Read one unfamiliar open-source module per day and explain it in five sentences, without AI. This is the "reading code" muscle the AI round depends on.

### Week 2: The verification drills

- **Spot-the-bug drill.** Ask an AI to solve a problem *and deliberately include one subtle bug* (off-by-one, wrong boundary, mutation of input, wrong complexity). Find it without running the code first, then confirm with tests.
- **Test-first drill.** For each problem, write the test cases *before* asking the AI for an implementation. Check whether the AI's code passes your tests, and whether your tests would catch a broken version.
- **Review drill.** Take an AI-generated 100-line solution and do a written review: correctness, edge cases, complexity, readability. Compare with a second model's review.

### Week 3: Multi-file, codebase-style practice (AI on)

- Clone a small real project (a CLI tool, a tiny web service, a data-processing script). Break a few things on purpose, or have a friend do it.
- Practice the Meta-style flow: read → run tests → fix a bug → add a feature across files → add tests, **in 60 minutes**, narrating out loud. Record yourself.
- Use the AI only through chat (no inline autocomplete) to match the interview setup, and paste code yourself.

### Week 4: Full mocks

- Two or three timed mock interviews with a friend (or a mock-interview service) in an AI-enabled environment. Ask them to grade on the five rubric dimensions above.
- One or two classic algorithm mocks with AI off.
- Refine your narration: what you ask the AI, why, and what you change.

### Using AI as a practice partner (without cheating yourself)

```text
Act as a Meta-style interviewer for an AI-enabled coding round. Give me a
small multi-file Python project (3-4 files) with one failing test, one bug,
and a feature request. Do NOT give hints unless I ask. After I finish,
grade me on: problem understanding, delegation judgment, verification,
communication, and ownership, with specific examples.
```

```text
I'm practicing. Here's my solution. Don't rewrite it. Ask me three questions
an interviewer would ask about its correctness, complexity, and edge cases.
```

### A self-scoring rubric

| Dimension | 1 (weak) | 3 (solid) | 5 (strong) |
|---|---|---|---|
| Understanding | Prompted immediately | Read the task and code first | Ran tests, stated hypotheses, clarified ambiguity |
| Delegation | Asked AI to solve it all | Delegated pieces | Delegated the right pieces with precise specs; kept design |
| Verification | No tests | Ran given tests | Added edge-case tests; caught and fixed AI mistakes |
| Communication | Silent | Explained occasionally | Narrated decisions and trade-offs throughout |
| Ownership | Couldn't explain code | Explained most | Simplified AI code; defended every line |

---

## Act VIII: For ML Engineers, the Silent-Bug Problem

ML code is especially dangerous to generate unconsciously, because most ML bugs **don't crash**; they just make the model quietly worse.

| Silent bug | How AI code often introduces it | How to catch it |
|---|---|---|
| Broadcasting mistakes | `(B,1) - (B,)` silently becomes `(B,B)` | Assert shapes at every boundary |
| Data leakage | Fitting the scaler or tokenizer on train+test; random split on time series | Review the split and fit order explicitly |
| Wrong loss reduction | `mean` vs `sum` across padded tokens; forgetting the padding mask | Overfit a single batch; check loss on a hand-computed example |
| Eval on the wrong mode | Missing `model.eval()` / `torch.no_grad()`; dropout on during eval | Run eval twice; results must match |
| Off-by-one in sequence shifts | Labels not shifted for next-token prediction | Print one decoded example of (input, target) |
| Distributed subtleties | Metrics averaged on rank 0 only; samplers not re-seeded per epoch | Compare 1-GPU vs N-GPU results on a small run |
| Silent dtype issues | bf16 accumulation, integer division, NaNs masked by `nan_to_num` | Assert finite values; track loss scale |

The ML version of "give it a way to verify" is **invariants**: shape assertions, overfit-one-batch tests, golden examples with hand-computed losses, and 1-GPU vs multi-GPU parity checks. These are also great things to say out loud in an ML coding interview. (For the infrastructure side of these pipelines, see [Datasets & Dataloaders](/posts/mlops/datasets-and-dataloaders) and [the Ray Field Guide](/posts/mlops/ray-field-guide-core-to-production).)

---

## Checklists

### Before you accept AI code

- [ ] I can explain what every line does and why it's there.
- [ ] Tests exist that would **fail** if this code were wrong (I checked by breaking it).
- [ ] Edge cases covered: empty, null, duplicates, large inputs, invalid input, concurrency if relevant.
- [ ] Every new dependency exists, is the intended package and is pinned.
- [ ] No secrets, unsafe string-built queries, unescaped output or overly broad permissions.
- [ ] No suppressed errors, skipped tests or blanket `except`.
- [ ] It reuses existing helpers instead of duplicating them.
- [ ] It's the simplest version that meets the requirements.

### Weekly "am I still growing?" check

- [ ] Did I solve at least a few problems with AI off this week?
- [ ] Did I learn one new concept deeply (not just use it)?
- [ ] What did AI get wrong this week that I almost missed? (Add it to the mistakes log.)
- [ ] Are my PRs smaller and better-tested, or just bigger?

---

## Check Yourself

<details>
<summary><strong>Why might an experienced developer be slower with AI on their own codebase?</strong></summary>

They already have a strong mental model, so typing was never the bottleneck. Time shifts to writing prompts, waiting, reading and reviewing plausible-but-wrong output, and re-prompting. On large, highly-constrained codebases with implicit conventions, the model lacks context the developer has, so its suggestions need more correction. Meanwhile, the removal of typing makes the work *feel* faster.

</details>

<details>
<summary><strong>What usage patterns preserved learning in Anthropic's study?</strong></summary>

Conceptual inquiry (asking only conceptual questions and fixing errors yourself), asking for code together with explanations, and generating code then asking follow-up questions to understand it. Full delegation, progressive reliance and having the AI fix every error were associated with low comprehension scores.

</details>

<details>
<summary><strong>What is slopsquatting and how do you defend against it?</strong></summary>

LLMs sometimes recommend packages that don't exist, and often the same fake names repeatedly. Attackers register those names with malicious code. Defend by verifying every new dependency (exists, maintained, expected publisher), pinning and locking versions, using private mirrors or allowlists, and scanning dependencies in CI.

</details>

<details>
<summary><strong>How do you make sure AI-written tests actually test something?</strong></summary>

Specify the test cases yourself (inputs and expected outputs from the requirements, not from the implementation), avoid mocking the unit under test, and do a quick mutation check: break the code on purpose and confirm the tests fail.

</details>

<details>
<summary><strong>In an AI-enabled interview, what do you do in the first 10 minutes?</strong></summary>

Read the task and the code, run the existing tests to see what fails, form hypotheses, and state a plan out loud. Use the AI at most to summarize unfamiliar files. Don't ask it to solve the problem.

</details>

<details>
<summary><strong>What would you do if the AI gives you a solution you don't fully understand during the interview?</strong></summary>

Don't paste it. Ask it to explain the approach and the tricky lines, check it against a small example by hand, simplify it, or write a simpler version yourself. Say this out loud: it's exactly the judgment the round is testing.

</details>

<details>
<summary><strong>When should you write code yourself even though AI is available?</strong></summary>

When it's faster to type than to specify (one-liners, small fixes), when it's high-risk code in a domain you're still learning, when you're deliberately practicing a skill, and when the design decision *is* the work.

</details>

<details>
<summary><strong>Name three silent ML bugs AI code can introduce, and the invariant that catches each.</strong></summary>

Broadcasting errors (shape assertions), data leakage from fitting preprocessing on test data (review split/fit order; time-based splits), and missing `model.eval()` (deterministic eval: two runs must match). Others: loss reduction over padding (hand-computed golden example) and metric aggregation across ranks (1-GPU vs N-GPU parity).

</details>

---

## References and Further Reading

**Research**
- METR, [Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/) (and coverage in [eWeek](https://eweek.com/news/news-ai-tools-slow-developer-productivity-study)).
- Anthropic, [How AI assistance impacts the formation of coding skills](https://www.anthropic.com/research/AI-assistance-coding-skills) (2026).
- [Stack Overflow Developer Survey 2025: AI](https://survey.stackoverflow.co/2025/ai).
- Google Cloud, [DORA 2025: State of AI-assisted Software Development](https://www.dora.dev/research/ai) ([announcement](https://cloud.google.com/blog/products/ai-machine-learning/announcing-the-2025-dora-report)).
- Veracode, [2025 GenAI Code Security Report](https://www.veracode.com/blog/genai-code-security-report).
- Spracklen et al., package hallucinations at USENIX Security 2025, summarized by [Dark Reading](https://www.darkreading.com/application-security/ai-code-tools-widely-hallucinate-packages) and [TechRepublic](https://www.techrepublic.com/article/news-slopsquatting-vibe-coding-ai-cybersecurity-risk/).
- GitClear, AI Copilot Code Quality 2025, summarized by [DevClass](https://devclass.com/2025/02/20/ai-is-eroding-code-quality-states-new-in-depth-report/).

**Workflow guides**
- Anthropic, [Best practices for Claude Code](https://code.claude.com/docs/en/best-practices).

**AI-enabled interviews**
- interviewing.io, [How to use AI in Meta's AI-assisted coding interview](https://interviewing.io/blog/how-to-use-ai-in-meta-s-ai-assisted-coding-interview-with-real-prompts-and-examples).
- [Inside Meta's New AI-Enabled Coding Round](https://dglearning.substack.com/p/inside-metas-new-ai-enabled-coding).
- [AI Tool Policy Tracker: Which Tech Companies Allow AI in Interviews (2026)](https://www.techinterview.org/post/3233474911/ai-tool-policy-tracker-tech-companies-2026/) (third-party; confirm with recruiters).
- formation.dev, [AI-assisted coding interviews](https://formation.dev/blog/ai-assisted-coding-interviews).

**On this site**
- [The 45-Minute Interview Flow](/posts/interview-prep/interview-problem-solving-flow) · [Python Interview Hacks 2026](/posts/fundamentals/python-interview-hacks-2026) · [Agentic AI Design Patterns](/posts/agentic-ai/agentic-ai-design-patterns) · [Agentic MLOps & MCP](/posts/mlops/beyond-inference-agentic-mlops-mcp).
