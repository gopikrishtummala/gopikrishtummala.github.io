---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-15T00:00:00Z
modDatetime: 2025-01-15T00:00:00Z
title: 'Agentic AI Design Patterns — Part 4: Failure Modes & Safety'
slug: agentic-ai-design-patterns-part-4
featured: true
draft: false
tags:
  - agentic-ai
  - llm
  - design-patterns
  - ai-systems
  - machine-learning
  - artificial-intelligence
description: 'Part 4 of a comprehensive guide to agentic AI design patterns. Covers common failure modes, safety mechanisms, verifiable pipelines, and how to build reliable production systems.'
---

*By Gopi Krishna Tummala*

---

<div class="series-nav" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Agentic AI Design Patterns Series</div>
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;">
    <a href="/posts/agentic-ai-design-patterns-part-1" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 1: Foundations</a>
    <a href="/posts/agentic-ai-design-patterns-part-2" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 2: Production</a>
    <a href="/posts/agentic-ai-design-patterns-part-3" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 3: Specialized</a>
    <a href="/posts/agentic-ai-design-patterns-part-4" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Part 4: Failure Modes</a>
    <a href="/posts/agentic-ai-design-patterns-part-5" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 5: Production Guide</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Part 4: Failure Modes & Safety</strong> — Engineering reality: how agents fail and how to prevent it</div>
</div>

Remember the toddler with a credit card? This section is about all the ways they can mess up, and how we stop them.

Just as software must account for concurrency and exceptions (like "what if two people try to buy the last item?"), agentic AI must anticipate these common, repeatable failures. Understanding failure modes is crucial for building production-ready agent systems.

The good news? They fail in predictable ways. The bad news? You have to plan for all of them.

---

<div id="article-toc" class="article-toc">
  <div class="toc-header">
    <h3>Table of Contents</h3>
    <button id="toc-toggle" class="toc-toggle" aria-label="Toggle table of contents"><span>▼</span></button>
  </div>
  <div class="toc-search-wrapper">
    <input type="text" id="toc-search" class="toc-search" placeholder="Search sections..." autocomplete="off">
  </div>
  <nav class="toc-nav" id="toc-nav">
    <ul>
      <li><a href="#tool-overuse">A. The "Tool Overuse" Trap</a></li>
      <li><a href="#contextual-amnesia">B. The Contextual Amnesia Loop</a></li>
      <li><a href="#goal-drift">C. The Goal Drift Problem</a></li>
      <li><a href="#hallucinated-api">D. The Hallucinated API Call</a></li>
      <li><a href="#infinity-loop">E. The Infinity Loop</a></li>
      <li><a href="#premature-termination">F. Premature Termination</a></li>
      <li><a href="#verifiable-pipelines">G. Verifiable Agent Pipelines</a></li>
      <li><a href="#safety-planning">H. Safety-Aware Planning</a></li>
      <li><a href="#failure-summary">Summary: Failure Mode Mitigation</a></li>
    </ul>
  </nav>
</div>

---

## **A. The "Tool Overuse" Trap** {#tool-overuse}

### **Failure:**

The agent defaults to calling a tool (e.g., web search) even when the answer is in its context or memory. This wastes tokens, latency, and API costs.

**Example:** User asks "What is 2+2?" and the agent calls a calculator tool instead of using its internal knowledge.

### **Mitigation (Pattern #3):**

Implement a **"Triage" Prompt**—a meta-step before $\text{Thought}_t$ that explicitly asks the LLM to decide: **Internal Knowledge vs. Tool Use**.

```python
def triage_step(query: str, context: str) -> str:
    """Decide if tool use is necessary"""
    decision = llm.invoke(
        f"Query: {query}\n"
        f"Context: {context}\n\n"
        "Can this be answered from context alone? "
        "Respond: INTERNAL or TOOL_NEEDED"
    )
    return decision

# Use before tool selection
if triage_step(user_query, agent_memory.retrieve(user_query)) == "INTERNAL":
    return llm.invoke(user_query)  # No tool call
else:
    return agent.select_and_call_tool(user_query)
```

---

## **B. The Contextual Amnesia Loop** {#contextual-amnesia}

### **Failure:**

The LLM's finite context window forces it to "forget" crucial observations from $t-5$ steps ago, leading to re-planning or repeating failed actions.

**Example:** Agent searches for flights, finds results, but 10 steps later forgets the search results and searches again.

### **Mitigation (Pattern #6):**

Implement **Structured Working Memory ($M_{work}$)**. Force the agent to distill the core findings of every $N$ steps into a structured JSON/YAML object that *always* gets injected into the next prompt.

```python
class WorkingMemory:
    def __init__(self):
        self.facts: Dict[str, Any] = {}
        self.decisions: List[str] = []
    
    def distill_step(self, observations: List[str], step_num: int):
        """Compress observations into structured facts"""
        if step_num % 5 == 0:  # Every 5 steps
            summary = llm.invoke(
                f"Observations: {observations}\n"
                "Extract key facts as JSON: {fact: value, ...}"
            )
            self.facts.update(json.loads(summary))
    
    def inject_into_prompt(self, base_prompt: str) -> str:
        """Always include working memory in prompt"""
        memory_context = f"""
Working Memory:
{json.dumps(self.facts, indent=2)}

Recent Decisions:
{self.decisions[-3:]}
"""
        return f"{base_prompt}\n\n{memory_context}"
```

---

## **C. The Goal Drift Problem (The Agent's "Shiny Object Syndrome")** {#goal-drift}

### **Failure:**

The agent gets distracted by an interesting sub-problem and loses sight of the original, top-level goal ($g$).

**Example:** Goal is "Book a flight to Austin," but agent gets sidetracked researching hotel prices and never books the flight.

### **Mitigation (Pattern #2):**

Enforce the **Plan-Execute-Reflect (PER)** structure. The Planner's output is immutable for $k$ steps. The Reflector's primary job is to check the current output against the *original $g$*, not just the sub-task.

```python
class GoalAwareReflector:
    def __init__(self, original_goal: str):
        self.original_goal = original_goal
        self.plan: List[str] = []
    
    def reflect(self, current_state: Dict) -> Dict:
        """Check if we're still aligned with original goal"""
        reflection = llm.invoke(
            f"Original Goal: {self.original_goal}\n"
            f"Current Plan: {self.plan}\n"
            f"Current State: {current_state}\n\n"
            "Are we still working toward the original goal? "
            "If not, what corrective action is needed?"
        )
        
        if "DRIFT_DETECTED" in reflection:
            # Reset to original goal
            return {"action": "reset_to_goal", "goal": self.original_goal}
        return {"action": "continue"}
```

---

## **D. The Hallucinated API Call** {#hallucinated-api}

### **Failure:**

The LLM invents a non-existent tool name or generates correct code with entirely fabricated function arguments.

**Example:** Agent calls `search_flights_api(destination="Austin", date="2025-01-20")` but the actual API requires `to_city` and `departure_date`.

### **Mitigation (Pattern #17):**

Utilize **Pydantic/Instructor Pattern** for all tool calls. Force the tool-call LLM to output a JSON object strictly conforming to a defined schema. If the JSON parsing fails (a non-LLM error), trigger the **Compensatory Reflex** to generate a corrected JSON structure.

$$
\text{Tool Call} = \text{parse}_{\text{pydantic}}(\pi_{\text{tool}}(o_t))
$$

```python
from pydantic import BaseModel, ValidationError
from instructor import patch

class FlightSearchTool(BaseModel):
    """Search for flights - schema enforced"""
    to_city: str  # Not "destination"!
    departure_date: str  # Format: YYYY-MM-DD
    from_city: str = "SFO"  # Default

client = patch(ChatOpenAI())

def safe_tool_call(user_request: str) -> FlightSearchTool:
    """Tool call with automatic schema correction"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                response_format=FlightSearchTool,
                messages=[{"role": "user", "content": user_request}]
            )
            return response.parsed  # Type-safe!
        except ValidationError as e:
            if attempt < max_retries - 1:
                # Reflex: fix the schema error
                user_request = f"{user_request}\n\n"
                f"Previous error: {e}\n"
                "Generate a corrected request matching the schema."
            else:
                raise
```

---

## **E. The Infinity Loop (The Circular Argument)** {#infinity-loop}

### **Failure:**

The agent falls into a closed loop, e.g., $\text{Action}_1$ fails $\rightarrow \text{Thought}_2$ says "Try $\text{Action}_1$ again" $\rightarrow \text{Action}_1$ fails...

**Example:** Agent tries to call an API, gets 404, thinks "maybe the URL is wrong," tries again with same URL, gets 404 again, repeats.

### **Mitigation:**

Implement **Episodic Memory Pruning** and a **Backtrack Limit**. Store a hash of the last three actions/thoughts. If the current sequence matches a recent pattern, trigger a hard reflex action like $\text{Action}_{\text{backtrack}}$ or $\text{Action}_{\text{reset\_plan}}$.

```python
import hashlib
from collections import deque

class LoopDetector:
    def __init__(self, window_size: int = 3):
        self.action_history: deque = deque(maxlen=window_size)
        self.seen_patterns: set = set()
    
    def detect_loop(self, current_action: str) -> bool:
        """Check if we're repeating a pattern"""
        # Add current action
        self.action_history.append(current_action)
        
        # Create pattern hash
        pattern = " -> ".join(self.action_history)
        pattern_hash = hashlib.md5(pattern.encode()).hexdigest()
        
        if pattern_hash in self.seen_patterns:
            return True  # Loop detected!
        
        self.seen_patterns.add(pattern_hash)
        return False

# Use in agent loop
loop_detector = LoopDetector()
if loop_detector.detect_loop(action):
    # Trigger backtrack or reset
    action = backtrack_or_reset()
```

---

## **F. Premature Termination** {#premature-termination}

### **Failure:**

Agent thinks it's done when it's not. Returns incomplete results.

**Example:** Task is "Find and summarize 10 papers," but agent stops after finding 3.

### **Mitigation (Pattern #18):**

Implement **Completion Verification** using introspective agents:

```python
def verify_completion(task: str, result: str) -> bool:
    """Check if task is actually complete"""
    verification = llm.invoke(
        f"Task: {task}\n"
        f"Result: {result}\n"
        "Is this task complete? Respond: COMPLETE or INCOMPLETE"
    )
    return "COMPLETE" in verification
```

---

## **G. Verifiable Agent Pipelines (Safety & Grounding)** {#verifiable-pipelines}

LLM output is stochastic (probabilistic). Modern systems are designed for explicit verification:

* **Tool-Grounded Cross-Check:** Any factual claim must be checked against a trusted tool (search, database, code execution).

* **Prediction with Uncertainty:** Agents should express their confidence score, making trust explicit. This is crucial for high-stakes tasks.

$$
\text{confidence} = p_\theta(y | x)
$$

* **Safety-Aware Planning:** Agents actively avoid actions with high-risk predicted outcomes by incorporating a risk model into the planning phase.

$$
\text{risk}(a) = \mathbb{E}[\text{negative outcome} | s, a]
$$

The planner is constrained to select a trajectory $\tau$ where the maximum predicted risk is below a defined threshold.

### **Implementation:**

```python
class VerifiableAgent:
    def __init__(self):
        self.llm = ChatOpenAI()
        self.code_executor = CodeExecutor()
        self.uncertainty_estimator = UncertaintyModel()
    
    def generate_with_verification(self, prompt: str):
        """Generate output with automatic verification"""
        # Generate response
        response = self.llm.invoke(prompt)
        
        # Extract factual claims
        claims = extract_claims(response)
        
        # Verify each claim
        verified_claims = []
        for claim in claims:
            if self.verify_fact(claim):
                verified_claims.append(claim)
            else:
                # Remove unverified claim or flag it
                response = remove_claim(response, claim)
        
        # Estimate confidence
        confidence = self.uncertainty_estimator.estimate(response)
        
        return {
            "response": response,
            "confidence": confidence,
            "verified_claims": verified_claims
        }
    
    def verify_facts(self, facts: List[str]) -> Dict[str, bool]:
        """Verify facts against trusted sources"""
        results = {}
        for fact in facts:
            # Check against search, database, etc.
            search_result = web_search(fact)
            results[fact] = validate_against_source(fact, search_result)
        return results
```

---

## **H. Safety-Aware Planning** {#safety-planning}

Agents must assess risk before taking actions, especially in high-stakes environments.

### **Implementation:**

```python
class SafetyAwareAgent:
    def __init__(self):
        self.risk_estimator = RiskModel()
        self.safety_threshold = 0.1  # Max acceptable risk
    
    def select_safe_action(self, state, goal: str):
        """Select action with risk assessment"""
        # Generate candidate actions
        candidates = self.llm.propose_actions(state, goal)
        
        # Assess risk for each
        safe_actions = []
        for action in candidates:
            risk = self.estimate_risk(state, action)
            
            if risk < self.safety_threshold:
                safe_actions.append((action, risk))
            else:
                # Log high-risk action (don't execute)
                self.log_risk_event(state, action, risk)
        
        if not safe_actions:
            # No safe actions - request human intervention
            return self.request_human_guidance(state, goal)
        
        # Select safest action
        return min(safe_actions, key=lambda x: x[1])[0]
    
    def estimate_risk(self, state, action) -> float:
        """Estimate risk of negative outcomes"""
        # Risk model predicts probability of negative outcomes
        risk = self.risk_estimator.predict(state, action)
        
        # Risk formula
        risk_score = (
            0.4 * risk.data_loss +
            0.3 * risk.security_breach +
            0.2 * risk.performance_degradation +
            0.1 * risk.user_harm
        )
        
        return risk_score
```

### **Risk Assessment:**

$$
\text{risk}(a) = \mathbb{E}[\text{negative outcome} | s, a]
$$

Agents avoid actions where $\text{risk}(a) > \text{threshold}$.

---

## **Summary: Failure Mode Mitigation Patterns** {#failure-summary}

| Failure Mode | Primary Mitigation Pattern | Key Technique |
|:---|:---|:---|
| Tool Overuse | Pattern #3 (Toolformer) | Triage prompt |
| Contextual Amnesia | Pattern #6 (Memory) | Structured working memory |
| Goal Drift | Pattern #2 (PER) | Goal-aware reflection |
| Hallucinated API Calls | Pattern #17 (Reflexes) | Pydantic schema enforcement |
| Infinity Loops | Pattern #17 (Reflexes) | Loop detection + backtracking |
| Premature Termination | Pattern #18 (Introspection) | Completion verification |
| Unverifiable Outputs | Pattern #18 (Introspection) | Multi-layer verification |
| High-Risk Actions | Pattern #15 (Imagination) | Risk-aware planning |

These mitigation strategies transform theoretical patterns into production-ready safeguards.

---

# **Part III: Engineering Reality — Safety, Verification, and Failure Taxonomy**

Agent engineering is mostly failure management. For these systems to leave the lab, we must design for trustworthiness.

## **1. Verifiable Agent Pipelines**

LLM output is stochastic (probabilistic). Modern systems are designed for explicit verification:

* **Tool-Grounded Cross-Check:** Any factual claim must be checked against a trusted tool (search, database, code execution).

* **Prediction with Uncertainty:** Agents should express their confidence score, making trust explicit. This is crucial for high-stakes tasks.

$$
\text{confidence} = p_\theta(y | x)
$$

* **Safety-Aware Planning:** Agents actively avoid actions with high-risk predicted outcomes by incorporating a risk model into the planning phase.

$$
\text{risk}(a) = \mathbb{E}[\text{negative outcome} | s, a]
$$

The planner is constrained to select a trajectory $\tau$ where the maximum predicted risk is below a defined threshold.

## **2. Failure Taxonomy in the Wild**

| Failure Mode | Description | Mitigation Pattern |
| :--- | :--- | :--- |
| **Contextual Amnesia** | Forgetting crucial context due to context window limits. | Pattern #6 (Memory Rewriting), Structured Working Memory. |
| **Goal Drift** | Getting distracted by an interesting sub-task. | Pattern #2 (Reflector) constantly checks against original $g$. |
| **Hallucinated API** | Inventing a non-existent tool or argument fields. | Pattern #17 (Reflexes), Pydantic/Schema validation for tool calls. |
| **Grounding Failure** | Generating an action impossible in the environment (e.g., trying to grasp an unreachable object). | Pattern #14 (3D Scene Graph) for pre-action feasibility checks. |

---

<div class="series-nav" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-top: 3rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; justify-content: space-between;">
    <a href="/posts/agentic-ai-design-patterns-part-3" style="background: rgba(255,255,255,0.1); padding: 0.75rem 1.5rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">← Previous: Part 3</a>
    <a href="/posts/agentic-ai-design-patterns-part-5" style="background: rgba(255,255,255,0.25); padding: 0.75rem 1.5rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Next: Part 5: Production Guide →</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">Learn about 2025 trends, cost optimization, case studies, and the production checklist</div>
</div>

