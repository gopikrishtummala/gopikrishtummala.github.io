---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-15T00:00:00Z
modDatetime: 2025-01-15T00:00:00Z
title: 'Agentic AI Design Patterns — Part 04: Failure Modes & Safety'
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
track: Agentic AI
difficulty: Advanced
interview_relevance:
  - System Design
  - ML-Infra
estimated_read_time: 30
---

*By Gopi Krishna Tummala*

---

<div class="series-nav" style="background: linear-gradient(135deg, #6366f1 0%, #9333ea 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
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
      <li><a href="#exception-handling">I. Exception Handling and Recovery</a></li>
      <li><a href="#human-in-loop">J. Human-in-the-Loop (HITL)</a></li>
      <li><a href="#failure-taxonomy">K. Failure Taxonomy</a></li>
      <li><a href="#failure-summary">Summary: Failure Mode Mitigation</a></li>
    </ul>
  </nav>
</div>

---

<a id="tool-overuse"></a>
## **A. The "Tool Overuse" Trap**

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

<a id="contextual-amnesia"></a>
## **B. The Contextual Amnesia Loop**

### **Failure:**

The LLM's finite context window forces it to "forget" crucial observations from $t-5$ steps ago, leading to re-planning or repeating failed actions.

**Example:** Agent searches for flights, finds results, but 10 steps later forgets the search results and searches again.

### **Mitigation (Pattern #7):**

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

<a id="goal-drift"></a>
## **C. The Goal Drift Problem (The Agent's "Shiny Object Syndrome")**

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

<a id="hallucinated-api"></a>
## **D. The Hallucinated API Call**

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

<a id="infinity-loop"></a>
## **E. The Infinity Loop (The Circular Argument)**

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

<a id="premature-termination"></a>
## **F. Premature Termination**

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

<a id="verifiable-pipelines"></a>
## **G. Verifiable Agent Pipelines (Safety & Grounding)**

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

<a id="safety-planning"></a>
## **H. Safety-Aware Planning**

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

<a id="failure-summary"></a>
## **Summary: Failure Mode Mitigation Patterns**

| Failure Mode | Primary Mitigation Pattern | Key Technique |
|:---|:---|:---|
| Tool Overuse | Pattern #3 (Toolformer) | Triage prompt |
| Contextual Amnesia | Pattern #7 (Memory) | Structured working memory |
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

<a id="exception-handling"></a>
## **2. Exception Handling and Recovery**

When things go wrong, agents need graceful recovery mechanisms. Unlike traditional software where exceptions are caught and handled, agentic systems must **detect, understand, and recover from failures autonomously**.

### **The Simple Idea:**

Think of exception handling like a safety net. When an agent tries to do something and it fails, instead of crashing, it should:
1. **Detect the failure** (tool call failed, API error, invalid output)
2. **Understand what went wrong** (analyze the error message, check logs)
3. **Recover gracefully** (retry with different parameters, try alternative approach, or escalate to human)

### **Common Exception Scenarios:**

**1. Tool/API Failures**
- **Problem:** External API is down, rate limited, or returns unexpected format
- **Recovery:** Retry with exponential backoff, try alternative tool, or use cached data

**2. Invalid Output**
- **Problem:** LLM generates malformed JSON, invalid code, or nonsensical response
- **Recovery:** Validate output against schema, request regeneration, or fallback to simpler approach

**3. Context Overflow**
- **Problem:** Conversation history exceeds context window
- **Recovery:** Compress/summarize old messages, use memory retrieval, or reset with key facts

**4. Goal Unreachable**
- **Problem:** Task cannot be completed with available tools/resources
- **Recovery:** Break into smaller sub-tasks, request additional permissions, or escalate to human

### **Implementation Pattern:**

```python
def robust_agent_step(goal: str, tools: list, max_retries: int = 3):
    """Agent step with exception handling"""
    for attempt in range(max_retries):
        try:
            # Attempt the action
            result = agent.execute(goal, tools)
            
            # Validate result
            if validate_output(result):
                return result
            else:
                raise ValueError("Invalid output format")
                
        except ToolError as e:
            # Tool-specific error
            if attempt < max_retries - 1:
                # Try alternative tool
                alternative_tool = find_alternative(tools, e.failed_tool)
                tools = [alternative_tool] + [t for t in tools if t != e.failed_tool]
                continue
            else:
                # Escalate to human
                return request_human_intervention(goal, e)
                
        except ValidationError as e:
            # Output validation failed
            if attempt < max_retries - 1:
                # Request regeneration with stricter constraints
                goal = add_validation_constraints(goal)
                continue
            else:
                return request_human_intervention(goal, e)
                
        except Exception as e:
            # Unknown error
            log_error(e)
            if attempt < max_retries - 1:
                # Simplify the goal and retry
                goal = simplify_goal(goal)
                continue
            else:
                return request_human_intervention(goal, e)
    
    return None  # All retries exhausted
```

**Key Principle:** Agents should fail gracefully, learn from errors, and know when to ask for help.

---

<a id="human-in-loop"></a>
## **3. Human-in-the-Loop (HITL) Pattern**

Not every decision should be fully automated. The **Human-in-the-Loop** pattern integrates human judgment at critical decision points, ensuring AI systems remain aligned with human values, ethics, and goals.

### **The Simple Idea:**

Think of HITL like having a supervisor review important decisions. The agent does most of the work autonomously, but for critical choices—especially those involving ethics, high risk, or ambiguity—it pauses and asks a human.

### **When to Use HITL:**

**1. High-Stakes Decisions**
- Financial transactions above a threshold
- Medical diagnoses or treatment recommendations
- Legal document generation
- Content moderation decisions

**2. Ambiguous Situations**
- When confidence is low (< 70%)
- When multiple valid interpretations exist
- When user intent is unclear

**3. Ethical Boundaries**
- Content that might be harmful or biased
- Decisions affecting people's lives or livelihoods
- Creative work that requires human judgment

**4. Learning and Improvement**
- Collecting human feedback for model refinement
- Correcting errors to improve future performance
- Validating novel approaches

### **HITL Interaction Patterns:**

**1. Human Oversight**
- **What:** Monitor agent performance in real-time via dashboards
- **When:** Continuous monitoring for adherence to guidelines
- **Example:** Review agent logs, check outputs before deployment

**2. Intervention and Correction**
- **What:** Human steps in when agent encounters errors or ambiguous scenarios
- **When:** Agent requests help or detects low confidence
- **Example:** Agent asks "Should I proceed with this transaction?" and waits for approval

**3. Human Feedback for Learning**
- **What:** Collect human preferences to refine agent behavior
- **When:** After agent actions, especially novel ones
- **Example:** "Was this response helpful?" → Use feedback to improve

**4. Decision Augmentation**
- **What:** Agent provides analysis, human makes final decision
- **When:** Complex decisions requiring human judgment
- **Example:** Agent analyzes market data and recommends trades, human approves execution

### **Implementation Example:**

```python
def agent_with_hitl(goal: str, confidence_threshold: float = 0.8):
    """Agent that requests human input when needed"""
    result = agent.execute(goal)
    confidence = result.confidence
    
    if confidence < confidence_threshold:
        # Request human review
        human_decision = request_human_review(
            goal=goal,
            agent_result=result,
            reason="Low confidence"
        )
        return human_decision
    
    if is_high_stakes(result):
        # Require human approval
        approved = request_human_approval(
            action=result.action,
            context=result.context
        )
        if approved:
            return execute_action(result)
        else:
            return request_alternative_approach(goal)
    
    return result
```

**Key Principle:** HITL ensures AI systems remain trustworthy, ethical, and aligned with human values while maintaining efficiency through selective human involvement.

---

<a id="failure-taxonomy"></a>
## **4. Failure Taxonomy in the Wild**

| Failure Mode | Description | Mitigation Pattern |
| :--- | :--- | :--- |
| **Contextual Amnesia** | Forgetting crucial context due to context window limits. | Pattern #8 (Memory Rewriting), Structured Working Memory. |
| **Goal Drift** | Getting distracted by an interesting sub-task. | Pattern #2 (Reflector) constantly checks against original $g$. |
| **Hallucinated API** | Inventing a non-existent tool or argument fields. | Pattern #17 (Reflexes), Pydantic/Schema validation for tool calls. |
| **Grounding Failure** | Generating an action impossible in the environment (e.g., trying to grasp an unreachable object). | Pattern #14 (3D Scene Graph) for pre-action feasibility checks. |
| **Exception Cascade** | One failure triggers multiple downstream failures. | Exception Handling pattern with circuit breakers and graceful degradation. |
| **Human Overload** | Too many HITL requests overwhelm human operators. | Smart escalation: only critical decisions require human input, use confidence thresholds. |

---

<div class="series-nav" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-top: 3rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; justify-content: space-between;">
    <a href="/posts/agentic-ai-design-patterns-part-3" style="background: rgba(255,255,255,0.1); padding: 0.75rem 1.5rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">← Previous: Part 3</a>
    <a href="/posts/agentic-ai-design-patterns-part-5" style="background: rgba(255,255,255,0.25); padding: 0.75rem 1.5rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Next: Part 5: Production Guide →</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">Learn about 2025 trends, cost optimization, case studies, and the production checklist</div>
</div>

