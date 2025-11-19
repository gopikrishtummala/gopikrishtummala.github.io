---
author: Gopi Krishna Tummala
pubDatetime: 2025-01-15T00:00:00Z
modDatetime: 2025-01-15T00:00:00Z
title: 'Agentic AI Design Patterns — Part 1: Foundations'
slug: agentic-ai-design-patterns-part-1
featured: true
draft: false
tags:
  - agentic-ai
  - llm
  - design-patterns
  - ai-systems
  - machine-learning
  - artificial-intelligence
description: 'Part 1 of a comprehensive guide to agentic AI design patterns. Covers the fundamentals: ReAct loops, planning, tool use, self-consistency, and graph-based reasoning.'
---

*By Gopi Krishna Tummala*

---

<div class="series-nav" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Agentic AI Design Patterns Series</div>
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;">
    <a href="/posts/agentic-ai-design-patterns-part-1" style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">Part 1: Foundations</a>
    <a href="/posts/agentic-ai-design-patterns-part-2" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 2: Production</a>
    <a href="/posts/agentic-ai-design-patterns-part-3" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 3: Specialized</a>
    <a href="/posts/agentic-ai-design-patterns-part-4" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 4: Failure Modes</a>
    <a href="/posts/agentic-ai-design-patterns-part-5" style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white; opacity: 0.9;">Part 5: Production Guide</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">📖 You are reading <strong>Part 1: Foundations</strong> — Core patterns every agent needs</div>
</div>

# 🤖 The Cognitive Nexus: Agentic AI as the Engine of Complex Systems

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
      <li><a href="#introduction">Introduction: Beyond Generation to Autonomous Discovery</a>
        <ul>
          <li><a href="#mathematical-foundation">The Mathematical Foundation</a></li>
        </ul>
      </li>
      <li><a href="#pattern-1">Pattern #1: The ReAct Loop</a>
        <ul>
          <li><a href="#core-mechanism">The Core Mechanism</a></li>
          <li><a href="#mathematical-formulation">The Mathematical Formulation</a></li>
          <li><a href="#pseudo-code">From Theory to Code</a></li>
          <li><a href="#framework-code">Framework Implementation</a></li>
        </ul>
      </li>
      <li><a href="#pattern-overview">Pattern Overview: Beyond ReAct</a></li>
      <li><a href="#pattern-2">Pattern #2: Plan-Execute-Reflect (PER)</a></li>
      <li><a href="#pattern-3">Pattern #3: Tool Use</a></li>
      <li><a href="#pattern-4">Pattern #4: Self-Consistency Sampling</a></li>
      <li><a href="#pattern-5">Pattern #5: Graph-of-Thoughts (GoT)</a></li>
      <li><a href="#pattern-6">Pattern #6: Search-Augmented Agents</a></li>
      <li><a href="#domain-applications">Domain-Specific Applications</a>
        <ul>
          <li><a href="#gaming">Gaming and Creative Content</a></li>
          <li><a href="#scientific-discovery">Scientific Discovery</a></li>
          <li><a href="#product-design">Product and Engineering Design</a></li>
        </ul>
      </li>
      <li><a href="#references">References</a></li>
      <li><a href="#series-roadmap">What's Next: Series Roadmap</a></li>
    </ul>
  </nav>
</div>

<a id="introduction"></a>
## Introduction: Beyond Generation to Autonomous Discovery

The evolution of Artificial Intelligence has entered its most consequential phase: the transition from **Generative AI**—systems focused on producing single, static outputs (text, images)—to **Agentic AI**—autonomous systems capable of **multi-step planning, iterative execution, and self-evaluation** in dynamic environments. This shift repositions the Large Language Model (LLM) from a passive content creator to an active, goal-directed **Cognitive Engine**.

The core of Agentic AI lies in the continuous **Perceive → Plan → Act → Reflect** (PRAR) loop. This self-governing workflow, inspired by established AI paradigms, allows agents to:

1. **Decompose Complex Goals:** Break a high-level user objective (e.g., "Design a new CPU architecture" or "Find a novel antidepressant compound") into a logical sequence of actionable sub-tasks.

2. **Utilize Specialized Tools:** Dynamically select and orchestrate external functions, APIs, simulators, and databases with high precision, moving beyond simple web searches to complex **tool-use policies**. This process, known as **Tool Use** or **Tool Calling** (Yao et al., 2024; Schick et al., 2024), extends the agent's capabilities beyond its training data.

3. **Self-Correction and Learning:** Employ internal **Reflection** and **Self-Evaluation** mechanisms (Shinn et al., 2023) to critique intermediate results, identify errors (such as failed API calls or invalid outputs), and **iteratively refine** their strategy until the goal is achieved. This capacity for autonomous debugging is the key to enterprise-grade reliability.

This paradigm shift is not merely an efficiency gain; it is the establishment of a **Unified Agent Runtime** that fundamentally changes how we approach creative synthesis, scientific research, and immersive digital experiences. This approach transforms the LLM into a sequential decision-maker that reasons about the environment and selects actions to achieve a long-term goal (Wang et al., 2023).

<a id="mathematical-foundation"></a>
### **The Mathematical Foundation (Action Selection Policy)**

Formally, an agentic system's behavior is governed by a policy $\pi$:

$$
\pi(a_t | o_{\le t}, g, M)
$$

Where:

* $o_{\le t}$ = all observations (percepts, tool results) up to time $t$
* $g$ = the goal (high-level objective)
* $M$ = memory (episodic, semantic, and working context)
* $a_t$ = the action selected at time $t$ (e.g., thought, tool call, code execution)

This contrasts with a traditional LLM's static role as a next-token predictor.

---

<a id="pattern-1"></a>
## **Pattern #1 — The ReAct Loop**

The **ReAct (Reason and Act) loop** (Yao et al., 2023) is the foundational pattern that transforms LLMs from static predictors into sequential decision-makers. It's the "hello world" of agentic AI—deceptively simple, but getting this loop right is where 90% of the magic happens.

<a id="core-mechanism"></a>
### The Core Mechanism

ReAct enforces a structured, verifiable cycle that inherently reduces hallucination because every major internal decision must be followed by an external, factual check:

| Step | Pattern | Description | Function |
|:---|:---|:---|:---|
| 1. | **Thought/Reasoning** | The LLM analyzes the current observation, determines the next logical step, and decides if a tool is necessary. | Inference, Planning |
| 2. | **Action** | The agent executes the determined step, typically by calling a specialized function or API. | Execution, Tool Use |
| 3. | **Observation** | The agent receives the output or result from the action (e.g., the result of a code run or an API call). | Perception, Feedback |
| 4. | **Reflection** | An optional, but critical, meta-reasoning step where the agent critiques the Observation, identifies failures, and updates its internal plan for the next cycle. | Self-Correction, Debugging |

<a id="mathematical-formulation"></a>
### The Mathematical Formulation

The ReAct loop implements the policy $\pi(a_t | o_{\le t}, g, M)$ through this iterative process:

$$
\text{Thought}_t \rightarrow \text{Action}_t \rightarrow \text{Observation}_{t+1} \rightarrow \text{Reflection}_{t+1}
$$

Translation: **Think → Do → See what happened → Critique → Think again.**

### The ReAct Loop Flow

```mermaid
flowchart TD
    Start([User Query]) --> Thought[Generate Thought]
    Thought --> Action[Execute Action]
    Action --> Observation[Observe Result]
    Observation --> Check{Task Complete?}
    Check -->|No| Thought
    Check -->|Yes| End([Final Answer])
    
    style Thought fill:#e1f5ff
    style Action fill:#fff4e1
    style Observation fill:#e8f5e9
```

### **Example:**

Here's what it looks like inside the machine:

```
Thought: User wants cheapest flight to Austin.

Thought: I need prices, so I should search.

Action: call the flight-search tool with "Austin"

Observation: here are 50 flights, cheapest is Delta $187

Thought: That seems good, but let me double-check Southwest…

Action: call tool again with Southwest filter

Observation: actually $174!

Thought: Cool, that's the winner.

Final Answer: Book the Southwest flight for $174.
```

It's just the key-finding loop, but with airplane tickets instead of keys.

**Strengths:** You can see exactly what it's thinking (transparency). You can stop it if it goes wrong (controllability).

**Weakness:** Sometimes it talks too much and overthinks simple things. Like a teenager narrating every thought out loud.

<a id="pseudo-code"></a>
### From Theory to Code: The Pseudo-Code Bridge

Before diving into framework implementations, let's see how the PRAR loop maps directly to code logic:

```python
# Pseudo-code: The ReAct Loop Core Logic
def react_loop(goal: str, max_iterations: int = 10):
    """Core ReAct loop implementation"""
    observations = []
    memory = []
    
    for iteration in range(max_iterations):
        # 1. PERCEIVE: Gather all context
        context = build_context(goal, observations, memory)
        
        # 2. PLAN: Generate thought/reasoning
        thought = llm.generate_thought(context, goal)
        
        # 3. ACT: Decide if tool is needed and execute
        if needs_tool(thought):
            action = select_tool(thought, available_tools)
            observation = execute_tool(action)
            observations.append(observation)
        else:
            # Direct answer
            return thought
        
        # 4. REFLECT: Critique the observation
        reflection = llm.reflect(thought, observation, goal)
        
        # Check if goal is satisfied
        if is_goal_satisfied(reflection, goal):
            return extract_final_answer(reflection)
        
        # Update memory for next iteration
        memory.append((thought, action, observation, reflection))
    
    # Max iterations reached
    return "Task incomplete after max iterations"
```

This pseudo-code directly implements the policy $\pi(a_t | o_{\le t}, g, M)$: it takes observations, goal, and memory as input, and outputs the next action.

<a id="framework-code"></a>
### **Implementation: Framework Code**

Modern frameworks implement ReAct with a simple interface:

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI

# Initialize the agent with tools
llm = ChatOpenAI(model="gpt-4")
tools = [search_tool, calculator_tool, code_executor]

# Create ReAct agent
agent = create_react_agent(llm, tools)
agent_executor = AgentExecutor(agent=agent, max_iterations=10)

# Run the agent
result = agent_executor.invoke({
    "input": "Book me the cheapest flight to Austin"
})
```

The agent automatically alternates between reasoning (generating thoughts) and acting (calling tools) until it reaches a final answer or hits the iteration limit.

### **Citation:**

*Yao et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)*

---

<a id="pattern-overview"></a>
## Pattern Overview: Beyond ReAct

While ReAct is the foundation, production systems require additional patterns to handle complexity, reliability, and cost. Here's a brief overview of the other foundational patterns covered in this part:

---

<a id="pattern-2"></a>
## **Pattern #2 — Plan-Execute-Reflect (PER)**

Sometimes the omelette is more complicated: you want a three-course dinner for six people.

You don't want one frantic parrot running around the kitchen. You want:

- One calm **chef** who writes the full menu and timeline on a whiteboard (the Planner)

- Several **cooks** who actually chop onions and stir sauce (Executors)

- One annoying **food critic** who tastes everything and yells "THIS SOUP HAS NO SOUL!" (Reflector)

Only when the critic finally says "Okay, I guess this is edible" do you serve the guests.

This is the second big pattern: **Plan → Execute → Reflect**, repeated until the critic shuts up.

It sounds overkill, but it stops the agent from serving raw chicken because it got excited and skipped steps.

### **How It Works:**

Formally:

$$
\pi_{plan}, \pi_{exec}, \pi_{reflect}
$$

### **Plan-Execute-Reflect Flow:**

```mermaid
flowchart TD
    Start([Goal]) --> Planner[Planner LLM<br/>Create Sub-tasks]
    Planner --> Executor[Executor LLM<br/>Perform Tasks]
    Executor --> Reflector[Reflector LLM<br/>Critique Results]
    Reflector --> Check{Goal Satisfied?}
    Check -->|No| Executor
    Check -->|Yes| End([Final Output])
    
    style Planner fill:#e1f5ff
    style Executor fill:#fff4e1
    style Reflector fill:#f3e5f5
```

### **Implementation:**

LangGraph's StateGraph provides a clean abstraction for PER:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    goal: str
    plan: list[str]
    results: dict
    reflection: str

def planner(state: AgentState) -> AgentState:
    """Break goal into sub-tasks"""
    plan = llm.invoke(f"Create a plan for: {state['goal']}")
    return {"plan": parse_plan(plan)}

def executor(state: AgentState) -> AgentState:
    """Execute each task in the plan"""
    results = {}
    for task in state['plan']:
        results[task] = execute_task(task)
    return {"results": results}

def reflector(state: AgentState) -> AgentState:
    """Critique results against original goal"""
    reflection = llm.invoke(
        f"Goal: {state['goal']}\n"
        f"Results: {state['results']}\n"
        "Does this satisfy the goal? What's missing?"
    )
    return {"reflection": reflection}

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_node("reflector", reflector)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "reflector")
workflow.add_conditional_edges("reflector", 
    lambda x: "executor" if needs_rework(x) else END)
```

Modern frameworks like **OpenAI's Swarm**, LangGraph, and Instructor patterns use this.

### **Citation:**

*Shinn et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)*

---

<a id="pattern-3"></a>
## **Pattern #3 — Tools Are Just Extra Hands**

Imagine you're a carpenter with no arms. Someone glues a hammer, a saw, and a drill to long sticks and says "Here, use these."

At first you wave the sticks around like a drunk octopus. After a while you learn exactly when to pick the hammer-stick versus the saw-stick.

That's what "tool use" is for an AI. The tools are:

- Google search
- a calculator
- your email inbox
- a code-running sandbox
- the mouse and keyboard of your computer

The agent doesn't have them built in — they're just extra hands it can grab when needed.

The clever part: modern agents don't wait for you to say "use the calculator." They decide themselves, the same way you don't ask permission to pick up a hammer when you see a nail.

**In math-speak:**

$$
\arg\max_{a_t \in \{tools\}} \text{usefulness}(a_t | context)
$$

Translation: "Which tool is most useful right now?" The agent picks the best one automatically.

### Tools include:

* Web search

* Code execution

* Databases

* Embeddings

* Email/calendar

* Vision models

* Simulators

### **Implementation:**

Using Pydantic/Instructor pattern ensures type-safe tool calls:

```python
from pydantic import BaseModel
from instructor import patch

class SearchTool(BaseModel):
    """Search the web for information"""
    query: str
    max_results: int = 5

class CalculatorTool(BaseModel):
    """Evaluate a mathematical expression"""
    expression: str

# Patch the LLM to use structured outputs
client = patch(ChatOpenAI())
tools = [SearchTool, CalculatorTool]

# The LLM automatically chooses and calls tools
response = client.chat.completions.create(
    model="gpt-4",
    response_format=SearchTool,  # Enforces schema
    messages=[{"role": "user", "content": "Find recent papers on agentic AI"}]
)

# Type-safe tool call
search_result = response.parsed  # Automatically a SearchTool instance
```

This is crucial for enterprise AI.

### **Citation:**

*Schick et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools." [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)*

---

## **Tool Router Pattern (Simpler Alternative to RL-Based Tool Use)**

**Note:** Full PPO training on tool policies is rarely used in production (2025). Instead, most teams use a simple **Tool Router**—a lightweight classifier that decides which tool/specialist to call.

### **Why Tool Router Instead of RL:**

* **Simplicity:** Easy to implement and debug
* **Fast:** No training required, works immediately
* **Production-Ready:** Used in 90%+ of deployments
* **Effective:** 85-90% accuracy for tool selection

### **Implementation:**

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class ToolRouter:
    """Lightweight classifier for tool selection"""
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.tool_features = {}  # Pre-computed tool embeddings
    
    def route(self, query: str, available_tools: list) -> str:
        """Route query to best tool"""
        # Extract features from query
        features = self.extract_features(query)
        
        # Predict best tool
        tool_scores = {}
        for tool in available_tools:
            score = self.classifier.predict_proba([features])[0]
            tool_scores[tool] = score
        
        return max(tool_scores, key=tool_scores.get)
    
    def extract_features(self, query: str) -> np.array:
        """Extract features for routing"""
        # Simple keyword-based features (can use embeddings)
        features = [
            "search" in query.lower(),
            "calculate" in query.lower(),
            "code" in query.lower(),
            "database" in query.lower(),
            len(query),
            query.count("?"),
        ]
        return np.array(features)
```

**For production:** Use this simple router instead of full RL training. It's faster, cheaper, and easier to maintain.

---

<a id="pattern-4"></a>
## **Pattern #4 — Self-Consistency Sampling**

You know how when you're not sure about something, you ask multiple people and go with what most of them say? That's this pattern.

Instead of picking a single chain of thought, agents generate many and vote.

$$
y = \text{mode} \left(\{ \pi_\theta(x)^{(i)} \}_{i=1..k} \right)
$$

Translation: "Ask 10 experts instead of 1, then pick the most common answer."

### **Why This Works:**

### **Implementation:**

This is an efficient form of Monte Carlo Tree Search (MCTS) applied to the thought space:

```python
def self_consistency_sampling(prompt: str, n_samples: int = 10):
    """Generate multiple reasoning paths and vote"""
    samples = []
    
    for _ in range(n_samples):
        # Generate a chain of thought
        response = llm.generate(
            prompt,
            temperature=0.7,  # Higher temp = more diversity
            max_tokens=500
        )
        samples.append(extract_answer(response))
    
    # Return the most common answer
    from collections import Counter
    return Counter(samples).most_common(1)[0][0]

# Example: Math problem
question = "Solve: 2x + 5 = 15"
answer = self_consistency_sampling(question, n_samples=10)
# If 8/10 samples say x=5, that's the answer
```

Used in:

* Math reasoning

* Code generation

* Planning

* Safety filtering

### **Citation:**

*Wang et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)*

---

<a id="pattern-5"></a>
## **Pattern #5 — Graph-of-Thoughts (GoT)**

You know how when you're brainstorming, you don't think in a straight line? You have one idea, it branches into three, two of those merge back together, you cross out the bad one, and keep going.

That's Graph-of-Thoughts. Instead of linear reasoning (A → B → C), thoughts form a **graph**:

* **Parallel branches** — explore multiple ideas at once
* **Merging** — combine good ideas from different paths
* **Pruning** — cut off dead ends
* **Revising** — go back and fix earlier thoughts

### **The Analogy:**

Brainstorming on a whiteboard with arrows and clusters. Like a mind map, but the AI actually uses it to solve problems.

### **Graph-of-Thoughts Structure:**

```mermaid
graph TD
    Root[Initial Thought] --> Branch1[Thought Branch 1]
    Root --> Branch2[Thought Branch 2]
    Root --> Branch3[Thought Branch 3]
    Branch1 --> Merge1[Merge Point]
    Branch2 --> Merge1
    Branch3 --> Prune1[Pruned Branch]
    Merge1 --> Final[Final Thought]
    
    style Root fill:#e1f5ff
    style Merge1 fill:#fff4e1
    style Final fill:#e8f5e9
    style Prune1 fill:#ffebee,stroke-dasharray: 5 5
```

### **Implementation:**

GoT allows parallel exploration and merging of reasoning paths:

```python
from typing import List, Set

class ThoughtNode:
    def __init__(self, content: str, score: float):
        self.content = content
        self.score = score
        self.children: List[ThoughtNode] = []
        self.parents: List[ThoughtNode] = []

def graph_of_thoughts(initial_prompt: str, max_depth: int = 3):
    """Build a graph of reasoning paths"""
    root = ThoughtNode(initial_prompt, 0.0)
    frontier = [root]
    
    for depth in range(max_depth):
        # Expand all nodes in parallel
        new_nodes = []
        for node in frontier:
            # Generate multiple child thoughts
            children = generate_thoughts(node.content, n=3)
            for child_content, score in children:
                child = ThoughtNode(child_content, score)
                child.parents.append(node)
                node.children.append(child)
                new_nodes.append(child)
        
        # Merge similar thoughts
        merged = merge_similar_thoughts(new_nodes)
        # Prune low-scoring branches
        frontier = [n for n in merged if n.score > threshold]
    
    # Extract best path
    return extract_best_path(root)
```

Modern agent frameworks now implement explicit DAG-based reasoning.

### **Citation:**

*Besta et al. (2023). "Graph of Thoughts: Solving Elaborate Problems with Large Language Models." [arXiv:2308.09687](https://arxiv.org/abs/2308.09687)*

---

<a id="pattern-6"></a>
## **Pattern #6 — Search-Augmented Agents (When Thoughts Become a Tree)**

Instead of single ReAct trajectories or self-consistency voting, modern agents use **systematic search** over the space of possible reasoning paths.

### **Analogy:**

Like a chess engine exploring possible moves, but for reasoning and planning.

### **MCTS Search Tree:**

```mermaid
graph TD
    Root[Root State] --> N1[Node 1]
    Root --> N2[Node 2]
    Root --> N3[Node 3]
    N1 --> N4[Node 4]
    N1 --> N5[Node 5]
    N2 --> N6[Node 6]
    N3 --> N7[Node 7]
    N4 --> Best[Best Path]
    N5 --> Best
    
    style Root fill:#e1f5ff
    style Best fill:#e8f5e9
```

### **Formal Structure:**

Each search node represents:

$$
\text{node} = (s, a, r, s', \text{thought})
$$

Where:
* $s$ = current state
* $a$ = proposed action
* $r$ = predicted reward
* $s'$ = predicted next state
* $\text{thought}$ = LLM-generated reasoning

### **Implementation:**

LLM-MCTS (Monte-Carlo Tree Search with LLM guidance):

```python
from collections import defaultdict
import numpy as np

class SearchNode:
    def __init__(self, state, thought, action=None):
        self.state = state
        self.thought = thought
        self.action = action
        self.visits = 0
        self.value = 0.0
        self.children = []
        self.parent = None

def llm_mcts(root_state, goal, max_iterations=100):
    """Monte-Carlo Tree Search guided by LLM"""
    root = SearchNode(root_state, llm.generate_thought(root_state, goal))
    
    for _ in range(max_iterations):
        # Selection: traverse to leaf
        node = root
        while node.children:
            node = select_best_child(node)  # UCB1 or similar
        
        # Expansion: LLM proposes actions
        if not is_terminal(node.state):
            actions = llm.propose_actions(node.state, goal)
            for action in actions:
                next_state = world_model.predict(node.state, action)
                child = SearchNode(next_state, 
                    llm.generate_thought(next_state, goal), action)
                child.parent = node
                node.children.append(child)
        
        # Simulation: roll out to terminal
        value = simulate_rollout(node, goal)
        
        # Backpropagation: update values
        backpropagate(node, value)
    
    # Return best action
    return max(root.children, key=lambda n: n.value / n.visits).action

def world_model_predict(state, action):
    """Learned or LLM-based world model"""
    # Can be a neural network, physics simulator, or LLM
    return model.predict(state, action)

def simulate_rollout(node, goal, max_depth=10):
    """Roll out trajectory and evaluate"""
    current = node
    for depth in range(max_depth):
        if is_terminal(current.state):
            return evaluate(current.state, goal)
        action = llm.sample_action(current.state, goal)
        current = world_model_predict(current.state, action)
    return evaluate(current.state, goal)
```

### **Key Advantages:**

* **Systematic exploration** vs. single-shot reasoning
* **World model integration** for accurate predictions
* **Optimal path selection** via search
* **Handles long-horizon planning** better than ReAct alone

### **Used In:**

* Robotics (manipulation, navigation)
* Theorem proving
* Autonomous coding (multi-file refactoring)
* Web navigation (complex multi-step tasks)
* Game playing (Chess, Go, strategy games)

### **Citation:**

*Recent work combining LLMs with MCTS and learned world models (2024-2025)*

---

<a id="references"></a>
## **References**

**Shinn, K., et al. (2023).** Reflexion: An Autonomous Agent with Dynamic Memory and Self-Evaluation.  
[arXiv:2303.11366](https://arxiv.org/abs/2303.11366)

**Wang, G., et al. (2023).** Plan-and-Solve: Towards a Decision-Oriented Approach to LLM-Based Task Execution.  
[arXiv:2305.10174](https://arxiv.org/abs/2305.10174)

**Yao, S., et al. (2023).** ReAct: Synergizing Reasoning and Acting in Language Models.  
[arXiv:2210.03629](https://arxiv.org/abs/2210.03629)

**Yao, S., et al. (2024).** Tool Use: Large Language Models as Agents That Use External Tools.  
[arXiv:2402.13962](https://arxiv.org/abs/2402.13962)

**Ramaseri-Chandra, A. N. (2023).** Generative AI-based Non-person Character (NPC) For Navigating Virtual Worlds.  
*Cyber Awareness and Research Symposium 2023, University of North Dakota.*  
[ResearchGate](https://www.researchgate.net/publication/376030547_Generative_AI-based_Non-person_Character_NPC_For_Navigating_Generative_AI-based_Non-person_Character_NPC_For_Navigating_Virtual_Worlds_Virtual_Worlds)

**Wu, Z., et al. (2025).** UniVA: Universal Video Agent towards Open-Source Next-Generation Video Generalist.  
*Introduces a "Plan-and-Act" dual-agent architecture for automated and self-reflective video creation.*  
[ResearchGate](https://www.researchgate.net/publication/397521474_UniVA_Universal_Video_Agent_towards_Open-Source_Next-Generation_Video_Generalist)

**Sun, S., et al. (2025).** An LLM-LVLM Driven Agent for Iterative and Fine-Grained Image Editing.  
*LLM-LVLM driven agent for iterative and fine-grained image editing with planning and execution.*  
[arXiv:2508.17435](https://arxiv.org/html/2508.17435v1)

**Wang, J., et al. (2025).** FilmAgent: A Multi-Agent Framework for End-to-End Film Production.  
*Multi-agent framework designed for end-to-end film production, from script to final output.*  
[arXiv](https://arxiv.org/html/2501.12909v1)

**Additional References on Multi-Agent Systems and Reflection:**

- **Anim-Director:** A Large Multimodal Model Powered Agent for Controllable Animation Video Generation.  
  [ResearchGate](https://www.researchgate.net/publication/386398237_Anim-Director_A_Large_Multimodal_Model_Powered_Agent_for_Controllable_Animation_Video_Generation)

- **Reflection in AI Agents:** Research on feedback loops, memory systems, and explicit self-critique in agent systems.  
  See: [Akira AI Blog](https://www.akira.ai/blog/reflection-agent-prompting), [DeepLearning.AI](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/)

- **Commercial Tools:** Adobe Creative Cloud (Sensei), RunwayML, Descript, Unreal Engine 5.4 (Modular Control Rig) demonstrate AI-driven automation in professional creative pipelines.

**Electronic Design Automation (EDA) and Reinforcement Learning:**

- **Machine Learning for Electronic Design Automation: A Survey**  
  *Comprehensive overview of ML techniques, including RL, across the EDA design flow.*  
  [ResearchGate](https://www.researchgate.net/publication/349106567_Machine_Learning_for_Electronic_Design_Automation_A_Survey)

- **Application of Machine Learning in FPGA EDA Tool Development**  
  *Research on ML applications in FPGA design automation.*  
  [ResearchGate](https://www.researchgate.net/publication/374500231_Application_of_Machine_Learning_in_FPGA_EDA_Tool_Development)

- **Google TPU Design:** Google's use of deep reinforcement learning for optimizing TPU physical layouts.  
  See: [Wikipedia: Tensor Processing Unit](https://en.wikipedia.org/wiki/Tensor_Processing_Unit)

- **Synopsys DSO.ai:** Commercial RL-based design-space optimization tool for chip implementation.  
  See: [Wikipedia: Synopsys](https://en.wikipedia.org/wiki/Synopsys)

- **General EDA References:**  
  - [ScienceDirect: Electronic Design Automation](https://www.sciencedirect.com/topics/computer-science/electronic-design-automation)  
  - [ACM Digital Library: ML in EDA](https://dl.acm.org/doi/10.1145/3736167)  
  - [IEEE Xplore: RL in EDA](https://ieeexplore.ieee.org/iel7/6287639/10005208/10272331.pdf)

**Autonomous Scientific Discovery and AI Agents in Chemistry:**

- **AutoLabs: A Self-Correcting, Multi-Agent Framework for Autonomous Chemical Experimentation**  
  *Self-correcting, multi-agent architecture that translates natural-language instructions into executable protocols for high-throughput liquid handlers, achieving near-expert procedural accuracy.*  
  [arXiv:2509.25651](https://arxiv.org/abs/2509.25651)

- **Autonomous LLM Agent for Chemical Literature Data Mining**  
  *AI agent that performs high-fidelity extraction from extensive chemical literature, automating data collection and analysis with high accuracy, recall, and F1 scores.*  
  [arXiv:2402.12993](https://arxiv.org/abs/2402.12993)

- **Chemist-X: AI Agent for Automated Reaction Condition Recommendations**  
  *AI agent that automates reaction condition recommendations using retrieval-augmented generation, interrogating molecular databases and distilling data from latest literature.*  
  [arXiv:2311.10776](https://arxiv.org/abs/2311.10776)

- **dZiner: AI Agent for Rational Inverse Design of Materials**  
  *LLM-powered agent that facilitates rational inverse design of materials by leveraging domain-specific insights from scientific literature, proposing new materials with enhanced properties.*  
  [arXiv:2410.03963](https://arxiv.org/abs/2410.03963)

- **A Review of Large Language Models and Autonomous Agents in Chemistry**  
  *Comprehensive review covering recent history, current capabilities, and design of LLMs and autonomous agents in chemistry, addressing challenges and future directions.*  
  [Royal Society of Chemistry](https://pubs.rsc.org/doi/d4sc03921a)

- **Multi-agent-driven robotic AI chemist**  
  *A multi-agent-driven robotic AI chemist enabling autonomous chemical research on demand, operating through hierarchical multi-agent framework for complex, multi-step experiments.*  
  [ChemRxiv](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66a8c11bc9c6a5c07a7a59c0/original/a-multi-agent-driven-robotic-ai-chemist-enabling-autonomous-chemical-research-on-demand.pdf)

- **Argonne National Laboratory - Self-Driving Chemist Platform**  
  *Self-driving chemist platform that autonomously executes chemical reactions to improve processes and create new materials, with minimal human intervention.*  
  [Argonne National Laboratory](https://www.anl.gov/autonomous-discovery/selfdriving-chemist)

- **Empowering biomedical discovery with AI agents**  
  *Perspective on AI agents that formulate biomedical hypotheses, critically evaluate them, and characterize uncertainty in biomedical research.*  
  [ScienceDirect](https://www.sciencedirect.com/)

---

<a id="domain-applications"></a>
## The Multiverse of Agency: Domain-Specific Applications

The agentic framework is powerful because it applies across various fields requiring multi-stage complexity and verifiable results.

<a id="gaming"></a>
### 🎮 Gaming and Creative Content: Emergent Worlds

Agentic AI is moving game and content creation from scripted narratives to **Emergent Worlds**.

**Self-Evolving Non-Player Characters (NPCs):** Agent-based NPCs maintain **long-term episodic memory** (e.g., remembering a player's past deeds), **dynamic motivations**, and the ability to **plan social interactions**, creating believable character relationships and truly *dynamic storytelling* where the plot is co-authored in real-time by player and agent interactions (Ramaseri-Chandra, 2023).

**Multi-Step Asset Editing:** In design pipelines (like film, animation, and game asset creation), agents handle complex, multi-modal edits. For example, an agent can be tasked to "change the lighting on the main character in all 50 cutscenes to match the new mood board." The agent executes this by iteratively calling specialized tools (vision models, 3D rendering APIs, color grading functions), with a **Self-Evaluation** agent ensuring visual fidelity and style consistency across all assets. Recent research demonstrates this through multi-agent frameworks like UniVA (Wu et al., 2025) for video creation, RefineEdit-Agent (Sun et al., 2025) for iterative image editing, and FilmAgent (Wang et al., 2025) for end-to-end film production. Commercial tools like RunwayML and Descript are incorporating AI-assisted automation into professional pipelines.

<a id="scientific-discovery"></a>
### 🔬 Scientific Discovery: From Hypothesis to Lab Result

In the scientific arena, agents act as tireless, intelligent researchers, accelerating the discovery process by orders of magnitude.

**Circuit Design and Electronic Design Automation (EDA):** Agents are tasked with generating novel **Electronic Circuits** (e.g., optimizing a filter or a low-power digital block) based on high-level constraints (power, area, speed). They utilize EDA tools as their "tools," running simulations, evaluating physical layouts, and iteratively refining the design using **Reinforcement Learning** signals derived from the simulation results. This approach is formulated as a Markov Decision Process (MDP), where the agent interacts with the design environment and receives rewards based on metrics like wirelength, congestion, and power consumption.

Notable applications include Google's use of deep reinforcement learning to optimize the physical layouts of Tensor Processing Units (TPUs), and commercial products like Synopsys DSO.ai, which applies RL to automate chip implementation processes. Research in this area spans major EDA conferences (DAC, ICCAD) and demonstrates RL's effectiveness in tasks like floorplanning, placement, and routing optimization.

**Compound Search and Materials Discovery:** The agent acts as an autonomous chemist, navigating the vast and noisy landscape of chemical space. It performs **literature review** (via Retrieval-Augmented Generation/RAG), formulates a **hypothesis** for a new material (e.g., a high-temperature superconductor or a new drug candidate), designs the optimal **synthesis procedure** (a multi-step chemical protocol), and, in advanced setups, even controls robotic labs to execute the experiment—closing the loop from thought to physical action. Recent research demonstrates this through systems like AutoLabs (arXiv:2509.25651), which uses a self-correcting, multi-agent architecture to translate natural-language instructions into executable protocols for high-throughput liquid handlers. Multi-agent robotic AI chemists enable autonomous chemical research on demand, combining LLM-driven planning with robotic synthesis and analysis. AI agents can autonomously mine chemical literature (arXiv:2402.12993), automate reaction condition recommendations (Chemist-X, arXiv:2311.10776), and perform rational inverse design of materials (dZiner, arXiv:2410.03963).

<a id="product-design"></a>
### 📐 Product and Engineering Design: The Iterative Creator

Any domain centered on user design benefits from agents that can critique and refine proposals.

**Design Refinement and Optimization:** An agent can take a user-designed 3D model, run it through a structural simulator to check for stress points, and automatically suggest or implement design modifications to improve strength or reduce material usage, effectively performing **autonomous engineering optimization**.

**Code Generation and Debugging:** Agent teams (Planner, Coder, Reviewer) can maintain and evolve massive codebases. A **Coder Agent** writes the code, and a **Reviewer Agent** runs unit tests and static analysis. Upon failure, the Reviewer provides the full traceback as an "Observation," forcing the Coder Agent to **self-debug** and iterate without human intervention.

---

<a id="series-roadmap"></a>
## What's Next: Series Roadmap

This series provides a systematic exploration of the architectural patterns that enable autonomous, reliable agentic systems. The content is organized into five parts:

### **Part 2: Production Patterns**
Focuses on patterns essential for real-world deployment:
- **Memory Management** — Episodic, semantic, and working memory systems
- **Memory Compression** — Active memory rewriting and relevance filtering (the context window budget)
- **Supervisor/Orchestrator** — The #1 production pattern for cost efficiency
- **Parallel Tool Execution** — Fan-out patterns for latency reduction
- **Hidden Reasoning** — Test-time compute scaling (o1-style)

### **Part 3: Specialized Patterns**
Advanced patterns for domain-specific applications:
- **Embodied Agent Loops** — Vision-language-action integration
- **3D Grounded Agents** — Scene graph reasoning for robotics
- **Imagination Loops** — World model simulation and planning
- **Multi-Agent Societies** — Specialized agent teams and protocols
- **Compensatory Reflexes** — Error recovery and automatic correction
- **Introspective Agents** — Self-debugging and constraint verification

### **Part 4: Failure Modes & Safety**
Engineering reality: how agents fail and how to prevent it:
- **Common Failure Modes** — Tool overuse, contextual amnesia, goal drift, the "doom loop"
- **Verifiable Agent Pipelines** — Safety-aware planning and uncertainty quantification
- **Failure Taxonomy** — Systematic classification of agent failures
- **Eval-Driven Development** — Building agent unit tests and measuring failure modes
- **Mitigation Strategies** — Production-ready safeguards

### **Part 5: Production Guide**
The 2025 practitioner's handbook:
- **The Economics of Agency** — Latency vs. cost trade-offs, when to use agents vs. chains
- **Human Handoff Patterns** — Gracefully returning control when confidence is low
- **Cost Cheat Sheet** — Real-world cost and reliability numbers
- **Case Study** — Complete agent workflow for a research task
- **Production Checklist** — What actually ships in 2025
- **State of the Field** — Where research is converging

---

<div class="series-nav" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 12px; margin-top: 3rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Continue Reading</div>
  <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;">
    <a href="/posts/agentic-ai-design-patterns-part-2" style="background: rgba(255,255,255,0.25); padding: 0.75rem 1.5rem; border-radius: 6px; text-decoration: none; color: white; font-weight: 600; border: 2px solid rgba(255,255,255,0.5);">→ Next: Part 2: Production Patterns</a>
  </div>
  <div style="margin-top: 0.75rem; font-size: 0.875rem; opacity: 0.8;">Learn about memory management, supervisor patterns, parallel execution, and hidden reasoning</div>
</div>

