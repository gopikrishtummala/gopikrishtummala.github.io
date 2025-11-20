---
author: Gopi Krishna Tummala
pubDatetime: 2025-11-12T00:00:00Z
modDatetime: 2025-11-12T00:00:00Z
title: Reinforcement Learning — From Intuition to Algorithms
slug: reinforcement-learning-intuition-to-algorithms
featured: true
draft: false
tags:
  - reinforcement-learning
  - deep-learning
  - optimization
  - math
description: A narrative-first walkthrough of reinforcement learning, starting with everyday intuition and ending with the math behind Q-learning and DQN.
track: Fundamentals
difficulty: Advanced
interview_relevance:
  - Theory
estimated_read_time: 35
---

# Reinforcement Learning — From Intuition to Algorithms

*How curiosity, reward, and a little calculus teach machines to make decisions.*

---

## 1. Learning by Reward (Teaching a Dog)

Picture teaching a dog to fetch. You throw the ball (environment). The dog (agent) explores states (spots the ball, smells the grass), takes actions (run left, grab, return). When the dog finally brings the ball back, you give a treat (reward).

Over time the dog learns that earlier actions—sprinting toward the ball or bending to pick it up—lead to the treat that arrives later. That delayed credit assignment is the central challenge reinforcement learning solves: decide what to do now when the payoff comes much later.

> Want to roll up your sleeves immediately? [OpenAI’s Spinning Up](https://spinningup.openai.com/en/latest/) is a hands-on primer with code.

---

## 2. The Sandbox: Markov Decision Processes (MDPs)

Picture a video game where you walk through rooms, pick doors, bump into treasure or traps, and your score ticks up or down. That game world is the sandbox where an RL agent learns. We bundle that sandbox into a **Markov Decision Process (MDP)**:

$$
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

- **States ($\mathcal{S}$)** — the rooms or situations you might be in; what the agent can observe.
- **Actions ($\mathcal{A}$)** — the doors or moves you can take from that room.
- **Transition dynamics ($P(s' \mid s, a)$)** — if you’re in room $s$ and push door $a$, what are the chances you arrive in room $s'$?
- **Reward ($R(s, a)$)** — the immediate score for that choice (maybe +10 for treasure, −5 for a trap).
- **Discount ($\gamma \in [0,1]$)** — how much future points matter compared with immediate ones. If $\gamma$ is close to 1 you plan long-term; near 0 you focus on instant payoff.

The **Markov property** is the fancy way of saying “your current room summarises all the useful history.” You don’t need to remember the entire path you took. The agent wants to learn a **policy** $\pi(a \mid s)$ — a rulebook stating how likely it is to pick each action in each state — so that total reward is maximised.

---

## 3. Value Functions and Bellman Recursion

Once the agent is acting, you naturally want to grade how good different rooms or moves are.

- “If I start in $s$ and follow policy $\pi$, how many points do I expect?” — the **state value**:
  $$
  V^\pi(s) = \mathbb{E}_\pi\Bigg[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \,\Bigg|\, S_0 = s\Bigg]
  $$
- “If I start in $s$, first take action $a$, then follow $\pi$, how many points?” — the **action value**:
  $$
  Q^\pi(s, a) = \mathbb{E}_\pi\Bigg[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \,\Bigg|\, S_0 = s, A_0 = a\Bigg]
  $$

These values obey the **Bellman recursion**—value now equals immediate reward plus the discounted value of where you land next.

- Under policy $\pi$:
  $$
  V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a) \big[r + \gamma V^\pi(s')\big]
  $$
- For the optimal policy:
  $$
  V^*(s) = \max_{a} \sum_{s', r} P(s', r \mid s, a)\big[r + \gamma V^*(s')\big]
  $$

These are **fixed points**: plug in the true $V^\pi$ or $V^*$ and the equations hold exactly. Much of RL is about finding or approximating value functions that satisfy them.

---

## 4. Q-Learning — Learning Action Quality Without a Model

Suppose you don’t know the map. You push a door, see what happens, and jot down whether it was good or bad. **Q-learning** turns that trial-and-error into a recipe for finding the best action in each state:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big(R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\Big)
$$

- $(a')$ just means “a possible next action in the next state.” In this update you look ahead one step, consider every action you could take in $S_{t+1}$, pick the one with the highest $Q$, and use that as the target.

| Symbol | Meaning |
| --- | --- |
| $S_t$ | current state |
| $A_t$ | action you just took |
| $R_{t+1}$ | reward you just received |
| $S_{t+1}$ | resulting next state |
| $a'$ | candidate action you could take from $S_{t+1}$ |
| $\max_{a'} Q(S_{t+1}, a')$ | best predicted future value if you act optimally from $S_{t+1}$ |

Intuition: “I took $A_t$ in $S_t$, landed in $S_{t+1}$, and now I imagine choosing the best possible move $a'$ in that next state.” It’s like a dog thinking “I ran forward, now I’m near the ball — should I jump or keep running? I’ll assume the best choice.”

- $(S_t, A_t)$ — the state and action you just took.
- $R_{t+1}$ — the reward you observed.
- $S_{t+1}$ — the next state you landed in.
- The bracketed term is the **temporal-difference (TD) error**: a one-step lookahead target minus your current estimate $Q(S_t, A_t)$.
- If that error is positive you underestimated; if negative you overestimated. Either way you adjust $Q(S_t, A_t)$ by a fraction $\alpha$ of the error.

**Learning rate $\alpha$ — how fast you update**

$\alpha$ controls how much weight you give to new information versus what you already believed:

- $\alpha$ close to 1: you jump almost all the way to the new estimate. Fast but volatile—easy to overreact to noise.
- $\alpha$ small (e.g., 0.1): you move slowly, trusting your accumulated experience. Stable but slower to adapt.

Example: suppose $Q(S_t, A_t) = 5$, the target is $8$, and $\alpha = 0.2$:

$$
Q_{\text{new}} = 5 + 0.2 \times (8 - 5) = 5.6
$$

You moved 20% toward the new estimate, keeping 80% of your old value—like nudging your aim in a game of darts based on the latest throw.

| Symbol | Meaning | Typical range |
| --- | --- | --- |
| $\alpha$ | learning rate; how much you update on each experience | 0.01–0.5 |
| $\gamma$ | discount factor; how much you value future reward | 0.9–0.999 |

Together, $\alpha$ (speed of learning) and $\gamma$ (patience) shape how quickly your agent adapts and how far ahead it plans.

To learn well you must **explore**: ε-greedy means “with probability ε choose a random action; otherwise go with $\arg\max_a Q(s, a)$.” Shrink ε over time so you exploit more as the estimates mature.

Because Q-learning uses $\max_{a'} Q(S_{t+1}, a')$ even when you’re exploring, it’s **off-policy**: it learns what the optimal strategy would do. The sibling algorithm **SARSA** plugs in the action you actually take next, $Q(S_{t+1}, A_{t+1})$, so it evaluates your real behaviour—helpful when random exploration might be dangerous.

With sufficient exploration and diminishing learning rates, tabular Q-learning converges to the true $Q^*(s,a)$.

---

## 5. Deep Q-Networks (DQN) — Scaling Q-learning with Neural Nets

### When tables collapse

Lookup tables work only when the state space is tiny. In games like Pong or Breakout a single state is an entire screen image (millions of pixels). The number of possible screens is astronomical—you can’t enumerate every $Q(s,a)$. This is the **curse of dimensionality**.

### Function approximation to the rescue

DeepMind’s **Deep Q-Network (DQN)** tackled this by replacing the table with a neural network $Q(s, a; \theta)$. The network acts as a **function approximator**:

- Early convolutional layers learn features (“ball heading toward paddle”, “enemy nearby”).
- Later layers combine those features to estimate the value of states the agent has **never seen before**.
- The agent still picks actions via $A_t = \arg\max_a Q(S_t, a; \theta)$, but now $Q$ generalises rather than memorises.

Neural networks make the learning loop far more unstable, so DQN introduced two key stabilisers.

### 💡 Experience replay — the flashcard shuffle

Sequential experiences $(S_t, A_t, R_{t+1}, S_{t+1})$ are highly correlated. Training on the raw sequence is like studying only the last page of a textbook—you forget everything else. DQN stores each transition in a large **replay buffer** $D$ and trains on random mini-batches sampled from it.

*Analogy:* Replay is like shuffling flashcards. You don’t cram the most recent experience; you revisit older, rarer experiences again and again, breaking correlation and preventing catastrophic forgetting.

### 💡 Target network — stop chasing a moving goal

The Q-learning target
$$
Y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta)
$$
uses the same parameters $\theta$ you’re trying to update, so as you adjust $\theta$, the target moves immediately and optimisation becomes chaotic. DQN keeps a **lagged copy** $Q(s, a; \theta^-)$,
$$
Y_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-),
$$
and updates $\theta^- \leftarrow \theta$ only every few thousand steps.

*Analogy:* It’s like shooting at a target that only moves occasionally. With the target network frozen, the main network can steadily reduce the error before the goalpost shifts again.

### The loss you optimise

With experience replay drawing mini-batches from $D$ and the target network providing a stable $Y_t$, the loss is simply the squared TD error:
$$
L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D}\Big[\big(Y_t - Q(s, a; \theta)\big)^2\Big].
$$

By iterating this cycle—act, store, sample, compute stable targets, update $\theta$, slowly refresh $\theta^-$—DQN learned dozens of Atari games directly from pixels, showing that model-free RL could finally scale. ([Mnih et al., 2015](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf))

---

## 6. Minimal Q-Learning Pseudocode

```python
Initialize Q(s, a) arbitrarily
for episode in range(num_episodes):
    s = env.reset()
    done = False
    while not done:
        a = epsilon_greedy(Q, s)
        s_next, r, done, _ = env.step(a)
        td_target = r + gamma * max(Q[s_next, a_prime] for a_prime in actions)
        td_error  = td_target - Q[s, a]
        Q[s, a]  += alpha * td_error
        s = s_next
```

For DQN:

- replace `Q[s, a]` with a neural network,
- sample mini-batches from the replay buffer,
- compute the target with the separate network.

---

## 7. Worked Example — Gridworld

A 3×3 grid, start at top-left, goal at bottom-right (+1 reward), $\gamma = 0.9$. Initialize $Q=0$, run ε-greedy episodes. Watching Q-values evolve shows how value propagates backward from the goal via Bellman updates.

---

## 8. Variants and Modern Notes

- **SARSA**: on-policy update; tracks the value of the behaviour actually executed.
- **Improved DQN variants**: Double DQN (reduces overestimation), Dueling Networks (separate state vs. action salience), Prioritized Replay.
- **Policy Gradient / Actor-Critic**: directly parameterize the policy $\pi(a \mid s; \theta)$. The actor updates the policy; the critic estimates value to reduce variance. Essential for continuous action spaces.
- **Caveats**: deep RL is sample-inefficient and brittle. Hyperparameters, reward shaping, and exploration strategies can dramatically alter performance.

---

## 9. Deriving the Q-Learning Update

Bellman optimality for $Q^*$:
$$
Q^*(s, a) = \mathbb{E}\big[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \mid S_t = s, A_t = a\big].
$$

Define the Bellman optimality operator:
$$
(\mathcal{T}Q)(s, a) = \mathbb{E}\big[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') \mid s, a\big].
$$

Q-learning is stochastic approximation of this operator:
$$
Q \leftarrow Q + \alpha \big(\text{sampled target} - Q\big),
$$
which converges to $Q^*$ under standard assumptions.

---

## 10. Analogies that Stick

- **Value as a bank balance**: $V(s)$ is “how much you expect to earn” starting from state $s$.
- **TD error as surprise**: $\delta = r + \gamma \max_{a'} Q(s', a') - Q(s,a)$. Positive? You underestimated. Negative? You were overconfident.
- **Experience replay as flashcard rotation**: mix up your experiences; revisit older ones to prevent forgetting.
- **Bootstrapping**: today’s estimate is based on tomorrow’s estimate. It’s self-consistency: “my expected balance equals my immediate deposit plus tomorrow’s expectation.”

---

## 11. Practice Challenges

1. **Tabular Q-learning (Gridworld)**: implement and visualise the learned policy arrows.
2. **Compare SARSA vs. Q-learning**: test both in a windy gridworld; note the differences.
3. **Minimal DQN (CartPole)**: implement experience replay + target network. Observe stability across different replay buffer sizes.

Need starter notebooks or code? Ping me.

---

## 12. References

1. OpenAI — *Spinning Up in Deep RL*  
2. Sutton & Barto — *Reinforcement Learning: An Introduction*  
3. Watkins & Dayan — “Technical Note: Q-Learning”  
4. Mnih et al. — “Playing Atari with Deep Reinforcement Learning”  
5. David Silver — DeepMind/UCL RL Lecture Series

---

## 13. Summary

Reinforcement learning teaches agents to make decisions by trying actions and receiving rewards. Bellman equations define value functions as fixed points. Q-learning updates action values using a one-step lookahead (the temporal-difference error), and Deep Q-Networks scale that idea with neural networks, experience replay, and target networks. These tools power everything from game-playing agents to control systems.

