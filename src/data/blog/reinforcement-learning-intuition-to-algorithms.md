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

Think of an adventure game: you wander rooms, choose doors, bump into treasure or traps, and collect points. That world is the sandbox where the agent learns. In RL we formalise that sandbox as a **Markov Decision Process (MDP)**:

$$
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

- $\mathcal{S}$ — the **states** (the rooms the agent can stand in; what it observes).
- $\mathcal{A}$ — the **actions** (the doors available in each room).
- $P(s' \mid s, a)$ — the **transition dynamics** (push door $a$ in room $s$, land in room $s'$ with some probability).
- $R(s, a)$ — the **reward** (the immediate points for that action: +10 for treasure, −5 for a trap).
- $\gamma \in [0, 1]$ — the **discount factor** (how much future points matter relative to immediate ones).

The **Markov property** just says “the current state contains all the relevant history.” You don’t need to remember the entire path you took; knowing where you are now is enough. The agent wants to learn a **policy** $\pi(a \mid s)$ — a rulebook describing which action to take in each state — that maximises total accumulated reward.

---

## 3. Value Functions and Bellman Recursion

Once the agent plays the game you naturally ask:

- “If I start in state $s$ and follow policy $\pi$, how many points do I expect?” → the **state value**.
- “If I start in $s$, take action $a$ first, then follow $\pi$, how many points?” → the **action value** or **Q-value**.

- **State value** under policy $\pi$:
  $$
  V^\pi(s) = \mathbb{E}_\pi\Bigg[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \,\Bigg|\, S_0 = s\Bigg]
  $$
  Average the discounted future rewards if you start in $s$ and behave according to $\pi$.

- **Action value (Q)**:
  $$
  Q^\pi(s, a) = \mathbb{E}_\pi\Bigg[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \,\Bigg|\, S_0 = s, A_0 = a\Bigg]
  $$
  Take action $a$ first, then follow $\pi$ thereafter.

These values obey the **Bellman recursion**. It says: the value today equals the expected reward now plus the discounted value of tomorrow.

- Bellman expectation equation (for a fixed policy):
  $$
  V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a) \big[r + \gamma V^\pi(s')\big]
  $$

- Bellman optimality equation (for the best possible behaviour):
  $$
  V^*(s) = \max_{a} \sum_{s', r} P(s', r \mid s, a)\big[r + \gamma V^*(s')\big]
  $$

They are **fixed-point equations**: plug in the true value functions and the equations hold exactly. Learning is about finding functions that satisfy them.

---

## 4. Q-Learning — Learning Action Quality Without a Model

**Q-learning** (Watkins) is what you use when you don’t know the map ahead of time. You just try actions and observe the outcome. The update rule is:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big(R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\Big)
$$

Break it down:

- $(S_t, A_t)$ — where you are and what you chose.
- $R_{t+1}$ — the reward you just saw.
- $S_{t+1}$ — the next state you landed in.
- The bracketed term is the **temporal-difference (TD) error**: a one-step lookahead ($R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$) minus your current guess $Q(S_t, A_t)$.

If the TD error is positive you underestimated, so you bump $Q$ up; if negative you overestimated, so you pull it down. The learning rate $\alpha$ controls how fast you adjust.

To learn well you must **explore**: ε-greedy means “with probability ε, pick a random action; otherwise pick $\arg\max_a Q(s,a)$.” Shrink ε over time so you exploit more as you learn.

Q-learning is **off-policy**: it learns about the optimal policy $Q^*$ even while you explore randomly. The on-policy sibling **SARSA** replaces $\max_{a'}$ with $Q(S_{t+1}, A_{t+1})$, the value of the action you actually take next. That makes SARSA evaluate your real behaviour, which is safer if random actions could cause trouble.

With sufficient exploration and diminishing learning rates, tabular Q-learning converges to the true $Q^*(s,a)$.

---

## 5. Deep Q-Networks (DQN) — Scaling with Neural Nets

High-dimensional states (like images) make tables infeasible. DeepMind’s **Deep Q-Network (DQN)** replaces the table with a neural network $Q(s, a; \theta)$ that approximates Q-values.

Two stabilisers make it work:

1. **Experience Replay** — store transitions $(s, a, r, s')$ in a buffer and sample random mini-batches. This breaks correlations between successive experiences and lets you reuse data.
2. **Target Network** — keep a lagged copy $Q(s, a; \theta^-)$ and produce the target
   $$
   y = r + \gamma \max_{a'} Q(s', a'; \theta^-).
   $$
   Periodically copy $\theta \gets \theta^-$; between copies $\theta^-$ stays fixed. That keeps the target from chasing a moving estimate.

The loss over a mini-batch sampled from replay buffer $D$ is:
$$
L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D}\Big[\big(y - Q(s, a; \theta)\big)^2\Big].
$$

With these two ideas DQN famously learned many Atari games from raw pixels. ([Mnih et al., 2015](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf))

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

