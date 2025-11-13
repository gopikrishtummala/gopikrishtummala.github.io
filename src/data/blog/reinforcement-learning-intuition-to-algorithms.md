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

Reinforcement learning typically models problems as **Markov Decision Processes** (MDPs):

$$
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

- $\mathcal{S}$: state space (the observations)
- $\mathcal{A}$: action space (the choices)
- $P(s' \mid s, a)$: transition dynamics
- $R(s, a)$: expected immediate reward
- $\gamma \in [0, 1]$: discount factor (how much the future matters)

The **Markov property** says the current state summarises everything relevant from the past; given $S_t$, the future doesn’t depend on the earlier history. The agent’s goal is to find a **policy** $\pi(a \mid s)$ that maximises cumulative reward.

---

## 3. Value Functions and Bellman Recursion

To compare actions and states, we define value functions.

- **State value** under policy $\pi$:
  $$
  V^\pi(s) = \mathbb{E}_\pi\Big[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \,\Big|\, S_0 = s\Big]
  $$

- **Action value (Q)**:
  $$
  Q^\pi(s, a) = \mathbb{E}_\pi\Big[\sum_{t=0}^{\infty} \gamma^t R_{t+1} \,\Big|\, S_0 = s, A_0 = a\Big]
  $$

They satisfy **Bellman expectation equations**:

$$
V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a) \big[r + \gamma V^\pi(s')\big]
$$

The **Bellman optimality equation** replaces the expectation over actions with a max:

$$
V^*(s) = \max_{a} \sum_{s', r} P(s', r \mid s, a)\big[r + \gamma V^*(s')\big]
$$

These equations define fixed points. Solving reinforcement learning problems amounts to finding those fixed points.

---

## 4. Q-Learning — Learning Action Quality Without a Model

**Q-learning** (Watkins) is a model-free, **off-policy** algorithm: it learns the optimal action value $Q^*$ even if you explore with a different behaviour.

Given experience $(S_t, A_t, R_{t+1}, S_{t+1})$, the update is:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big(R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\Big)
$$

Here $\alpha$ is the learning rate. The term in brackets is the **temporal-difference (TD) error**:

- Target: $R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')$
- Estimate: $Q(S_t, A_t)$

If the TD error is positive you were underestimating, so bump Q up; if negative you pull it down.

**Exploration:** ε-greedy (with probability ε choose a random action; otherwise choose the action with highest Q). Anneal ε so exploration gives way to exploitation.

**SARSA** (the on-policy sibling) uses the action actually taken in $S_{t+1}$: $r + \gamma Q(S_{t+1}, A_{t+1})$. It learns the value of the behaviour you follow, useful when exploratory actions are risky.

Under mild conditions (sufficient exploration, diminishing step sizes) tabular Q-learning converges to $Q^*$. See Watkins & Dayan for the proof.

---

## 5. Deep Q-Networks (DQN) — Scaling with Neural Nets

When states are high-dimensional (e.g., images), you replace the Q-table with a neural **function approximator** $Q(s, a; \theta)$.

DeepMind’s DQN (Mnih et al.) added two stabilising ingredients:

1. **Experience Replay**: store transitions $(s, a, r, s')$ and sample mini-batches randomly. This decorrelates updates, acting like shuffled flashcards.
2. **Target Network**: keep a lagged copy $Q(s, a; \theta^-)$ to compute targets $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$. Periodically copy $\theta \to \theta^-$. Otherwise the target moves with every update and optimization becomes unstable.

**Loss for a mini-batch**:
$$
L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D}\Big[\big(y - Q(s, a; \theta)\big)^2\Big]
$$

With these tricks, DQN learned many Atari games end-to-end from pixels. ([Mnih et al., 2015](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf))

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

