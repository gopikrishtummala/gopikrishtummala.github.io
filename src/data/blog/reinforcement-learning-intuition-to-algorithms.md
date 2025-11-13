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

- $(S_t, A_t)$ — the state and action you just took.
- $R_{t+1}$ — the reward you observed.
- $S_{t+1}$ — the next state you landed in.
- The bracketed term is the **temporal-difference (TD) error**: your one-step lookahead target minus your current guess.

If the TD error is positive you underestimated the action; if negative you overestimated it. Either way you adjust $Q(S_t, A_t)$ by a fraction $\alpha$ of that error.

To learn well you must **explore**: ε-greedy means “with probability ε choose a random action; otherwise go with $\arg\max_a Q(s, a)$.” Shrink ε over time so you exploit more as the estimates mature.

Because Q-learning uses $\max_{a'} Q(S_{t+1}, a')$ even when you’re exploring, it’s **off-policy**: it learns what the optimal strategy would do. The sibling algorithm **SARSA** plugs in the action you actually take next, $Q(S_{t+1}, A_{t+1})$, so it evaluates your real behaviour—helpful when random exploration might be dangerous.

With sufficient exploration and diminishing learning rates, tabular Q-learning converges to the true $Q^*(s,a)$.

---

## 5. Deep Q-Networks (DQN) — Scaling with Neural Nets

Tables break down when states are high-dimensional (images, sensor readings). DeepMind’s **Deep Q-Network (DQN)** replaces the table with a neural network $Q(s, a; \theta)$.

Two engineering tricks made DQN stable:

1. **Experience replay** — stash transitions $(s, a, r, s')$ in a replay buffer and sample random mini-batches. This breaks correlation between successive experiences and lets you reuse data efficiently.
2. **Target network** — keep a lagged copy $Q(s, a; \theta^-)$ to form the target
   $$
   y = r + \gamma \max_{a'} Q(s', a'; \theta^-).
   $$
   Every so often copy $\theta \gets \theta^-$. Between copies the target stays steady, preventing the network from chasing a moving goal.

Optimise
$$
L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D}\Big[\big(y - Q(s, a; \theta)\big)^2\Big]
$$
over mini-batches drawn from the buffer $D$. With experience replay and a target network, DQN famously learned dozens of Atari games straight from pixels. ([Mnih et al., 2015](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf))

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

