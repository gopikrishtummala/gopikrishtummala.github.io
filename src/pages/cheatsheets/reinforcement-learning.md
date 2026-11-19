---
layout: ../../layouts/CheatsheetLayout.astro
title: "Reinforcement Learning in 2 Minutes Each"
description: "Intuition cards for agents, rewards, MDPs, value functions, Q-learning, DQN, policy gradients, actor-critic, PPO, exploration, model-based and offline RL, and RL for language models."
---

Learning by trial, error and reward.

## The setup

An agent takes actions in an environment, gets rewards, and learns to maximize total future reward.

**Picture it:** training a dog with treats. No one shows the right answer; good behavior gets rewarded, sometimes much later.

**Remember:** the hard part is credit assignment: which earlier action earned the reward that came later?

**Go deeper:** [Reinforcement Learning](/posts/fundamentals/reinforcement-learning-intuition-to-algorithms).

## MDP (Markov decision process)

States, actions, rewards, and transitions, where the next state depends only on the current state and action.

**Remember:** the discount factor γ (e.g. 0.99) makes near rewards worth more than far ones and keeps sums finite.

## Policy

The agent's strategy: what action to take in each state.

**Remember:** deterministic (always do X) or stochastic (do X with 70% probability).

## Value functions

How good it is to be in a state, $V(s)$, or to take an action in a state, $Q(s,a)$, measured by expected future reward.

**Picture it:** a chess player's sense that a position is "winning," before any piece is captured.

**Remember:** if you know $Q$, act greedily: pick the action with the highest $Q$.

## Bellman equation

The value of a state = immediate reward + discounted value of where you land next.

**Remember:** it turns a long-term problem into a one-step recursion, the basis of most RL algorithms.

## Q-learning and DQN

Learn $Q(s,a)$ by nudging it toward "reward + best next $Q$."

**How it works:** DQN uses a neural network for $Q$, a replay buffer of past experience (breaks correlations), and a slowly updated target network (stabilizes learning). It learned Atari games from pixels.

**Remember:** value-based, off-policy, discrete actions.

## Policy gradients

Directly adjust the policy to make rewarded actions more likely.

**Picture it:** after each game, reinforce the moves of games you won, discourage those of games you lost.

**Remember:** works for continuous actions and stochastic policies; high variance, so subtract a baseline (average reward) to stabilize.

## Actor-critic

An actor (policy) chooses actions; a critic (value function) judges them.

**Remember:** the critic's estimate is the baseline that tames policy-gradient noise. SAC and TD3 are popular for continuous control.

## PPO

Policy gradient with a limit on how much the policy can change per update.

**Remember:** stable and simple enough to be the default for years, including in RLHF for language models.

## Exploration vs exploitation

Try new things to learn, or use what you know works?

**Remember:** ε-greedy (random action with small probability), entropy bonuses, optimism about unknown actions. Too little exploration gets stuck on mediocre strategies.

## Model-based RL

Learn a model of how the world responds, then plan or practice inside it.

**Remember:** far more sample-efficient (AlphaZero, MuZero, Dreamer); errors in the learned model can mislead the agent.

## Offline RL

Learn a policy only from logged data, without new interaction.

**Remember:** essential when trying things is dangerous (driving, healthcare); must avoid trusting actions the data never tried.

## Reward design

The reward defines the task, and agents exploit any gap.

**Watch out:** reward hacking: a boat-racing agent that spins in circles collecting points instead of finishing. Reward what you actually want.

## RL for language models

| Method | Signal |
|---|---|
| RLHF (PPO) | Reward model trained on human preferences |
| DPO | Preference pairs, no RL loop |
| GRPO | Checkable rewards; group average as baseline |

**Remember:** RL with verifiable rewards is how reasoning models learn to think longer and check their work.

**Go deeper:** [Post-Training](/posts/mlops/modern-post-training-peft-2026).
