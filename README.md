
---

# TicTacToe  

A reinforcement learning environment designed to train agents to compete in [tic-tac-toe](https://en.wikipedia.org/wiki/Tic-tac-toe). Due to its simplicity, this repository serves both educational purposes and as a starting point for generalizing to other games like m,n,k-games, chess, or Go.

---

## Table of Contents  
1. [Getting Started](#getting-started)  
2. [Introduction](#introduction)  
3. [Reinforcement Learning Overview](#reinforcement-learning-overview)  
   - [Tic-tac-toe Environment](#tic-tac-toe-environment)  
   - [Policy Network](#policy-network)  
   - [Episodic Learning](#episodic-learning)  
   - [Self-play](#self-play)  
4. [Learning Algorithms](#learning-algorithms)  
   - [Policy Gradients](#policy-gradients)  
   - [Deep Q-Learning](#deep-q-learning)  
5. [TODO](#todo)  

## Getting Started  

### Train Agents  

Run a training session using the specified learning algorithm:  

```bash
python train.py -a "policy_gradient"
python train.py -a "deep_q_learning"
```

### Monitor Training  

Track key metrics during training using TensorBoard:  

```bash
tensorboard --logdir runs/
```

### Play Against the Trained Agent  

After training, challenge the agent:  

```bash
python play.py -a deep_q_learning -mn agent_a
```

---

## Introduction  

Tic-tac-toe is a classic instance of the [m,n,k-game](https://en.wikipedia.org/wiki/M,n,k-game), a two-player, turn-based game with perfect information. The standard version corresponds to **m=n=k=3**.  

### How It Works  
- Agents take alternating turns on an **m × n** board until one player achieves **k** marks in a row.  
- This implementation supports arbitrary positive integers for m, n, and k.  

---

## Reinforcement Learning Overview  

Reinforcement learning trains agents to interact with an environment and learn optimal strategies through rewards. For tic-tac-toe:  
- **State**: Current board configuration (positions of `×` and `◦`).  
- **Action**: Legal moves available to the agent.  
- **Reward**:  
   - **+1**: Win  
   - **-1**: Loss or illegal move  
   - **0**: Draw or ongoing state  

---

### Tic-tac-toe Environment  

States are represented numerically:  
- `1` for `×`  
- `-1` for `◦` (opponent)  
- `0` for empty positions  

For example:  

```
Current State: 
◦ |   | ◦
---------
  | × |  
---------
◦ | × | ×

Numerical Representation:
-1 |  0 | -1
------------
 0 |  1 |  0
------------
-1 |  1 |  1
```

Actions are chosen based on the agent’s policy, modeled by a neural network.  

---

### Policy Network  

The policy network maps states to action probabilities using a softmax output layer. Example:  

\[
\text{Policy}(\textbf{state}; \boldsymbol{\theta}) =
\begin{pmatrix}
0.0 & 0.8 & 0.0 \\
0.1 & 0.0 & 0.1 \\
0.0 & 0.0 & 0.0
\end{pmatrix}
\]

---

### Episodic Learning  

- Each **episode** corresponds to a single game of tic-tac-toe.  
- Rewards are collected, and the agent’s policy is updated after the episode ends.  

---

### Self-play  

Agents can be trained using:  
1. **Self-competition**: Two agents train by competing against each other.  
2. **Progressive Training**: An agent improves by repeatedly beating weaker versions of itself.  

To generalize well, randomizing opponent strategies is recommended.  

---

## Learning Algorithms  

### Policy Gradients  

Policy Gradients optimize the agent’s policy to maximize future rewards using this equation:  

\[
\nabla_{\theta} \mathbb{E}[r(a)] = \mathbb{E}[\nabla_{\theta} \log (p(a|\theta)) \cdot r(a)]
\]

The algorithm unrolls the episode and computes gradients based on rewards for each step.  

---

### Deep Q-Learning  

Deep Q-learning trains a Q-function to predict action values. Key concepts include:  
- **Bellman Equation**:  

\[
Q(s,a) = r(s,a) + \gamma \max_{a'} Q(s',a')
\]

- **Exploration vs. Exploitation**: Uses epsilon-greedy sampling to balance learning and optimal actions.  

---

## TODO  

- Generalize implementation for m,n,k-games.  
- Add adaptive epsilon decay for Deep Q-Learning.  
- Implement Boltzmann exploration and epsilon-greedy sampling.  

---

