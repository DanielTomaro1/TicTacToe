"""Deep Q-Learning Class.

Implements a Deep Q-Learning agent for reinforcement learning.

"""

import random
from collections import deque

import torch
import torch.nn as nn

from src.agent import Agent


class DeepQLearning(Agent):
    """Deep Q-Learning Agent.

    Implements a Deep Q-Learner for environments with discrete action spaces.

    Attributes:
        size (int): Field size of the environment (e.g., tic-tac-toe board size).
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Number of samples in each training batch.
        epsilon (float): Exploration rate for epsilon-greedy action selection.
        epsilon_min (float): Minimum value for epsilon.
        decay_rate (float): Rate at which epsilon decays.
        gamma (float): Discount factor for future rewards.
        memory_size (int): Maximum size of the replay memory.
        memory (deque): Replay memory buffer for storing experiences.
        optimizer (torch.optim.Optimizer): Optimizer for training the policy network.
        criterion (nn.Module): Loss function for training.
    """

    def __init__(self, model: nn.Module, args) -> None:
        """
        Initializes the DeepQLearning agent.

        Args:
            model (nn.Module): Neural network representing the Q-function.
            args (Namespace): Parsed arguments containing configuration parameters.
        """
        super().__init__(model=model)

        self.size = args.field_size
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.decay_rate = args.decay_rate
        self.gamma = args.gamma
        self.memory_size = args.memory_size

        self.memory = deque(maxlen=self.memory_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    @torch.no_grad()
    def get_action(self, state: torch.Tensor) -> int:
        """
        Selects an action using epsilon-greedy strategy.

        Args:
            state (torch.Tensor): Current state observed by the agent.

        Returns:
            int: Action chosen by the agent.
        """
        if random.random() < self.epsilon:
            # Exploration: Choose random action
            return random.randint(0, self.size**2 - 1)
        else:
            # Exploitation: Choose action with the highest predicted Q-value
            return self.model.predict(state)

    def _memorize(self, events: dict) -> None:
        """
        Stores current events (experience) in replay memory.

        Args:
            events (dict): Dictionary containing states, actions, rewards, new states, and termination flags.
        """
        for state, action, reward, new_state, done in zip(*events.values()):
            self.memory.append([state, action, reward, new_state, done])

    def _epsilon_scheduler(self) -> None:
        """
        Decays the epsilon value to reduce exploration over time.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay_rate

    @torch.no_grad()
    def _create_training_set(self) -> tuple:
        """
        Prepares the training set by sampling experiences from the replay memory.

        Returns:
            tuple: States tensor and target Q-values tensor.
        """
        # Sample a mini-batch from the replay memory
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        # Normalize rewards in the sampled batch
        rewards = torch.tensor([experience[2] for experience in batch])
        normalized_rewards = self._normalize_rewards(rewards)

        for i, experience in enumerate(batch):
            experience[2] = normalized_rewards[i]

        # Prepare states and new states
        states = torch.vstack([experience[0] for experience in batch])
        new_states = torch.vstack([experience[3] for experience in batch])

        self.model.eval()
        q_targets = self.model(states).clone()
        q_next = self.model(new_states)
        self.model.train()

        for i, (_, action, reward, _, done) in enumerate(batch):
            if not done:
                q_targets[i, action] = reward + self.gamma * torch.max(q_next[i])
            else:
                q_targets[i, action] = reward

        return states, q_targets

    def train_on_batch(self, states: torch.Tensor, q_targets: torch.Tensor) -> float:
        """
        Performs a single optimization step on a batch of training data.

        Args:
            states (torch.Tensor): Batch of states.
            q_targets (torch.Tensor): Batch of target Q-values.

        Returns:
            float: Training loss for the batch.
        """
        self.optimizer.zero_grad()
        predicted_q_values = self.model(states)
        loss = self.criterion(predicted_q_values, q_targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def step(self, events: dict) -> None:
        """
        Runs a single optimization step and updates the agent.

        Args:
            events (dict): Dictionary containing experience (states, actions, rewards, new states, dones).
        """
        self._memorize(events)
        if len(self.memory) >= self.batch_size:
            states, q_targets = self._create_training_set()
            loss = self.train_on_batch(states, q_targets)
            self.stats["loss"] = loss

        self.stats["reward"] = sum(events["rewards"])
        self.stats["epsilon"] = self.epsilon
        self._epsilon_scheduler()
