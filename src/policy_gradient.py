"""Policy Gradient Class.

Implements the Policy Gradient algorithm for reinforcement learning.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.agent import Agent


class PolicyGradient(Agent):
    """Policy Gradient Agent.

    Uses policy gradients to train an agent by learning a probability distribution
    over actions and sampling from this distribution.

    Attributes:
        size (int): Size of the Tic-Tac-Toe board (field size).
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for rewards.
        optimizer (torch.optim.Optimizer): Optimizer for the model's parameters.
        criterion (torch.nn.Module): Loss function for policy updates.
    """

    def __init__(self, model: nn.Module, args) -> None:
        """
        Initializes the Policy Gradient agent.

        Args:
            model (nn.Module): Neural network representing the policy.
            args: Argument object containing configuration parameters.
        """
        super().__init__(model=model)

        self.size = args.field_size
        self.learning_rate = args.learning_rate
        self.gamma = args.gamma

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    @torch.no_grad()
    def get_action(self, state: torch.Tensor) -> int:
        """
        Samples an action from the policy's probability distribution given a state.

        Args:
            state (torch.Tensor): Tensor representing the current state of the environment.

        Returns:
            int: Action sampled from the policy distribution.
        """
        self.model.eval()
        action_prob = self.model(state)  # Probability distribution over actions
        action = torch.multinomial(action_prob, num_samples=1).item()  # Sample an action
        self.model.train()
        return action

    def step(self, events: dict) -> None:
        """
        Runs a single optimization step to update the policy network.

        Args:
            events (dict): Dictionary containing:
                - states (list[torch.Tensor]): List of observed states.
                - actions (list[int]): List of actions taken.
                - rewards (list[float]): List of rewards received.
        """
        # Extract events
        states = events["states"]
        actions = events["actions"]
        rewards = events["rewards"]

        # Compute discounted rewards
        reward_sum = 0.0
        discounted_rewards = []
        for reward in reversed(rewards):
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.insert(0, reward_sum)  # Reverse the list

        # Normalize rewards
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = self._normalize_rewards(rewards=discounted_rewards)

        # Prepare inputs for training
        states = torch.vstack(states)  # Stack states into a single tensor
        target_actions = F.one_hot(torch.tensor(actions), num_classes=self.size**2).float()

        # Zero gradients and compute loss
        self.optimizer.zero_grad()
        output_actions = self.model(states)
        loss = self.criterion(output_actions, target_actions)
        loss = (discounted_rewards * loss).mean()  # Weighted loss using discounted rewards
        loss.backward()
        self.optimizer.step()

        # Update stats
        self.stats["loss"] = loss.item()
        self.stats["reward"] = sum(rewards)
