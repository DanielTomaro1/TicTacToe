"""Agent Base Class.

Defines the abstract agent class used as the foundation for reinforcement learning agents.

"""

import torch


class Agent:
    """Abstract Agent Class.

    Attributes:
        model (torch.nn.Module): The policy network for the agent.
        stats (dict): Tracks key statistics like epsilon, loss, and reward for the agent.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Initializes the Agent class.

        Args:
            model (torch.nn.Module): The neural network model representing the agent's policy.
        """
        self.model = model
        self.stats = {"epsilon": None, "loss": None, "reward": None}

    def _normalize_rewards(self, rewards: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        Normalizes rewards to ensure numerical stability during training.

        If there are multiple rewards and the standard deviation is non-zero,
        the rewards are normalized using mean and standard deviation.

        Args:
            rewards (torch.Tensor): A tensor containing the rewards.
            eps (float, optional): A small value to prevent division by zero. Default is 1e-5.

        Returns:
            torch.Tensor: Normalized rewards tensor.
        """
        if rewards.numel() > 1:  # Check if there is more than one reward
            std = torch.std(rewards)
            if std > 0:
                mean = torch.mean(rewards)
                rewards = (rewards - mean) / (std + eps)
        return rewards

    @staticmethod
    def print_events(events: dict) -> None:
        """
        Prints events in a structured and readable format for debugging purposes.

        Args:
            events (dict): A dictionary holding states, actions, rewards, next states,
                           and termination signals.
        """
        for key, value in events.items():
            print(f"{key}:\n{'-' * len(key)}")
            for item in value:
                print(f"{item}\n")
