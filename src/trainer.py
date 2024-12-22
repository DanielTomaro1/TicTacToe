"""Training Method.

Defines the training process for agents using the selected reinforcement learning algorithm.

"""

import random
from torch.utils.tensorboard import SummaryWriter
from src.utils import save_checkpoint
from src.environment import Environment
from src.agent import Agent


def train(env: Environment, agent_a: Agent, agent_b: Agent, args) -> None:
    """
    Trains two agents in the specified environment using reinforcement learning.

    Args:
        env (Environment): The environment where agents interact and compete.
        agent_a (Agent): The first agent to be trained.
        agent_b (Agent): The second agent to be trained.
        args: Argument object containing training hyperparameters like number of episodes.
    """
    writer = SummaryWriter()  # For logging metrics to TensorBoard

    for episode in range(args.num_episodes):
        # Randomize which agent plays first to avoid bias
        if random.random() < 0.5:
            events_a, events_b = env.run_episode(agent_a, agent_b)
        else:
            events_b, events_a = env.run_episode(agent_b, agent_a)

        # Perform a single optimization step for each agent
        agent_a.step(events_a)
        agent_b.step(events_b)

        # Log statistics every 500 episodes
        if episode % 500 == 0:
            for key, value in agent_a.stats.items():
                if value is not None:
                    writer.add_scalar(f"agent_a/{key}", value, episode)

            for key, value in agent_b.stats.items():
                if value is not None:
                    writer.add_scalar(f"agent_b/{key}", value, episode)

    # Close the TensorBoard writer
    writer.close()

    # Save trained models for both agents
    save_checkpoint(model=agent_a.model, model_name="agent_a", args=args)
    save_checkpoint(model=agent_b.model, model_name="agent_b", args=args)
