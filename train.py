"""Train Agents to Play Tic-Tac-Toe.

This script trains agents using the selected reinforcement learning algorithm 
(policy gradient or deep Q-learning) to play Tic-Tac-Toe.

"""

from src.utils import print_args, set_random_seed
from src.argparser import argument_parser
from src.environment import TicTacToe
from src.model import Model
from src.policy_gradient import PolicyGradient
from src.deep_q_learning import DeepQLearning
from src.trainer import train


if __name__ == "__main__":
    # Parse command-line arguments
    args = argument_parser()
    print_args(args=args)

    # Set up Tic-Tac-Toe environment
    env = TicTacToe(size=args.field_size)

    # Set random seed for reproducibility
    set_random_seed(seed=args.random_seed)

    # Initialize models and agents based on the chosen algorithm
    if args.algorithm == "policy_gradient":
        model_a = Model(args)
        model_b = Model(args)
        agent_a = PolicyGradient(model=model_a, args=args)
        agent_b = PolicyGradient(model=model_b, args=args)

    elif args.algorithm == "deep_q_learning":
        model_a = Model(args)
        model_b = Model(args)
        agent_a = DeepQLearning(model=model_a, args=args)
        agent_b = DeepQLearning(model=model_b, args=args)

    else:
        raise NotImplementedError(f"Reinforcement algorithm '{args.algorithm}' is not implemented.")

    # Train agents in the environment
    train(env=env, agent_a=agent_a, agent_b=agent_b, args=args)
