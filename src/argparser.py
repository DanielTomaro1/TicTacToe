"""Argument Parser.

Defines environment, learning parameters, and model configurations for training the Tic-tac-toe agent.

"""

import argparse


def argument_parser() -> argparse.Namespace:
    """
    Parses command-line arguments for configuring the Tic-tac-toe training environment and agent.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        prog="TicTacToe",
        description="Trains an agent to play Tic-tac-toe using reinforcement learning algorithms.",
    )

    # General Configuration
    parser.add_argument(
        "-rs", "--random-seed",
        dest="random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    # Agent Configuration
    parser.add_argument(
        "-a", "--algorithm",
        dest="algorithm",
        type=str,
        choices=["policy_gradient", "deep_q_learning"],
        default="policy_gradient",
        help="Reinforcement learning algorithm to use."
    )

    # Trainer Configuration
    parser.add_argument(
        "-lr", "--learning-rate",
        dest="learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for training."
    )
    parser.add_argument(
        "-ne", "--num-episodes",
        dest="num_episodes",
        type=int,
        default=100_000,
        help="Total number of training episodes."
    )
    parser.add_argument(
        "-g", "--gamma",
        dest="gamma",
        type=float,
        default=1.0,
        help="Discount factor for future rewards (0 <= gamma <= 1)."
    )
    parser.add_argument(
        "-e", "--epsilon",
        dest="epsilon",
        type=float,
        default=1.0,
        help="Initial epsilon value for epsilon-greedy exploration."
    )
    parser.add_argument(
        "-em", "--epsilon-min",
        dest="epsilon_min",
        type=float,
        default=0.01,
        help="Minimum epsilon value for epsilon-greedy exploration."
    )
    parser.add_argument(
        "-dr", "--decay-rate",
        dest="decay_rate",
        type=float,
        default=0.9999,
        help="Decay rate for epsilon value per episode."
    )
    parser.add_argument(
        "-ms", "--memory-size",
        dest="memory_size",
        type=int,
        default=500_000,
        help="Replay memory size for experience replay. Set to 1 for no memory."
    )
    parser.add_argument(
        "-bs", "--batch-size",
        dest="batch_size",
        type=int,
        default=128,
        help="Batch size for training updates."
    )

    # Environment Configuration
    parser.add_argument(
        "-fs", "--field-size",
        dest="field_size",
        type=int,
        default=3,
        help="Size of the tic-tac-toe board (field_size x field_size)."
    )

    # Model Configuration
    parser.add_argument(
        "-dp", "--dropout-probability",
        dest="dropout_probability",
        type=float,
        default=0.0,
        help="Dropout probability for the policy network."
    )
    parser.add_argument(
        "-l", "--layers",
        dest="num_layers",
        type=int,
        default=1,
        help="Number of hidden layers in the neural network."
    )
    parser.add_argument(
        "-hu", "--hidden-units",
        dest="num_hidden_units",
        type=int,
        default=128,
        help="Number of units per hidden layer."
    )
    parser.add_argument(
        "-mn", "--model-name",
        dest="model_name",
        type=str,
        default=None,
        help="Name of the model to load for training or evaluation."
    )

    return parser.parse_args()
