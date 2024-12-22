"""Play Against an Agent.

This script allows a user to play a game of Tic-Tac-Toe against a trained agent.

"""

from src.utils import print_args, load_checkpoint
from src.argparser import argument_parser
from src.model import Model
from src.environment import TicTacToe

if __name__ == "__main__":
    # Parse command-line arguments
    args = argument_parser()
    print_args(args=args)

    # Initialize the policy model and load trained weights
    model = Model(args=args)
    load_checkpoint(model=model, args=args)

    # Set up the Tic-Tac-Toe environment and play against the agent
    env = TicTacToe(size=args.field_size)
    env.play(model=model)
