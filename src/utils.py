"""Utility Functions.

Contains helper functions for setting random seeds, saving/loading model checkpoints,
printing arguments, and managing evaluation mode for PyTorch models.

"""

import warnings
import functools
import pathlib
import random

import numpy as np
import torch


def set_random_seed(seed: int = 0) -> None:
    """
    Sets the random seed for reproducibility across libraries.

    Args:
        seed (int): Random seed value. Default is 0.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(model: torch.nn.Module, model_name: str, args) -> None:
    """
    Saves the model's state dictionary as a checkpoint.

    Args:
        model (torch.nn.Module): PyTorch model to save.
        model_name (str): Name to assign to the checkpoint file.
        args: Parsed arguments containing algorithm information.
    """
    checkpoint_name = f"{model_name}_{args.algorithm}" if model_name else "model"
    checkpoint_path = pathlib.Path("weights")
    checkpoint_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    model_path = checkpoint_path / f"{checkpoint_name}.pth"

    torch.save(model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")


def load_checkpoint(model: torch.nn.Module, args) -> None:
    """
    Loads a model's state dictionary from a checkpoint file.

    Args:
        model (torch.nn.Module): PyTorch model to load weights into.
        args: Parsed arguments containing the model name and algorithm information.
    """
    checkpoint_name = f"{args.model_name}_{args.algorithm}" if args.model_name else "model"
    checkpoint_path = pathlib.Path("weights") / f"{checkpoint_name}.pth"

    if checkpoint_path.is_file():
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
        print(f"\nModel '{checkpoint_name}' loaded successfully.\n")
    else:
        warnings.warn(
            f"\nModel checkpoint '{checkpoint_name}' not found. "
            f"Continuing with random weights.\n"
        )


def print_args(args) -> None:
    """
    Prints parsed arguments in a clean format to the console.

    Args:
        args: Parsed arguments to display.
    """
    print("Configuration:\n")
    for key, value in vars(args).items():
        print(f"{key:.<32}{value}")
    print()


def eval(function: callable) -> callable:
    """
    Decorator to set a PyTorch module to evaluation mode during inference.

    Ensures the model switches to evaluation mode before the function executes
    and back to training mode afterward.

    Args:
        function (callable): The function to wrap.

    Returns:
        callable: Wrapped function.
    """

    @functools.wraps(function)
    def eval_wrapper(self, *args, **kwargs):
        self.eval()
        output = function(self, *args, **kwargs)
        self.train()
        return output

    return eval_wrapper
