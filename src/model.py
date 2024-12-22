"""Policy Model.

Defines a fully connected neural network with residual connections to represent 
the agent's policy. The model maps states to actions.

"""

import torch
import torch.nn as nn
from src.utils import eval


class ResidualBlock(nn.Module):
    """Residual Block for MLP-based models.

    A block with two linear layers, GELU activation, dropout, and layer normalization,
    ensuring the input is added back to the output for a residual connection.
    """

    def __init__(self, in_features: int, out_features: int, args) -> None:
        """
        Initializes a residual MLP block.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            args: Argument object containing model hyperparameters like dropout.
        """
        super().__init__()
        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.GELU(),
            nn.Linear(in_features=out_features, out_features=out_features),
            nn.Dropout(p=args.dropout_probability),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with residual connection.
        """
        return x + self.mlp_block(x)


class Model(nn.Module):
    """Fully Connected Neural Network for Policy Learning.

    This model maps the state of the playing field to a probability distribution
    over actions (policy gradient) or raw outputs (deep Q-learning).
    """

    def __init__(self, args) -> None:
        """
        Initializes the policy model.

        Args:
            args: Argument object containing model hyperparameters.
        """
        super().__init__()

        # Dimensions
        field_size = args.field_size
        input_dim = field_size**2
        hidden_dim = args.num_hidden_units
        num_actions = field_size**2
        num_layers = args.num_layers
        dropout_prob = args.dropout_probability

        # Input layer
        input_layer = [
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
        ]

        # Hidden layers with residual blocks
        hidden_layers = [
            ResidualBlock(in_features=hidden_dim, out_features=hidden_dim, args=args)
            for _ in range(num_layers)
        ]

        # Output layer
        output_layer = [
            nn.Linear(in_features=hidden_dim, out_features=num_actions),
            nn.Softmax(dim=-1) if args.algorithm == "policy_gradient" else nn.Identity(),
        ]

        # Assemble the full model
        self.model = nn.Sequential(*input_layer, *hidden_layers, *output_layer)

    @eval
    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> int:
        """
        Predicts the best action for a given state.

        Args:
            state (torch.Tensor): Tensor representing the current state of the playing field.

        Returns:
            int: The selected action as an integer.
        """
        prediction = self(state)
        action = torch.argmax(prediction, dim=-1).item()
        return action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing action values or policy probabilities.
        """
        return self.model(x)
