"""Tic-Tac-Toe Game Engine.

A flexible implementation of the Tic-Tac-Toe game, supporting arbitrarily large playing fields.

Typical Usage:
    # Create a Tic-Tac-Toe environment and play against an agent.
    env = TicTacToe(size=3)
    env.play(model)
"""

import copy
import torch
import torch.nn as nn
from src.agent import Agent


class Environment:
    """Base class for all environments."""

    def __init__(self) -> None:
        """Initializes the Environment base class."""
        self.debug = False


class TicTacToe(Environment):
    """Tic-Tac-Toe Game Environment.

    A reinforcement learning environment for playing Tic-Tac-Toe.

    Attributes:
        size (int): Size of the Tic-Tac-Toe playing field (size x size).
        field (torch.Tensor): Tensor representing the game board.
    """

    def __init__(self, size: int) -> None:
        """
        Initializes the Tic-Tac-Toe environment.

        Args:
            size (int): Size of the playing field (e.g., size=3 for a 3x3 board).
        """
        super().__init__()
        self.size = size
        self.field = torch.zeros(size=(size, size), dtype=torch.long)

    @torch.no_grad()
    def _has_won(self, player: int) -> bool:
        """
        Checks if the given player has won the game.

        Args:
            player (int): Player ID (1 for "X" or -1 for "O").

        Returns:
            bool: True if the player has won, False otherwise.
        """
        criterion = player * self.size

        # Check rows, columns, and diagonals
        if any(self.field[i, :].sum() == criterion for i in range(self.size)) or \
           any(self.field[:, i].sum() == criterion for i in range(self.size)) or \
           self.field.diagonal().sum() == criterion or \
           torch.rot90(self.field).diagonal().sum() == criterion:
            return True

        return False

    @torch.no_grad()
    def _is_draw(self) -> bool:
        """
        Checks if the game has ended in a draw.

        Returns:
            bool: True if the game is a draw, False otherwise.
        """
        return torch.all(self.field != 0)

    @torch.no_grad()
    def is_finished(self) -> tuple[bool, int]:
        """
        Determines if the game is finished and identifies the winner.

        Returns:
            tuple[bool, int]: (is_finished, winner) where:
                - is_finished (bool): True if the game is over.
                - winner (int): 1 for player "X", -1 for player "O", 0 for a draw.
        """
        if self._has_won(player=1):
            return True, 1
        elif self._has_won(player=-1):
            return True, -1
        elif self._is_draw():
            return True, 0
        return False, 0

    def is_free(self, x: int, y: int) -> bool:
        """
        Checks if a specific position on the board is free.

        Args:
            x (int): Row index.
            y (int): Column index.

        Returns:
            bool: True if the position is free, False otherwise.
        """
        return self.field[x, y] == 0

    def mark_field(self, x: int, y: int, player: int) -> None:
        """
        Marks a position on the board for the specified player.

        Args:
            x (int): Row index.
            y (int): Column index.
            player (int): Player ID (1 for "X", -1 for "O").
        """
        self.field[x, y] = player

    def _index_to_coordinate(self, index: int) -> tuple[int, int]:
        """
        Converts a flat action index into 2D board coordinates.

        Args:
            index (int): Flat index representing an action.

        Returns:
            tuple[int, int]: (row, column) coordinates.
        """
        return divmod(index, self.size)

    def step(self, action: int, player: int) -> tuple:
        """
        Executes a single game move for the given player.

        Args:
            action (int): Flat index representing the chosen action.
            player (int): Player ID (1 for "X", -1 for "O").

        Returns:
            tuple: (state, reward, done) where:
                - state (torch.Tensor): The new game state.
                - reward (float): Reward for the action.
                - done (bool): True if the game has ended.
        """
        x, y = self._index_to_coordinate(action)

        if self.is_free(x, y):
            self.mark_field(x, y, player)
            is_finished, winner = self.is_finished()
            reward = 1.0 if is_finished and winner == player else 0.0
        else:
            reward = -1.0  # Penalty for illegal moves
            is_finished = True

        state = self.field.float()[None, ...]
        return state, reward, is_finished

    def run_episode(self, agent_a: Agent, agent_b: Agent) -> tuple:
        """
        Plays a single episode between two agents.

        Args:
            agent_a (Agent): The first agent.
            agent_b (Agent): The second agent.

        Returns:
            tuple: Events for each agent, containing states, actions, rewards, new_states, and dones.
        """
        events_a, events_b = {k: [] for k in ["states", "actions", "rewards", "new_states", "dones"]}, \
                             {k: [] for k in ["states", "actions", "rewards", "new_states", "dones"]}

        state = self.reset()
        done = False

        while not done:
            # Agent A's turn
            action = agent_a.get_action(state)
            new_state, reward, done = self.step(action, player=-1)
            events_a["states"].append(copy.deepcopy(state))
            events_a["actions"].append(action)
            events_a["rewards"].append(reward)
            events_a["new_states"].append(copy.deepcopy(new_state))
            events_a["dones"].append(done)

            if done and reward == 1:
                events_b["rewards"][-1] = -1
            state = new_state

            # Agent B's turn
            if not done:
                action = agent_b.get_action(state)
                new_state, reward, done = self.step(action, player=1)
                events_b["states"].append(copy.deepcopy(state))
                events_b["actions"].append(action)
                events_b["rewards"].append(reward)
                events_b["new_states"].append(copy.deepcopy(new_state))
                events_b["dones"].append(done)

                if done and reward == 1:
                    events_a["rewards"][-1] = -1
                state = new_state

        return events_a, events_b

    def play(self, model: nn.Module) -> None:
        """Allows a human to play against the model."""
        print("\nGame started.\n")
        done = False
        state = self.reset()

        while not done:
            print("Machine's Move:")
            action = model.predict(state)
            state, reward, done = self.step(action, player=-1)
            print(self)

            if done:
                print("You lose." if reward == 1 else "Illegal move. Computer lost." if reward == -1 else "Draw.")
                break

            action = int(input(f"Enter an index [0, {self.size**2 - 1}]: "))
            state, reward, done = self.step(action, player=1)
            print(self)

            if done:
                print("You win!" if reward == 1 else "Illegal move. Computer won." if reward == -1 else "Draw.")

    def reset(self) -> torch.Tensor:
        """Resets the game board to the initial empty state."""
        self.field = torch.zeros(size=(self.size, self.size), dtype=torch.long)
        return self.field.float()[None, ...]

    def __repr__(self) -> str:
        """Returns a string representation of the game board."""
        prototype = "{:3}" * self.size
        board = "\n".join(prototype.format(*row) for row in self.field.tolist())
        return f"\n{board.replace('-1', ' x').replace('1', ' o').replace('0', '.')}"

