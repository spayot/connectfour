# -*- coding: utf-8 -*-
"""

Abstract Game classes: https://github.com/int8/monte-carlo-tree-search
"""

from dataclasses import dataclass
from enum import Enum
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

from .config import ConnectFourGameConfig


class Player(Enum):
    x = 1
    o = -1
    
    def get_next(self):
        if self == Player.x:
            return Player.o
        else:
            return Player.x

    
class ConnectFourAction: 
    def __init__(self, x_coordinate: int, player: Player):
        """Actions in ConnectFour are defined by who is the player and
        which column do they insert their disc `x_coordinate`."""
        self.x_coordinate = x_coordinate
        self.player = player

    def __repr__(self) -> str:
        return f"x:{self.x_coordinate} p:{self.player.value}"
    

def _board_to_str(array: np.ndarray) -> str:
    """controls the way a board array should be represented as a string."""
    s = " " + str(array)
    o = str(Player.o.value)
    x = str(Player.x.value)
    to_replace = {
        '[': '', 
        ']': '', 
        '0': '.', 
        o: 'O'.rjust(len(o)), # rjust ensures the replacement has the same # of chars
        x: 'X'.rjust(len(x)), 
    } 
    for arg in to_replace.items():
        s = s.replace(*arg)

    return s

def _assert_board_is_valid(board: np.ndarray, game_config: ConnectFourGameConfig):
    """a set of assert statements to validate that the input array is in line with 
    the game configuration."""
    
    # validate shape
    assert board.shape == game_config.shape, f"""the board shape is {board.shape}, but the 
    game configuration defines a {game_config.shape} shape"""
        
    # verify that dtype is valid
    assert np.issubdtype(board.dtype, np.number), "board array should be of dtype np.int* "

    # verify that values are valid
    valid_values = {Player.x.value, Player.o.value, 0}
    assert set(np.unique(board)).issubset(valid_values), f"array values must be either {valid_values}"


class ConnectFourGameState: 
    def __init__(self, board: np.ndarray = None, 
                 next_player: Player = Player.x,
                 game_config: ConnectFourGameConfig = ConnectFourGameConfig()):
        """
        Args:
            board (np.ndarray, optional): an array of dtype int representing the board state.
                the array should only include values corresponding to the two players' values (1, -1),
                or 0 to denotate the absence of coin in a specific slot.
                Defaults to None. if None, an empty board is created."""
        if board is None:
            # initialize to an empty board
            board = np.zeros(shape=game_config.shape, dtype=int)
        else:
            _assert_board_is_valid(board, game_config)
        
        self.board = board
        self.next_player = next_player
        self.game_config = game_config
        
        
    def __repr__(self):
        return f"ConnectFourGameState(board:\n{_board_to_str(self.board)}\n\tnext_player={self.next_player.name}"
    
    @property
    def game_result(self):
        """determines whether the game state corresponds to an end state and which player won if any. 
        the game is won if one player aligns 4 coins next to each other, either across a row, a column or a diagonal"""
        # this condition can be interpreted as a series of convolutions across the board.
        
        # define number of coins in a row necessary to win (default: 4)
        N_TO_WIN = self.game_config.n_to_win
        
        # create kernels for convolution
        row_kernel = np.ones((1, N_TO_WIN))
        col_kernel = np.ones((N_TO_WIN, 1))
        diag1_kernel = np.eye(N_TO_WIN)
        diag2_kernel = np.zeros((N_TO_WIN, N_TO_WIN))
        
        for i in range(N_TO_WIN):
            diag2_kernel[i, N_TO_WIN-i-1] = 1
        
        kernels = [row_kernel, col_kernel, diag1_kernel, diag2_kernel]
        
        # apply convolutions
        convolutions = np.array([convolve(self.board, kernel, mode='constant', cval=0) for kernel in kernels])

        # check whether player X won
        if convolutions.max() == N_TO_WIN * Player.x.value:
            # player one has at least one 
            return Player.x.value
        
        # check whether player O won
        if convolutions.min() == N_TO_WIN * Player.o.value:
            return Player.o.value
        
        # check for draws
        if np.all(self.board != 0):
            return 0

        # if not over - no result
        return None

    
    def is_game_over(self) -> bool:
        """checks whether the game is over or not"""
        return self.game_result is not None

    
    def is_move_legal(self, action):
        # check if correct player moves
        if action.player != self.next_player:
            return False

        # check if inside the board on x-axis
        x_in_range = (0 <= action.x_coordinate < self.board.shape[1])
        if not x_in_range:
            return False

        # finally check if board column not full yet
        return self.board[0, action.x_coordinate] == 0

    
    def move(self, action):
        # check for move validity
        assert self.is_move_legal(action), f"move {action} on board is not legal"
        
        new_board = self.board.copy()
        
        # calculate y_coordinate: given x_position, disc is assigned to lowest position available (ie board value == 0) on the board
        y_coordinate = np.where(self.board[:,action.x_coordinate] == 0)[0].max()
        
        # assign new position to player
        new_board[y_coordinate, action.x_coordinate] = action.player.value
        
        # get next player
        next_player = self.next_player.get_next()

        return ConnectFourGameState(board=new_board, 
                                    next_player=self.next_player.get_next(), 
                                    game_config=self.game_config)

    
    def get_legal_actions(self):
        indices = np.where(self.board[0,:] == 0)[0]
        return [
            ConnectFourAction(idx, self.next_player)
            for idx in indices
        ]