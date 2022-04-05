import numpy as np

from .game import ConnectFourGameState
from .mcts import MctsNode


class GameRecord:
    """standardized format to record historical connect four games. 
    Args:
        None
        
    Attributes:
        history (list): a list of tuples (state, improved_policy), with 
            state being a ConnectFourGameState object, and improved_policy
            being a 7x1 array corresponding to the probability distribution 
            obtained after MCTS simulations.
        outcome (int): the game result (1: player 1 wins, -1: player 2 wins,
            0: draw)
        
    """
    def __init__(self):
        self.history: list[tuple[ConnectFourGameState, np.ndarray]] = []
        self.outcome: int = None # 1 for 
    
    def add_move(self, node: MctsNode, improved_policy: np.ndarray):
        """records a new (state, policy) tuple to the game history"""
        self.history.append((node.state, improved_policy))
    
    def update_outcome(self, outcome: int):
        """adds final outcome of a game to GameHistory (-1 or 1)"""
        self.history = [move + (outcome,) for move in self.history]
        self.outcome = outcome

    
    @property
    def game_length(self):
        """returns the number of moves played until the game was over."""
        return len(self.history)