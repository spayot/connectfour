# -*- coding: utf-8 -*-
"""
This script primarily defines the AzPlayer class, as well as the way historical
games are recorded.

AzPlayer is an agent that can play Connect Four, using a Monte-Carlo Tree Search 
approach, prioritized and truncated thanks to a state evaluation function, defined
in the `pvnet` module.

"""

import os
import pickle

import numpy as np
from scipy import stats

from .game import ConnectFourGameState
from .pvnet import PolicyValueNet
from .mcts import MctsNode, MctsAction, MCTS
from .config import board_config, selfplay_config, mcts_config

class GameHistory(object):
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
        self.history = []
        self.outcome = None
    
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


class TemperatureSchedule(object):
    """defines the evolution of temperature parameter tau at each round of the game.
    The temperature helps to control the level of exploration-exploitation tradeoff 
    by either smoothing the move distribution obtained after MCTS (tau > 1) or increasing 
    on the contrary thechances to select the best candidate identified for the next 
    step (tau << 1).
    
    Note: tau = 1 means that the distribution will be proporational to the number of 
    node visits.
    
    
    Args:
        tau_start (float, optional): the value for tau for the first steps
        threshold (int, optional): the number of steps until tau switches from 
        tau_start to tau_end.
        tau_end (float, optional): the value 
        
    Note: default values are defined in the config module of this package.
    """
    def __init__(self, tau_start: float=selfplay_config["tau_start"],  
                threshold: int=selfplay_config["threshold"],
                tau_end: float= selfplay_config["tau_end"]):
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.threshold = threshold
        self.tau = [tau_start if i < threshold else tau_end for i in range(board_config["width"] * board_config["height"])]
    
    def get(self, i: int) -> float:
        """get the temperature value for the i-th move.
        Args:
            i (int): the move number.
        
        Returns:
            float: the specified temperature value for this move."""
        return self.tau[i]
    
    def __repr__(self):
        return f"{self.tau_start}-{self.threshold}-{self.tau_end}".replace('.', '')


class AzPlayer(object):
    """an agent that can play Connect Four, using a Monte-Carlo Tree Search 
    approach, prioritized and truncated thanks to a state evaluation function, defined
    in the `pvnet` module.
    
    Args: 
        evaluator (puissance4.pvnet.PolicyValueNet): the evaluation function that outputs
            a policy and a state value prediction, given an input state.
            
    Attributes:
        evaluator (puissance4.pvnet.PolicyValueNet): the evaluation function that outputs
            a policy and a state value prediction, given an input state.
        memory: a list of GameHistory objects, corresponding to the records of
        all games played by this player.
        
    Example:
    >>> from puissance4.
    
    """
    def __init__(self, evaluator: PolicyValueNet):
        self.evaluator = evaluator
        self.memory = [] # a list of GameHistory objects
        
    def play(self, node: MctsNode, tau: float, n_sims: int) -> (MctsAction, np.ndarray):
        """implements sampling from available actions, using MCTS-based policy improvement """
        
        assert node.is_terminal_node() == False, f"node is a terminal node: {node.state}"
        
        mcts = MCTS(node, tau=tau)
        
        # get mcts-improved policy, and updates tree Q, N, ...
        actions, policy = mcts.policy_improvement(n_sims)
        
        assert abs((policy.sum()-1) < 1e-8), f"sum of policy is not equal to 1:\n{policy}"
        
        # transform policy into format similar to evaluator output (np.ndarray)
        Pi = np.zeros(node.state.board.shape[1])
        Pi[[action.move.x_coordinate for action in actions]] = policy
        
        # returns the index of the 
        xk = np.arange(len(actions))
        try: 
            action_chooser = stats.rv_discrete(name='policy_sampler', values=(xk, policy))
        except ValueError:
            print(f"policy is not summing to 1:\n {policy}")
            raise
        
        # sample from the distribution
        chosen_action = actions[action_chooser.rvs()]
        
        return chosen_action, Pi
        
    
    def self_play(self, tau: TemperatureSchedule = TemperatureSchedule(**selfplay_config), 
                  n_sims: int = mcts_config["sims"]) -> None:
        """player plays against itself and saves its improved policies and game outcome into a GameHistory object"""
        # initialize game
        game_history = GameHistory()
        
        if type(tau)==float:
            tau = TemperatureSchedule(tau_start=tau, threshold=0, tau_end=tau)
        
        starting_player = np.random.choice([-1,1])
        initial_board_state = ConnectFourGameState(board=np.zeros((6,7), dtype=np.int64), next_to_move=starting_player)
        node = MctsNode(state = initial_board_state, evaluator=self.evaluator)
        
        i = 0
        
        # play game
        while not node.is_terminal_node():
            action, policy = self.play(node, tau=tau.get(i), n_sims=n_sims)
            game_history.add_move(node, policy)
            node = action.take_action()
            
            # discard rest of tree
            node.parent = None
            
            i += 1
        
        outcome = node.state.game_result
        
        # add last node to the history
        game_history.add_move(node, np.ones(board_config['width']) / board_config['width'])
        
        # update final game outcome
        game_history.update_outcome(outcome)
        self.memory.append(game_history)
    
    def save_history(self, filepath, return_history=False, open_mode='ab'):
        """saves history from all games played so far into <filepath> in a pickle format"""
        assert open_mode in ['ab', 'wb'], "improper open_mode. needs to be either 'ab' or 'wb'"
        
        # vectorize memory
        boards = np.array([s.board for g in self.memory for s, p, r in g.history])
        next_to_move = np.array([s.next_to_move for g in self.memory for s, p, r in g.history])
        policies = np.array([p for g in self.memory for s, p, r in g.history])
        results = np.array([r for g in self.memory for s, p, r in g.history])
        
        # create object to save:
        history = {
            'player_name': self.evaluator.name,
            'input_boards': boards,
            'input_next_to_move': next_to_move,
            'output_policy': policies,
            'output_value': results}
        
        # if append=True, history is appended at the end of a pre-existing memory file
        
        with open(filepath, open_mode) as f:
            pickle.dump(history, f)
            
        if return_history:
            return history
        
        
    def __repr__(self) -> str:
        return __class__.__name__ + f"(evaluator={self.evaluator})"
        
