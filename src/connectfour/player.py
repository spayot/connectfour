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

from .config import ConnectFourGameConfig, board_config, selfplay_config
from .game import ConnectFourAction, ConnectFourGameState, Player
from .mcts import MctsAction, MctsNode, TruncatedMCTS
from .pvnet import PolicyValueNet
from .record import GameRecord
from .temperature import TemperatureSchedule


class AzAgent:
    def __init__(self,
                 evaluator: PolicyValueNet,
                 game_config: ConnectFourGameConfig = None,
                 temperature_schedule: TemperatureSchedule = None,
                 c_puct: float = 4.):
        
        if game_config is None:
            game_config=ConnectFourGameConfig()
            
        if temperature_schedule is None:
            temperature_schedule = TemperatureSchedule()
            

        self.evaluator = evaluator
        self.game_config = game_config
        self.temperature_schedule = temperature_schedule
        self.c_puct = c_puct
        self.memory: list[GameRecord] = []
        
        assert (evaluator.n, evaluator.m) == game_config.shape, """evaluator is not adapted to the game_config."""
        
    def initialize_game(self, next_player: Player = Player.x) -> None:
        """initialize current_node to the beginning of a game"""
        initial_state = ConnectFourGameState(next_player=next_player, game_config=self.game_config)
        self.current_node = MctsNode(state=initial_state, c_puct=self.c_puct)
        

    def select_next_action(self, n_sims: int) -> tuple[ConnectFourAction, np.ndarray]:
        """implements sampling from available actions, using MCTS-based policy improvement """
        
        assert self.current_node.is_terminal_node() == False, f"node is a terminal node: {self.current_node.state}"
        
        mcts = TruncatedMCTS(root=self.current_node, 
                             tau=next(self.temperature_schedule), 
                             evaluator=self.evaluator)
        
        # get mcts-improved policy, and updates tree Q, N, ...
        actions, policy = mcts.policy_improvement(n_sims)
        
        
        # transform policy into format similar to evaluator output (np.ndarray)
        Pi = np.zeros(self.game_config.shape[1])
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
        
        return chosen_action.move, Pi
    
    def update(self, action: ConnectFourAction) -> None:
        """update the current node based on which action was played
        by the opponent.
        
        Args:
            action: the action played by the opponent"""
        
        # find MctsAction corresponding to input action
        if not self.current_node.is_expanded:
            self.current_node.expand(self.evaluator)
        mcts_action = [a for a in self.current_node.actions if a.move == action]
        assert len(mcts_action) == 1, f"{action} does not match any child Mcts Actions for node {self.current_node}"
        
        mcts_action = mcts_action[0] # one and only one action should match
        
        self.current_node = mcts_action.take_action(prune=True)
        



class AzPlayer:
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
        self.memory = [] # a list of GameRecord objects
        
        
    
    def play_single_turn(self, node: MctsNode, tau: float, n_sims: int) -> tuple[MctsAction, np.ndarray]:
        """implements sampling from available actions, using MCTS-based policy improvement """
        
        assert node.is_terminal_node() == False, f"node is a terminal node: {node.state}"
        
        mcts = TruncatedMCTS(root=node, tau=tau, evaluator=self.evaluator)
        
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
    
    
        
    
    def self_play(self, 
                  tau: TemperatureSchedule, 
                  n_sims: int) -> None:
        """player plays one game against itself and saves its improved policies and game outcomes into a GameRecord object"""
        # initialize game
        game_record = GameRecord()
        
        if type(tau)==float:
            tau = TemperatureSchedule(tau_start=tau, threshold=0, tau_end=tau)
        
        starting_player = np.random.choice([-1,1])
        initial_board_state = ConnectFourGameState(next_player=starting_player)
        node = MctsNode(state = initial_board_state)

        
        # play game
        while not node.is_terminal_node():
            # select aciton and output policy
            action, policy = self.play_single_turn(node, tau=next(tau), n_sims=n_sims)
            game_record.add_move(node, policy)
            node = action.take_action(prune=True)
            
        
        outcome = node.state.game_result
        
        # add terminal node to the history
        board_width = node.state.game_config.shape[1]
        game_record.add_move(node, np.ones(board_width) / board_width)
        
        # update final game outcome
        game_record.update_outcome(outcome)
        self.memory.append(game_record)
    
    def save_history_to_pickle(self, 
                     filepath: str, 
                     return_history: bool = False, 
                     open_mode: str = 'ab'):
        """saves history from all games played so far into <filepath> in a pickle format"""
        
        assert open_mode in ['ab', 'wb'], "improper open_mode. needs to be either 'ab' or 'wb'"
        
        # vectorize memory
        boards = np.array([state.board for game in self.memory for state, policy, result in game.history])
        next_to_move = np.array([state.next_to_move for game in self.memory for state, policy, result in game.history])
        policies = np.array([policy for game in self.memory for state, policy, result in game.history])
        results = np.array([result for game in self.memory for state, policy, result in game.history])
        
        # create object to save:
        history = {
            'player_name': self.evaluator.name,
            'input_boards': boards,
            'input_next_to_move': next_to_move,
            'output_policy': policies,
            'output_value': results}
        
        # if open_mode == 'ab', history is appended at the end of a pre-existing memory file
        
        with open(filepath, open_mode) as f:
            pickle.dump(history, f)
            
        if return_history:
            return history
        
        
    # def __repr__(self) -> str:
    #     return self.__class__.__name__ + f"(evaluator={self.evaluator})"
        






        
        
        
        
            
    
    