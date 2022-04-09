# -*- coding: utf-8 -*-
"""
Description:
   Defines a class-based strategy pattern used in the compete module.
   Each strategy takes a ConnectFourGameState as an input and outputs a ConnectFourGameAction.

Example:
    /
    
TO DOs:
* MCTS strategy without policy/value evaluator
* Truncated MCTS with evaluator
"""

from typing import Protocol

import numpy as np
from scipy import stats

from .game import ConnectFourGameState, ConnectFourAction, Player
from .config import ConnectFourGameConfig
from .pvnet import PolicyValueNet
from .temperature import TemperatureSchedule
from .player import AzAgent


class ChooseActionStrategy(Protocol):
    def initialize_game(self) -> None:
        """re-initializes the game state strategy know that a new game is starting.
        Not needed for all strategies, but mandatory for the AlphaZero, which builds
        its own representation of the current state to navigate the search tree."""
        ...
    
    def select_action(self, state: ConnectFourGameState) -> ConnectFourAction:
        """defines how the strategy selects the next action."""
        ...
    

class RandomStrategy:
    def initialize_game(self) -> None:
        pass
    
    def select_action(self, state: ConnectFourGameState) -> ConnectFourAction:
        """randomly selects any legal action for a given state

        Args:
            state: a connect four game state

        Returns:
            ConnectFourAction: a legal action for that state randomly selected"""
        possible_actions = state.get_legal_actions()
        return np.random.choice(possible_actions)
    
    def __repr__(self) -> str:
        return "RandomStrategy()"
    
    def __str__(self) -> str:
        return "random"
    
    
class LeftMostStrategy:    
    def initialize_game(self) -> None:
        pass
    
    def select_action(self, state: ConnectFourGameState) -> ConnectFourAction:
        """always plays left-most legal action.

        Args:
            state: a connect four game state

        Returns:
            ConnectFourAction: a legal action for that state randomly selected"""
        possible_actions = state.get_legal_actions()
        return possible_actions[0]
    
    def __repr__(self) -> str:
        return "LeftMostStrategy()"
    
    def __str__(self) -> str:
        return "leftmost"


class RawPVNetStrategy:
    def __init__(self, evaluator: PolicyValueNet, temperature: float = 1):
        """evaluates value of each action using a PolicyValueNet
        and selects the action based on the raw policy for that state (no MCTS).
        
        Args:
            evaluator: """
        self.evaluator = evaluator
        self.temperature = temperature
    
    def initialize_game(self):
        pass
    
    def select_action(self, state: ConnectFourGameState) -> ConnectFourAction:
        """Evaluates each action's value (policy) expressed as a probability of being the best action, 
        then samples one action from that policy distribution.
        """
        policy, value = self.evaluator.infer_from_state(state)
        return choose_action_from_policy(state, policy, self.temperature)
    
    def __repr__(self) -> str:
        return f"RawPVNStrategy({self.evaluator.name},temperature={self.temperature})"
    
    def __str__(self) -> str:
        return "raw_pvn"
        
        
class MctsPvnStrategy:
    def __init__(self, evaluator: PolicyValueNet, 
                 temperature_schedule: TemperatureSchedule = None,
                 n_sims: int = 100):
        """AlphaZero Agent, building a truncated search tree
        with prioritized exploration thanks to the evaluator.
        
        Args:
            evaluator: the policy-value evaluation model that allows to estimate the 
                value of each action and future states.
            temperature_schedule: a schedule defining how greedily the next
                action should be selected.
            n_sims: the number of MCTS simulations to run to improve on the raw policy.
        """
        
        if temperature_schedule is None:
            temperature_schedule = TemperatureSchedule()
            
            
        self.agent = AzAgent(evaluator=evaluator, 
                             temperature_schedule=temperature_schedule)
        self.n_sims = n_sims
        self.agent.initialize_game()
    
    
    def __repr__(self) -> str:
        return f"MctsPvnStrategy(evaluator={self.agent.evaluator.name}, n_sims={self.n_sims}, tau={str(self.agent.temperature_schedule)})"
    
    def __str__(self) -> str:
        return f"az_{self.n_sims}_{str(self.agent.temperature_schedule)}"
        
    
    def initialize_game(self) -> None:
        self.agent.initialize_game()
        
    def select_action(self, state) -> ConnectFourAction:
        if state.last_action is not None:
            self.agent.update(state.last_action)
        else:
            self.agent.initialize_game()
            
        action, policy = self.agent.select_next_action(self.n_sims)
        self.agent.update(action)
        
        return action
        
        
        
def choose_action_from_policy(state: ConnectFourGameState, 
                              policy: np.ndarray,
                              temperature: float = 1.) -> ConnectFourAction:
    """Selects an action based on a policy distribution after applying 
    a temperature factor. 
    
    Args: 
        state: the state before the player
        policy: an array describing a distribution of probability that each child action is the best.
            the array is expected to be of the same size than the board (# of columns).
        temperature: a temperature factor. the lower the temperature, the more greedy the selection.
        
    Returns:
        ConnectFourAction"""
    
    # get legal_actions
    legal_actions = state.get_legal_actions()
    
    # filter out illegal actions from policy (assumes policy is a vector with fixed size based on shape)
    legal_policy = [policy[action.x_coordinate] for action in legal_actions]
    
    # apply temperature factor
    legal_policy = np.power(legal_policy, 1 / temperature)
    
    # normalize
    legal_policy = legal_policy / legal_policy.sum()
    
    # translate policy into a discrete distribution
    action_ids = np.arange(len(legal_actions))

    action_chooser = stats.rv_discrete(name='policy_sampler', values=(action_ids, legal_policy))

    # sample from that distribution
    action_id = action_chooser.rvs()
    
    # return corresponding action
    return legal_actions[action_id]