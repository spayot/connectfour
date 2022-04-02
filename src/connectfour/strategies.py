# -*- coding: utf-8 -*-
"""
Description:
   Defines a function-based Strategy Pattern used in the compete module.
   Each strategy takes a ConnectFourGameState as an input and outputs a ConnectFourGameAction.

Example:
    /
    
Todo:
* raw evaluator (PVNet) without MCTS
* MCTS strategy without policy/value evaluator
* Truncated MCTS with evaluator
"""

from typing import Callable

import numpy as np
from scipy import stats

from .game import ConnectFourGameState, ConnectFourAction, Player
from .config import ConnectFourGameConfig
from .pvnet import PolicyValueNet
# from .mcts import 


# strategy pattern is implemented here as a function with defined input and output
ChooseActionStrategy = Callable[[ConnectFourGameState], ConnectFourAction]

def random_strategy(state: ConnectFourGameState) -> ConnectFourAction:
    """randomly selects any legal action for a given state
    
    Args:
        state: a connect four game state
        
    Returns:
        action: a ConnectFourAction"""
    possible_actions = state.get_legal_actions()
    return np.random.choice(possible_actions)
    
    
def leftmost_strategy(state: ConnectFourGameState) -> ConnectFourAction:
    """always plays left-most legal action."""
    possible_actions = state.get_legal_actions()
    return possible_actions[0]


REGISTERED_STRATEGIES = {
    'random': random_strategy,
    'leftmost': leftmost_strategy,
}

class RawPVNetStrategy:
    """evaluates value of each action using a PolicyValueNet
    and selects the action based on the raw policy for that state (no MCTS)."""
    def __init__(self, pvn: PolicyValueNet, temperature: float = 1):
        self.pvn = pvn
        self.temperature = temperature
        
    def select_action(self, state: ConnectFourGameState) -> ConnectFourAction:
        """evaluates each action's value (policy) expressed as a probability of being the best action, 
        then samples one action from that policy distribution.
        """
        policy, value = self.pvn.infer_from_state(state)
        return choose_action_from_policy(state, policy, self.temperature)
        
        
# class MctsPvnStrategy:
#     def __init__(self, pvn: PolicyValueNet):
#         pass
        
        
def choose_action_from_policy(state: ConnectFourGameState, 
                              policy: np.ndarray,
                              temperature: float = 1.) -> ConnectFourAction:
    
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