# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
[options.entry_points] section in setup.cfg:

    console_scripts =
         fibonacci = puissance4.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""

import os

import numpy as np

from .game import ConnectFourGameState, ConnectFourAction
from .pvnet import PolicyValueNet

__author__ = "Sylvain Payot"
__copyright__ = "Sylvain Payot"
__license__ = "mit"

class Evaluator(object):
    def __init__(self, pvn: PolicyValueNet = None, 
                 name: str = None):
        """
        Abstraction class for any function that evaluates 
        Arguments:
        - pvn: a policy value net predicting value of position, and probability of each next action being the best.
        - name: """
        if not pvn:
            pvn = PolicyValueNet(name=name) # initialize with an untrained pvn
        
        if not name:
            name = pvn.name
            
        self.pvn = pvn
        self.name = name
    
    def evaluate_state(self, state: ConnectFourGameState) -> (float, np.ndarray):
        """ Policy and Value estimations based on the current state.
        Arguments:
            - state: current state
        
        Returns: 
        - policy: 7x1 array providing a probability distribution of each move being the best move for the player due to play.
        - value: expected outcome of the game (1 for a win of player 1, -1 for a win of player 2, 0 for a draw)
        
        Example:
        >>> evaluator = Evaluator(name='test')
        >>> evaluator.evaluate_state(ConnectFourGameState(board=np.zeros(6,7), next_player=1))"""
        
        if state.is_game_over():
            p,v = np.ones(7)/7, state.game_result
        
        else: 
            p, v = self.pvn.infer_from_state(state)
        
        return p, v
    
    @classmethod
    def from_file(cls, fname):
        gen_name = os.path.split(fname)[-1].split('.')[0]
        pvn = PolicyValueNet(filename=fname, name=gen_name, quiet=True)
        return cls(pvn=pvn)
