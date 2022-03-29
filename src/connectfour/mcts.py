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

import numpy as np

from .config import mcts_config
from .pvnet import PolicyValueNet

__author__ = "Sylvain Payot"
__copyright__ = "Sylvain Payot"
__license__ = "mit"

# hyperparameters

class MctsNode(object):
    def __init__(self, state, evaluator: PolicyValueNet, parent=None):
        self.state = state
        self.parent = parent
        self.evaluator = evaluator
        P, V = evaluator.evaluate_state(state)
        # V: estimated value of being in that state
        self.V = V
        # print(P, V) # temp
        self.actions = [MctsAction(state, move, prior=P[move.x_coordinate], parent=self) for move in state.get_legal_actions()]
        
        # if self.parent:
        #     self.parent.backpropagate(V)
    
    def __repr__(self):
        return f"MCTS NODE: \n{self.state}\n{self.V:.4f}"
    
   
    def get_ucb(self, action, total_node_visits):
        """computing UCB value for a given action"""
        ucb = mcts_config['C_PUCT'] * action.P * np.sqrt(total_node_visits)/ (1 + action.N)
        return ucb 
    
    def choose_action(self):
        """choose action with highest $Q(s,a)+U(s,a)$"""
        assert self.is_terminal_node() == False
        
        total_node_visits = np.sum([action.N for action in self.actions]) + 1
        
        # multiplying Q by next to move, so that action is chosen based to maximize the reward from the player's perspective
        next_to_move = self.state.next_player
        
        
        ucbs = [next_to_move.value * action.Q + self.get_ucb(action, total_node_visits) for action in self.actions]
        
        try:
            idx = np.argmax(ucbs)
        except ValueError:
            print(f"ValueError: empty array for ucbs on node {self}")
            raise
        
        return self.actions[idx]
    
    def backpropagate(self, V):
        if self.parent:
            self.parent.backpropagate(V)
            
    def is_terminal_node(self):
        return self.state.is_game_over()
    
    
    
class MctsAction(object):
    def __init__(self, state, move, prior, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.Q = 0
        self.N = 0
        self.W = 0
        self.P = prior
        self.child = None
        
    def __repr__(self):
        return f"MCTS Action:(Move:{self.move})"

        
    def describe(self):
        print(f"""
        move:      {self.move}
        Q-value:   {self.Q}
        N(visits): {self.N}""")
    
    def take_action(self):
        if self.N == 0:
            new_state = self.state.move(self.move)
            self.child = MctsNode(new_state, self.parent.evaluator, parent=self)
        return self.child
        
    def backpropagate(self, V):
        self.W += V
        self.N += 1
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backpropagate(V)
    
    
class MCTS(object):

    def __init__(self, node: MctsNode, tau: float=mcts_config['tau']):
        """
        Node
        Parameters
        ----------
        node : mctspy.tree.nodes.MctsAlaphaZeroNode
        """
        self.root = node
        self.tau = tau

    def policy_improvement(self, simulations_number) -> tuple[list, np.ndarray]:
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        Returns
        -------
        - list: actions
        - np.ndarray: improved policy, based on MCTS results

        """
        # run n simulations
        for _ in range(0, simulations_number): 
            # walk the state tree graph until running into an unexplored node
            n = self._tree_policy()
            # back propagate the node value
            n.backpropagate(n.V)

        # return improved policy based on number of visits of each actions (with a temperature coefficient and normalization step)
#         Ns = np.zeros(self.root.state.board_size[1])
#         for action in self.root.actions:
#             Ns[action.move.x_coordinate] = action.N 
        Ns = np.array([action.N for action in self.root.actions])
        
        # apply temperature factor
        Ns = np.power(Ns, (1 / self.tau))
        
        # normalize
        Ns = Ns / Ns.sum()
        
        return self.root.actions, Ns
        

    def _tree_policy(self):
        """
        selects node to playout until new action is expanded.

        Returns
        -------

        """
        # start from root
        current_node = self.root
        while not current_node.is_terminal_node():
            # choose action
            action = current_node.choose_action()
            # if node unexplored, end while loop and return the new node visited.
            if action.N == 0:
                return action.take_action()
            # if node explored, update current_node and restart loop.
            else:
                current_node = action.take_action()
        
        return current_node
        

    

