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

from .game import ConnectFourGameState, ConnectFourAction
from .config import mcts_config
from .pvnet import PolicyValueNet

__author__ = "Sylvain Payot"
__copyright__ = "Sylvain Payot"
__license__ = "mit"

# hyperparameters

class MctsNode:
    def __init__(self, 
                 state: ConnectFourGameState, 
                 evaluator: PolicyValueNet, 
                 parent=None,
                 c_puct: float = 4):
        """expresses a game-state in the form of a Monte Carlo Tree Search node.
        This node is created at Tree Expansion stage in the MCTS process. 
        
        Attributes:
            state: the Game State this node is representing
            evaluator: a callable that takes a state as an input and outputs a 
                policy distribution (np.ndarray of shape (7,)) and an expected
                value for that state (float between -1 and 1)
            parent: a MctsAction. defaults to None. if None: the node is the root
                of the search tree.
            V: the expected value for that state (Derived from evaluator)
            actions (list[MctsActions]): the list of legal child MctsAction for this state.
            c_puct: a coefficient driving how much weight is given to the evaluator's prior
                vs the results of MCTS simulations. Defaults to 4.
        
        Resources:
            https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
        """
        self.state = state
        self.parent = parent # MctsAction
        self.evaluator = evaluator
        self.c_puct = c_puct
        
        # at tree expansion, we need to estimate the value 
        P, V = evaluator.evaluate_state(state)
        # V: estimated value of being in that state
        self.V = V
        self.actions = [MctsAction(state, move, prior=P[move.x_coordinate], parent=self) for move in state.get_legal_actions()]
        
    
    def __repr__(self) -> str:
        return f"MCTS NODE: \n{self.state}\n{self.V:.4f}"
    
   
    def get_ucb(self, action, total_node_visits: int):
        """computing Upper-Confidence Bound (UCB) value for a given child action following.
        UCB = C_PUCT * Prior_{Action} * sqrt(total_node_visits) / (1 + action_visits)
        
        Args:
            action (MctsAction): the child action to compute ucb for
            total_node_visits: the total number of visits for this node
        """
        ucb = self.c_puct * action.P * np.sqrt(total_node_visits)/ (1 + action.N)
        return ucb 
    
    def choose_action(self):
        """Choose the child action with highest $Q(s,a)+U(s,a)$ value
        
        Returns:
            MctsAction: the child action selected.
            """
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
    
    def backpropagate(self, V: float) -> None:
        """Every time MCTS expands the search tree to a child node following that action, the
        expected value for each action is updated through backpropagation"""
        if self.parent:
            self.parent.backpropagate(V)
            
    def is_terminal_node(self) -> bool:
        """defines whether the game is over or not."""
        return self.state.is_game_over()
    
    
    
class MctsAction:
    def __init__(self, 
                 state: ConnectFourGameState, 
                 move: ConnectFourAction, 
                 prior: float, 
                 parent=None):
        self.state = state
        self.move = move 
        self.parent = parent # MctsNode
        self.Q: float = 0. # Q(s,a) action value
        self.N: int = 0 # number of times this action was selected during the Selection process
        self.W: float = 0. # sum of values of all expanded child nodes
        self.P: float = prior # action's value based on the original estimator
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
        """Every time MCTS expands the search tree to a child node following that action, the
        expected value for each action is updated through backpropagation"""
        self.W += V
        self.N += 1
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backpropagate(V)
    
    
class MCTS(object):
    def __init__(self, node: MctsNode, tau: float):
        """
        Node
        Parameters
        ----------
        node : the MctsNode to start hte tree search from.
        tau: 
        """
        self.root = node
        self.tau = tau

    def policy_improvement(self, simulations_number: int) -> tuple[list[MctsAction], np.ndarray]:
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
            # walk the state tree graph until running into an unexplored node (selection and tree expansion)
            node = self._tree_policy()
            # back propagate the node value
            node.backpropagate(node.V)

        # improved policy is proportional to the number of times each action
        # was selected during the simulations
        Ns = np.array([mcts_action.N for mcts_action in self.root.actions])
        
        # apply temperature factor
        Ns = np.power(Ns, (1 / self.tau))
        
        # normalize
        Ns = Ns / Ns.sum()
        
        return self.root.actions, Ns
        

    def _tree_policy(self) -> MctsNode:
        """
        Selects nodes to playout in the search tree it expands to a new node.

        Returns: 
            MctsNode: the new node after expansion

        """
        # start from root
        current_node = self.root
        while not current_node.is_terminal_node():
            # choose action based with highest $Q(s,a)+U(s,a)$
            action = current_node.choose_action()
            # if node unexplored, end while loop and return the new node visited.
            if action.N == 0:
                return action.take_action()
            # if node explored, update current_node and restart loop (Selection).
            else:
                current_node = action.take_action()
        
        return current_node
        

    

