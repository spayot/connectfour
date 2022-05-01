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

from typing import Optional

import numpy as np

from .game import ConnectFourAction, ConnectFourGameState, Player
from .pvnet import PolicyValueNet

__author__ = "Sylvain Payot"
__copyright__ = "Sylvain Payot"
__license__ = "mit"

# hyperparameters

class MctsNode:
    def __init__(self, 
                 state: ConnectFourGameState, 
                 parent=None,
                 c_puct: float = 4.):
        """expresses a game-state in the form of a Monte Carlo Tree Search node.
        This node is created at Tree Expansion stage in the MCTS process. 
        
        Attributes:
            state: the Game State this node is representing
            parent: a MctsAction. defaults to None. if None: the node is the root
                of the search tree.
            V: the expected value for that state (derived from evaluator)
            actions (list[MctsActions]): the list of legal child MctsAction for this state.
            c_puct: a coefficient driving how much weight is given to the evaluator's prior
                vs the results of MCTS simulations. Defaults to 4.
        
        Resources:
            https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
        """
        self.state = state
        self.parent = parent # MctsAction
        self.c_puct = c_puct
        
        self.is_expanded: bool = False
        self.V: Optional[float] = None # estimated value of being in that state
        self.actions: Optional[list[MctsAction]] = None
        
        
    
    def __repr__(self) -> str:
        return f"MctsNode( \n{self.state})"
    
    def expand(self, evaluator: PolicyValueNet):
        """"""
        # at tree expansion, we need to estimate the value 
        P, V = evaluator.evaluate_state(self.state)
        # V: estimated value of being in that state
        self.V = V
        self.actions = [MctsAction(self.state, move, prior=P[move.x_coordinate], parent=self) for move in self.state.get_legal_actions()]
        self.is_expanded = True
   
    def get_ucb(self, action):
        """computing Upper-Confidence Bound (UCB) value for a given child action following.
        UCB = C_PUCT * Prior_{Action} * sqrt(total_node_visits) / (1 + action_visits)
        
        Args:
            action (MctsAction): the child action to compute ucb for
            total_node_visits: the total number of visits for this node
        """
        ucb = self.c_puct * action.P * np.sqrt(self.total_node_visits)/ (1 + action.N)
        return ucb 
    
    @property
    def total_node_visits(self):
        """the number of times this node ahs been visited."""
        if not self.is_expanded:
            return 1
        return np.sum([action.N for action in self.actions]) + 1
    
    def choose_action(self):
        """Choose the child action with highest $Q(s,a)+U(s,a)$ value
        
        Returns:
            MctsAction: the child action selected.
            """
        assert self.is_terminal_node() == False
        
        # multiplying Q by next to move, so that action is chosen to maximize the reward 
        # from the current player's perspective
        next_to_move: Player = self.state.next_player
        
        
        ucbs: list[float] = [next_to_move.value * action.Q + self.get_ucb(action) for action in self.actions]
        
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
        return self.state.is_game_over
    
    
    
class MctsAction:
    def __init__(self, 
                 state: ConnectFourGameState, 
                 move: ConnectFourAction, 
                 prior: float, 
                 parent: Optional[MctsNode] = None):
        self.state = state
        self.move = move 
        self.parent = parent # MctsNode
        self.Q: float = 0. # Q(s,a) action value
        self.N: int = 0 # number of times this action was selected during the Selection process
        self.W: float = 0. # sum of values of all expanded child nodes
        self.P: float = prior # action's value based on the original estimator
        self.child: Optional[MctsNode] = None # 
        
    def __repr__(self):
        return f"MCTS Action(move={self.move}, prior={self.P}, q={self.Q})"

        
    def describe(self):
        print(f"""
        move:      {self.move}
        Q-value:   {self.Q}
        N(visits): {self.N}""")
    
    
    def take_action(self, prune: bool = False):
        """
        Args:
            prune: prunes the other branches of the tree search. 
                should be set to False during Monte Carlo Tree Search, and 
                True when executing the final action selected after MCTS.
                
        Returns:
            MctsNode: the node derived from selecting that action."""
        if self.N == 0:
            new_state = self.state.move(self.move)
            self.child = MctsNode(new_state, parent=self)
        
        if prune:
            self.child.parent = None # type: ignore
        
        return self.child
        
    def backpropagate(self, V):
        """Every time MCTS expands the search tree to a child node following that action, the
        expected value for each action is updated through backpropagation"""
        self.W += V
        self.N += 1
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backpropagate(V)
    
    
class TruncatedMCTS(object):
    def __init__(self, 
                 root: MctsNode, 
                 evaluator: PolicyValueNet, 
                 tau: float):
        """
        Node
        Parameters
        ----------
        root : the MctsNode to start the tree search from.
        tau: the temperature controlling the explore/exploit trade-off
        evaluator: PolicyValueNet used to evaluate nodes and children actions at
            expansion time.
        """
        self.root = root
        if not root.is_expanded:
            root.expand(evaluator)
            
        self.evaluator = evaluator
        self.tau = tau

        
    def policy_improvement(self, simulations_number: int) -> tuple[list[MctsAction], np.ndarray]:
        """

        Args:
            simulations_number: number of simulations performed to get the best action

        Returns:
            list: actions
            np.ndarray: improved policy, based on MCTS results

        """
        # run n simulations
        for _ in range(0, simulations_number): 
            # walk the state tree graph until running into an unexplored node (selection and tree expansion)
            node = self._tree_policy()
            # back propagate the node value
            node.backpropagate(node.V) # type: ignore

        # improved policy is proportional to the number of times each action
        # was selected during the simulations
        assert self.root.actions is not None, "the root node has not been expanded"
        
        Ns = np.array([mcts_action.N for mcts_action in self.root.actions]) 
        
        # apply temperature factor
        Ns = np.power(Ns, (1 / self.tau))
        
        # normalize
        Ns = Ns / Ns.sum()
        
        return self.root.actions, Ns
        

    
    def _tree_policy(self) -> MctsNode:
        """
        Selects nodes to playout in the search tree it expands to a new node.
        Approach:
            Go down the search tree by selecting at each step the action with 
            highest $Q(s,a)+U(s,a)$ until running into an unexplored node (meaning, 
            the action leading to it has 0 visits)
        
        Returns: 
            MctsNode: the new node after expansion

        """
        # start from root
        current_node = self.root
        
        # go down the search tree by selecting higher $Q(s,a)+U(s,a)$
        while not current_node.is_terminal_node():
            # choose action with highest $Q(s,a)+U(s,a)$
            action = current_node.choose_action()
            
            # define new node given the chosen action
            current_node = action.take_action(prune=False)
            
            # if node unexplored, end while loop, expand new node and return it
            if not current_node.is_expanded:
                current_node.expand(evaluator=self.evaluator)
                return current_node
            
            # if node already explored, continue down the tree
                
        # handle terminal nodes
        return current_node
        