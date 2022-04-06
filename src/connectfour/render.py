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

import matplotlib.pyplot as plt
import numpy as np

from .game import ConnectFourGameState
from .mcts import MctsNode
from .selfplay import SelfPlayConfig
from .player import AzPlayer, TemperatureSchedule
from .pvnet import PolicyValueNet


def renderConnectFour(state: ConnectFourGameState, ax=None):
    radius = .3
    board = np.flip(state.board,axis=0)
    if not ax:
        fig, ax = plt.subplots(figsize=(3.5,3))
    
    X1, Y1 = np.where(board == 1)
    X2, Y2 = np.where(board == -1)
    
    n, m = board.shape
    
    ax.set(xlim=(-.5, m - .5), ylim = (-.5, n - .5))
    #from matplotlib.patches import Circle
    for x,y in zip(X1,Y1):
        c = plt.Circle((y,x), radius=radius, color='grey')
        ax.add_artist(c)

    for x,y in zip(X2,Y2):
        c = plt.Circle((y,x), radius=radius, color='lightblue')
        # n-y so that "gravity" is respected :)
        ax.add_artist(c)


    ax.set_xticks(-.5+np.arange(m))
    ax.set_yticks(-.5 + np.arange(n))
    ax.grid()

    ax.axes.set_xticklabels([])
    ax.axes.set_yticklabels([])

def render_improved_policy(state: ConnectFourGameState, config: SelfPlayConfig, ax=None):
    """show policy distribution for a given player config and a given state."""
    node = MctsNode(state, evaluator=config.evaluator)
    player = AzPlayer(evaluator=config.evaluator)
    
    # move number i is inferred from number of rounds
    i = np.abs(state.board.array).sum()+1
    
    action, policy = player.play(node, tau=config.tau.get(i), n_sims=config.n_sims)
    
    render_policy(policy, title=f"improved policy, $tau={config.tau}$", ax=ax)
        
def render_policy(policy: np.ndarray, title, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(3.5,.5))
    
    ax.bar(np.arange(len(policy)), policy)
    ax.set_title(title)
    ax.set_ylim([0,1])
    ax.set_xlim([-.5,6.5])
    ax.set_yticks([])
    ax.set_xticks(np.arange(8)-.5)
    ax.set_xticklabels([])
    ax.grid(which="major")
    
    
def visualize_policy_improvement(state: ConnectFourGameState, pvn_fname: str, tau: float=1, n_sims: int=100) -> None:
    """visualize policy distribution before and after MCTS-based policy improvement for a given state
    Arguments:
        state: the state from which the player is to evaluate its policy
        pvn_fname: the filepath to the policy value net model to use as pvn 
        tau: temperature value to smoothen the MCTS output. (the lower, the greedier the policy)
        n_sims: number of MCTS simulations to run to improve the raw policy"""
    # instantiate player based on filepath
    evaluator = PolicyValueNet(filename=pvn_fname, quiet=True)
    player = AzPlayer(evaluator=evaluator)
    
    # instantiate node
    node = MctsNode(state, evaluator)
    
    # render board
    renderConnectFour(state)
    
    # render raw policy (before MCTS improvement)
    p0, v = evaluator.evaluate_state(state)
    
    render_policy(policy=p0, title=f"raw policy - value {v:.2f}")
    
    # render policy with MCTS improvement
    config = SelfPlayConfig(evaluator=evaluator, 
                                        tau= tau,
                                       n_sims=n_sims)
    render_improved_policy(state, config)

