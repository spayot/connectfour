# -*- coding: utf-8 -*-
"""
Description:
This module allows for two 

Example:
>>> from connectfour import strategies
>>> run_game(x_strategy=strategies.random_strategy, 
             o_strategy=strategies.leftmost_strategy)
Output: {'result': 1, 'game_length': 12}
Todo:

Author: Sylvain Payot
E-mail: sylvain.payot@gmail.com
"""

import itertools
import sys
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .game import ConnectFourGameState, Player
from .config import ConnectFourGameConfig
from .strategies import ChooseActionStrategy


class GameTimer:
    def __init__(self):
        self.last_move = time.time()
        self.moves_duration = []
    def record_move(self):
        new_time = time.time()
        self.moves_duration.append(new_time - self.last_move)
        self.last_move = new_time
    
    def get_avg_moves_duration(self):
        x_moves_avg_duration = np.mean(self.moves_duration[::2]).round(6)
        o_moves_avg_duration = np.mean(self.moves_duration[1::2]).round(6)
        return x_moves_avg_duration, o_moves_avg_duration
        
        
        
def run_game(x_strategy: ChooseActionStrategy, 
             o_strategy: ChooseActionStrategy, 
             config: ConnectFourGameConfig) -> dict:
    """allows to strategies to compete on a single game.
    x strategy starts. o_strategy follows."""

    # initialize game state
    state = ConnectFourGameState(game_config=config)
    game_length = 0
    timer = GameTimer()
    while not state.is_game_over():
        choose_action = x_strategy if state.next_player == Player.x else o_strategy
        action = choose_action(state)
        state = state.move(action)
        game_length += 1
        timer.record_move()
    
    
    return {'game_length': game_length, 
            'result': state.game_result, 
            'avg_move_duration': timer.get_avg_moves_duration()}


def compare_strategies(strat1: ChooseActionStrategy, 
                       strat2: ChooseActionStrategy, 
                       n_games: int = 100, 
                       config: ConnectFourGameConfig = None) -> dict[str, int]:
    """Allows 2 strategies to play against each other for `n_games` games.
    For fairness, every game, the starting strategy alternates.
    Returns statistics about game outcomes and game length."""
    if config is None:
        config = ConnectFourGameConfig()
        
    outcomes = []
    with tqdm(total=n_games, file=sys.stdout) as pbar:
        for i in range(n_games):
            # alternate which 
            x_strat, o_strat = (strat1, strat2) if i % 2 == 0 else (strat2, strat1)
            outcome = run_game(x_strategy= x_strat,
                               o_strategy=o_strat,
                               config=config)

            result_factor = 1 if i % 2 == 0 else -1
            outcome['result'] = outcome['result'] * result_factor
            if i % 2 == 1:
                t_x, t_o = outcome['avg_move_duration']
                outcome['avg_move_duration'] = t_o, t_x

            outcomes.append(outcome)
            pbar.update(1)

    outcomes = pd.DataFrame(outcomes)
    return outcomes


def plot_strat_competition_outcomes(outcomes: pd.DataFrame, 
                                    strat1_name: str, 
                                    strat2_name: str) -> None:
    """plot outcomes of a competition between 2 strategies, as 
    defined in compare_strategies.
    Args:
        outcomes: a dataframe in the format of the output of `compare_strategies`
        strat1_name: a string to be used to label the first strategy
        strat2_name: 
    
    
    """
    fig, ax = plt.subplots(1,3, figsize=(15,4))

    t = (outcomes.result
         .value_counts(normalize=True)
         .sort_index(ascending=False)
         .to_frame()
         .T) # transpose
    t.columns = t.columns.map({-1: strat2_name + ' wins', 0: 'draw', 1: strat1_name + ' wins'})
    plots = t.plot(kind='barh', stacked=False, ax=ax[0], cmap='coolwarm')
    
    for bar, col in zip(plots.patches, t.columns):
        plots.annotate(f'{col}: {bar.get_width():.1%}',
                       (bar.get_x() + .01, # + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2 +.03), ha='left', va='center',
                       size=20, xytext=(0, 8),
                       textcoords='offset points')
        
    ax[0].set_xticks([0], [''])
    ax[0].set_yticks([])
    ax[0].set_xlabel('winner')
    ax[0].legend([])
    ax[0].invert_yaxis()
    
    # show move_duration
    durations = (pd.DataFrame(outcomes.avg_move_duration.tolist())
                 .rename(columns={0: strat1_name, 1: strat2_name})
                 .stack(level=0)
                 .reset_index(level=1)
                )
    durations.columns = ['player', 'time to move (s)']
    sns.boxplot(data=durations, 
                x='time to move (s)', 
                y='player',
                orient='h', 
                showfliers=False, 
                palette='coolwarm',
                saturation = 1,
                ax=ax[1])
    
    # show game length
    sns.histplot(outcomes.game_length, ax=ax[2], bins=43, stat='probability')
    ax[2].set_xlim([0,42])
    
    plt.tight_layout()
    plt.show()
    
    
def tournament(strategies: dict[str, ChooseActionStrategy], 
               n_games: int = 100) -> pd.DataFrame:
    results = []
    for strat1, strat2 in itertools.combinations(strategies.keys(), 2):
        print(strat1, 'vs', strat2)
        outcomes = compare_strategies(strat1=strategies[strat1],
                                      strat2=strategies[strat2], 
                                      n_games=n_games)
        strat1wins = (outcomes.result > 0).mean()
        strat2wins = (outcomes.result < 0).mean()
        print(strat1, f"wins {strat1wins:.1%} of the time\n")
        results.extend([
            {'strat1': strat1, 'strat2': strat2, 'strat1wins': strat1wins},
            {'strat1': strat2, 'strat2': strat1, 'strat1wins': strat2wins},
        ])
        
    tournament_results = pd.pivot(pd.DataFrame(results), 
                                  index='strat1', 
                                  columns='strat2', 
                                  values='strat1wins')

    return tournament_results

def show_tournament_results(tournament_results: pd.DataFrame, *args, **kwargs) -> None:
    sns.heatmap(tournament_results, 
                cmap='Blues', 
                annot=True, 
                fmt=".1%", 
                *args, **kwargs)