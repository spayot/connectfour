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

import datetime
import itertools
import json
import logging
import os
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
from .logging import GameLogger

PALETTE = 'coolwarm'

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
    
    def get_current_time(self) -> str:
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')



def _get_h2h(x_strategy: ChooseActionStrategy,
             o_strategy: ChooseActionStrategy) -> str:
    return "_vs_".join(sorted([str(x_strategy), str(o_strategy)]))
        
def run_game(x_strategy: ChooseActionStrategy, 
             o_strategy: ChooseActionStrategy, 
             config: ConnectFourGameConfig) -> dict:
    """allows to strategies to compete on a single game.
    x strategy starts. o_strategy follows."""

    # initialize game state
    state = ConnectFourGameState(game_config=config)
    x_strategy.initialize_game()
    o_strategy.initialize_game()
    
    game_length = 0
    timer = GameTimer()
    
    while not state.is_game_over:
        strategy = x_strategy if state.next_player == Player.x else o_strategy
        action = strategy.select_action(state)
        state = state.move(action)
        game_length += 1
        timer.record_move()
    
    avg_move_1, avg_move_2 = timer.get_avg_moves_duration()
    
    result_to_winner_map = {1: str(x_strategy), 0: 'draw', -1: str(o_strategy)}
    
    return {
        'time': timer.get_current_time(),
        'h2h': _get_h2h(x_strategy, o_strategy),
        'strat1': str(x_strategy),
        'strat2': str(o_strategy),
        'result': state.game_result, 
        'game_length': game_length, 
        'avg_move_1': round(avg_move_1, 4),
        'avg_move_2': round(avg_move_2, 4),
        'winner': result_to_winner_map[state.game_result],
           }
    

def compare_strategies(strat1: ChooseActionStrategy, 
                       strat2: ChooseActionStrategy, 
                       n_games: int = 100, 
                       config: ConnectFourGameConfig = None,
                       logpath: str = None) -> dict[str, int]:
    """Allows 2 strategies to play against each other for `n_games` games.
    For fairness, every game, the starting strategy alternates.
    Returns statistics about game outcomes and game length."""
    if config is None:
        config = ConnectFourGameConfig()
    
    if logpath:
        logger = GameLogger(logpath)
    
    outcomes = []
    with tqdm(total=n_games, file=sys.stdout) as pbar:
        for i in range(n_games):
            # alternate which 
            x_strat, o_strat = (strat1, strat2) if i % 2 == 0 else (strat2, strat1)
            outcome = run_game(x_strategy=x_strat,
                               o_strategy=o_strat,
                               config=config)

            outcomes.append(outcome)
            if logpath:
                logger.log(outcome)
            pbar.update(1)

    outcomes = pd.DataFrame(outcomes)
    return outcomes

def _get_durations(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """transforms records dataframe into format necessary for 
    `compete.plot_strat_competition_outcomes`
    
    Args:
        records: records in direct format extracted from compete.log
        
    Returns:
        durations: a dataframe 
    """
    
    strat = pd.concat([records['strat1'], records['strat2']])
    avg_move_duration = pd.concat([records['avg_move_1'], records['avg_move_2']])
    
    return  pd.DataFrame({'player': strat, 'time to move (s)': avg_move_duration})


def subplot_results(records, ax):
    """barplot showing share of winners"""
    t = (records.winner
         .value_counts(normalize=True)
         .to_frame()
         .T) # transpose
    
    strat1, strat2 = records.h2h.values[0].split('_vs_')
    
    if 'draw' not in t.columns:
        t['draw'] = 0
    
    preferred_order = [strat1, 'draw', strat2]
    
    t = t.loc[:, preferred_order]
    
    plots = t.plot(kind='barh', stacked=False, ax=ax, cmap=PALETTE)
    
    for bar, col in zip(plots.patches, t.columns):
        plots.annotate(f'{col}: {bar.get_width():.1%}',
                       (bar.get_x() + .01, # + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2 +.03), ha='left', va='center',
                       size=10, xytext=(0, 8),
                       textcoords='offset points')
        
    ax.set_xlim([0,1])
    ax.set_xticks([0], [''])
    ax.set_yticks([])
    ax.set_xlabel('winner')
    ax.legend([])
    ax.invert_yaxis()
    
    
def subplot_durations(records: pd.DataFrame, ax):
    """boxplot showing the distribution of playing speed."""
    # show move_duration
    durations = _get_durations(records)
    durations.columns = ['player', 'distribution of time to move (s)']
    sns.violinplot(data=durations, 
                x='distribution of time to move (s)', 
                y='player',
                orient='h', 
                showfliers=False, 
                palette=PALETTE,
                saturation = 1,
                ax=ax)
    # # strat1, strat2 = records.h2h.values[0].split('_vs_')
    # durations.reset_index(inplace=True, drop=True)
    # sns.kdeplot(data=durations, x='time to move (s)', hue='player', shade='fill', ax=ax, palette='coolwarm')
    ax.set_xlim([1e-4,10])
    ax.set_xscale('log')
    ax.set_yticklabels(ax.get_yticklabels(),rotation = 90)

    
def subplot_game_length(records, ax):
    """show game length"""
    sns.histplot(records.game_length, ax=ax, bins=43, stat='probability')
    ax.set_xlim([0,42])

def plot_strat_competition_outcomes(records: pd.DataFrame, 
                                    figdir: str = None) -> None:
    """plot outcomes of a competition between 2 strategies, as 
    defined in compare_strategies.
    Args:
        records: a dataframe in the format of the output of `compare_strategies`
        strat1_name: a string to be used to label the first strategy
        strat2_name: a string to be used to label the second strategy
    
    
    """
    fig, ax = plt.subplots(1,3, figsize=(15,3))
    
    subplot_results(records, ax[0])
    
    subplot_durations(records, ax[1])
    
    subplot_game_length(records, ax[2])
    
    plt.tight_layout()
    
    if figdir:
        figure_path = os.path.join(figdir, f"{records.h2h.values[0]}.svg")
        plt.savefig(figure_path, facecolor='white')
        
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
                cmap=PALETTE, 
                annot=True, 
                fmt=".1%", 
                *args, **kwargs)