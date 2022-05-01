# -*- coding: utf-8 -*-
"""
Description:
This module allows to get two strategies to play a single game (run_h2h_game) or
compete over a series of games (compare_strategies)

Example:
>>> from connectfour import strategies
>>> run_h2h_game(x_strategy=strategies.RandomStrategy(), 
             o_strategy=strategies.LeftMostStrategy())
Output: {'result': 1, 'game_length': 12}
Todo:

"""

import datetime
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from connectfour.config import ConnectFourGameConfig
from connectfour.game import ConnectFourGameState, Player
from connectfour.logging import GameLogger
from connectfour.strategies import ChooseActionStrategy

TIME_RECORD_FORMAT = '%Y-%m-%d %H:%M:%S'
DURATION_PRECISION: int = 6


class GameTimer:
    """a class to time each move during a game, 
    and """
    def __init__(self):
        self.last_move = time.perf_counter()
        self.moves_duration = []
    
    def record_move(self) -> None:
        """records time taken to perform a move."""
        new_time = time.perf_counter()
        self.moves_duration.append(new_time - self.last_move)
        self.last_move = new_time
    
    def get_avg_moves_duration(self) -> tuple[float, float]:
        """return avg move time for each player, so far in the game."""
        x_moves_avg_duration = np.mean(self.moves_duration[::2]).round(DURATION_PRECISION)
        o_moves_avg_duration = np.mean(self.moves_duration[1::2]).round(DURATION_PRECISION)
        
        return x_moves_avg_duration, o_moves_avg_duration
    
    def get_current_time(self) -> str:
        """returns time of record in a formatted way"""
        return datetime.datetime.now().strftime(TIME_RECORD_FORMAT)



def _get_h2h(x_strategy: ChooseActionStrategy,
             o_strategy: ChooseActionStrategy) -> str:
    """generates a single h2h label allowing to identify a game between the 
    two same players, independently of who starts."""
    
    return "_vs_".join(sorted([str(x_strategy), str(o_strategy)]))
        
def play_h2h_game(x_strategy: ChooseActionStrategy, 
                 o_strategy: ChooseActionStrategy, 
                 config: ConnectFourGameConfig) -> dict:
    """allows two strategies to compete on a single game.
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
        'winner': result_to_winner_map[state.game_result],  # type: ignore
           }
    

def play_multi_h2h_games(strat1: ChooseActionStrategy, 
                         strat2: ChooseActionStrategy, 
                         n_games: int = 100, 
                         config: ConnectFourGameConfig = None,
                         logpath: str = None) -> pd.DataFrame:
    """Allows 2 strategies to play against each other for `n_games` games.
    For fairness, the strategy starting alternates at every game.
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
            outcome = play_h2h_game(x_strategy=x_strat,
                                    o_strategy=o_strat,
                                    config=config)

            outcomes.append(outcome)
            if logpath:
                logger.log(outcome)
            pbar.update(1)

    outcomes = pd.DataFrame(outcomes)
    return outcomes


    
def tournament(strategies: dict[str, ChooseActionStrategy], 
               n_games: int = 100) -> pd.DataFrame:
    results = []
    for strat1, strat2 in itertools.combinations(strategies.keys(), 2):
        print(strat1, 'vs', strat2)
        outcomes = play_multi_h2h_games(strat1=strategies[strat1],
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