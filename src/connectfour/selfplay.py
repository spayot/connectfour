# -*- coding: utf-8 -*-
"""
Self Play
"""

import os
import datetime
import time
from tqdm.notebook import tqdm

import numpy as np

from .pvnet import PolicyValueNet
from .player import AzPlayer, TemperatureSchedule
from .config import mcts_config, selfplay_config


class SelfPlayConfig(object):
    def __init__(self, evaluator: PolicyValueNet, n_sims: int, 
                 tau: TemperatureSchedule=TemperatureSchedule(**selfplay_config)):
        """if tau is constant across the game, the TemperatureSchedule can be replaced by a float value. 
        A TemperatureSchedule is automatically created based on that constant."""
        if type(tau) in (float, int):
            tau = TemperatureSchedule(tau_start=tau, tau_end=tau, threshold=0)
        self.evaluator = evaluator
        self.n_sims = n_sims
        self.tau = tau
    
    def __repr__(self):
        return f"{self.evaluator.name}_{self.n_sims}_{self.tau}"
    
    def _log_as_csv(self):
        return f"{self.evaluator.name},{self.n_sims},{self.tau}"

    

def write_game_stats_log_line(config: SelfPlayConfig, 
                              game_length: int, game_duration: float, 
                              log_path: str, log_mode: str='a') -> None:
    fname = str(config) + ".log"
    filepath = os.path.join(log_path, fname)
    date = datetime.datetime.now().strftime('%x %X')
    sep = ','
    s = sep.join([date, config._log_as_csv(), str(game_length), f"{game_duration:.3f}"])
    if not os.path.exists(filepath):
        s = "date,evaluator_name,n_sims,tau,game_length,game_duration"
    with open(filepath, log_mode) as f:
        print(s, file=f) 
    return filepath


def self_play_stats(config: SelfPlayConfig, 
                    n_games: int,
                    log_path: str = 'logs/', 
                    train_data_path: str='data/',
                    verbose: bool = False) -> (np.ndarray, np.ndarray):
    """self-plays `n_games` with a given evaluator, using a specified number of simulations, and temperature. 
    game statistics are saved in the logs, 
    (state, policy, value) tuples to be used for training data are used in data/.
    
    Arguments:
        config: a SelfPlayConfig object defining which evaluator, how many simulations are performed and the temperature.
        n_games: number of self-play games to play
        verbose: set to True to print status progress every 10 steps
        log_path: 
    
    Returns:
        np.ndarray: number of moves required to complete each self-play game
        np.ndarray: time (in secs) required to complete each self-play game
       """
    print("starting self-play:".upper())
    start = time.time()
    
    # intialize_stats
    game_duration, games_length = [], []
    with tqdm(total=n_games) as pbar:
        for i in range(n_games):
            st = time.time()

            # instantiate player and self play
            player = AzPlayer(evaluator=config.evaluator)
            player.self_play(tau=config.tau, n_sims=config.n_sims)

            et = time.time()

            # log stats
            filepath = write_game_stats_log_line(log_path=log_path, 
                                                  config=config,
                                                  game_duration=et-st,
                                                  game_length=player.memory[0].game_length)

            # save training data
            fname = str(config) + ".pkl"
            train_data_fullpath = os.path.join(train_data_path, fname)
            player.save_history(filepath=train_data_fullpath, open_mode='ab')
            
            pbar.update(1)

            if ((i+1) % 10 == 0) & verbose:
                # create string in format "HH:MM:SS"
                time_since_start = str(datetime.timedelta(seconds=int(time.time() - start)))
                pbar.write(f"{time_since_start} | {i+1} games performed")
        
    print("self-play completed. logs available at:".upper())
    print(filepath)
    print("training data saved in:".upper())
    print(train_data_fullpath)

