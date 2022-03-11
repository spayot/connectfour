# -*- coding: utf-8 -*-
"""
Suite of functions to plot CDF (cumulated density functions)
for the distribution of game length.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def cdf(array: np.ndarray) -> (np.ndarray, np.ndarray):
    """returns x and y vectors to plot the cumulated density function
    of an input array
    Args:
        array: array for which we want to calculate a cdf
    Returns:
        np.ndarray: sorted array of array's values
        np.ndarray: percentile associated with this value
    """
    N = len(array)
    x = np.sort(array)
    pct = np.array(range(N))/float(N)
    return x, pct

def _plot_self_play_cdf(df: pd.DataFrame, ax):
    """"""
    x,y = cdf(df.game_length)
    ax.plot(x,y, label=f"{df.evaluator_name[0]}_{df.n_sims[0]}_{df.tau[0]}")


def get_logs(config_list: List[str]= [], logpath: str= 'logs/') -> List[pd.DataFrame]:
    """returns a list of dataframes with log data"""
    if not config_list:
        # if no config list, plot cdf for all the log files in logpath
        config_list = sorted([file.split('.')[0] for file in os.listdir(logpath) if file[-4:] == '.log'])
    dfs = []
    for config in config_list:            
        dfs.append(pd.read_csv(os.path.join(logpath, config + ".log")))
    return dfs
    
def plot_log_cdfs(config_list: List[str]= [], logpath: str= 'logs/', ax=None) -> None:
    """plots game length CDFs for all configs listed in config_list"""
    dfs = get_logs(config_list, logpath)
    
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(10,7))
    
    for df in dfs:
        _plot_self_play_cdf(df, ax)
    ax.set_title("distribution of number of moves per game");
    plt.xlim([5,43])
    plt.ylim([0,1])
    plt.xlabel("number of moves before game over")
    ax.legend();
    


def plot_time_per_move(config_list=None, ax=None, logpath='logs/'):
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(15,6))
    if config_list==None:
        # if no config list, plot cdf for all the log files in logpath
        config_list = sorted([file.split('.')[0] for file in os.listdir(logpath) if file[-4:] == '.log'])
    
    df = pd.concat(get_logs(config_list))


    df['config']= df.evaluator_name + '_' + df.n_sims.astype(str) + '_' + df.tau.astype(str)
    df['time_per_move'] = df.game_duration / df.game_length

   
    sns.boxplot(x="config", 
                y="time_per_move", 
                data=df,
                ax=ax)
    ax.set_title("time per move (sec)")
    plt.ylim([0,2])
    
    
def compare_players(config_list=None, logpath='logs/'):
    fig, ax = plt.subplots(1,2,figsize=(20,6))
    plot_log_cdfs(config_list,ax=ax[0], logpath=logpath)
    plot_time_per_move(config_list,ax=ax[1])
    plt.tight_layout()
    
def summary_stats(logpath='logs/'):
    config_list = sorted([file.split('.')[0] for file in os.listdir(logpath) if file[-4:] == '.log'])
    df = pd.concat(get_logs(config_list))
    df['time_per_move'] = df.game_duration / df.game_length
    df['config'] = df.evaluator_name + '_' + df.n_sims.astype(str) + '_' + df.tau.astype(str)
    df = df.groupby('config')["game_length", "time_per_move"].aggregate({"game_length": "mean", "time_per_move": "median"})
    
    return df.sort_values('game_length', ascending=False)