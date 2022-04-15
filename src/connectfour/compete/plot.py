import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

COLOR_PALETTE = 'coolwarm'

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
    
    return  pd.DataFrame({'player': strat, 'distribution of time to move (s)': avg_move_duration})



def subplot_results(records: pd.DataFrame, ax: plt.Axes):
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
    
    plots = t.plot(kind='barh', stacked=False, ax=ax, cmap=COLOR_PALETTE)
    
    for bar, col in zip(plots.patches, t.columns):
        plots.annotate(f'{col}: {bar.get_width():.1%}',
                       (bar.get_x() + .01, # + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2 +.03), ha='left', va='center',
                       size=10, xytext=(0, 8),
                       textcoords='offset points')
        
    ax.set(xlim=[0,1], xticks=[0], xticklabels=[''], 
           xlabel='winner', yticks=[])
           
    ax.legend([])
    ax.invert_yaxis()
    
    
def subplot_durations(records: pd.DataFrame, ax: plt.Axes):
    """boxplot showing the distribution of playing speed."""
    # show move_duration
    durations = _get_durations(records)
    
    sns.violinplot(data=durations, 
                x='distribution of time to move (s)', 
                y='player',
                orient='h', 
                showfliers=False, 
                palette=COLOR_PALETTE,
                saturation = 1,
                ax=ax)
    
    ax.set(xlim=[1e-2,10], xscale='log')
    ax.set_yticklabels(ax.get_yticklabels(),rotation = 90)

    
def subplot_game_length(records: pd.DataFrame, ax: plt.Axes):
    """histogram showing the game length"""
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
    
    
def show_tournament_results(tournament_results: pd.DataFrame, *args, **kwargs) -> None:
    ax = sns.heatmap(tournament_results, 
                     cmap=COLOR_PALETTE, 
                     annot=True, 
                     fmt=".1%", 
                     *args, **kwargs)
    
    plt.show()