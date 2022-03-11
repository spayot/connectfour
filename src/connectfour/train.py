# -*- coding: utf-8 -*-
"""

- collect and concatenate last n gens of games
- augment data
- de-duplicate
- sample x parts
- load last gen model, compile
- train for e epochs
- save
"""

import glob
import os
import pickle

import numpy as np
import tensorflow as tf

from .pvnet import crossentropy_loss, PolicyValueNet


__author__ = "Sylvain Payot"
__copyright__ = "Sylvain Payot"
__license__ = "mit"

#------------------------------------------------------------------------------------------------------------------------
# reload historical data and consolidate for training
#------------------------------------------------------------------------------------------------------------------------

        
def _load_self_play_data_from_pickle(filepath: str) -> list:
    """"""
    data = []
    with open(filepath, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return data

def consolidate_selfplay_from_pickle(picklefile):
    """consolidation of separate game history into single sets of numpy arrays"""
    data = _load_self_play_data_from_pickle(picklefile)
    # concatenate data from all games
    data_c = {key: np.concatenate([game[key] for game in data]) for key in data[0].keys() if key != 'player_name'}
    return data_c


def get_normalized_hist_data(fname):
    """transform board view and the value target to consistently evaluate the board from the perspective of player 1"""
    hist = consolidate_selfplay_from_pickle(fname)
    b = hist['input_next_to_move'].reshape(-1,1,1) * hist['input_boards']
    v = hist['input_next_to_move'] * hist['output_value']
    p = hist['output_policy']
    return b,v,p


def get_training_dataset(next_gen: int, n_gens: int, batch_size: int=64, desired_steps: int=2000) -> tf.data.Dataset:
    """extract all historical data for the past n_gens before current_gen and turns it into a tf.data.Dataset 
    that can be ingested for training purposes of the policy-value network for the new generation.
    get_training_dataset successivily performs the following operations:
    - extract historical data from all relevant files
    - normalize the data to consistently evaluate the board from the perspective of player 1
    - augment the data by leveraging the game symmetry
    - consolidate the datset by grouping together all identical boards and averaging policy and values
    - finally, turn the dataset into a tf.data.Dataset"""
    # get all files to retrieve data from
    files = []
    for j in range(next_gen - n_gens,next_gen):
        files += glob.glob(f'data/gen{j}*.pkl')



    boards,values,policies = [], [], []

    for fname in files:
        b,v,p = get_normalized_hist_data(fname)
        boards.append(b)
        values.append(v)
        policies.append(p)

    boards = np.concatenate(boards)
    values = np.concatenate(values)
    policies = np.concatenate(policies)

    # augment with symmetries
    boards = np.concatenate([boards, np.flip(boards, axis=2)])
    policies = np.concatenate([policies, np.flip(policies, axis=1)])
    values = np.concatenate([values, values])

    # consolidate by grouping duplicates and taking mean values and policies (Note: idx shows which group each historical board is assigned to)
    boards, idx = np.unique(boards, axis=0, return_inverse=True)
    values = np.array([values[idx==i].mean() for i in np.unique(idx)])
    policies = np.array([policies[idx==i].mean(axis=0) for i in np.unique(idx)])
    
    # expand dimension to turn it into a shape (,6,7,1)
    boards = np.expand_dims(boards, axis=-1)
    
    n_samples = values.shape[0]
    
    print(f"loaded {n_samples:,} unique historical board positions from the past {min(n_gens, next_gen)} generations")

    # turn into a tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((boards, (policies, values)))
    
    # shuffle and turn dataset in batches
    train_dataset = train_dataset.shuffle(buffer_size=desired_steps * batch_size, reshuffle_each_iteration=True)
    
    # if training data is too small to get desired number of steps, repeat dataset (ie: equivalent to multiple epochs)
    train_dataset = train_dataset.repeat((desired_steps * batch_size) // n_samples + 1)
    
    # batch samples together
    train_dataset = train_dataset.batch(batch_size)
    
    # keep only the desired number of batches to achieve the target # of steps
    train_dataset = train_dataset.take(desired_steps)    
    
    print(f"returning {desired_steps:,} batches of size {batch_size}")
    
    return train_dataset


def load_train_save(next_gen: int, 
                    n_gens: int=10, 
                    steps: int=4000, 
                    batch_size: int=64, 
                    fname: str=None, **kwargs) -> (dict, PolicyValueNet):
    """train the next generation of PolicyValueNetwork, using historical board positions as input and mcts-policy estimations and average game outcome as targets"""
    # instantiate previous evaluator as starting point for retraining
    if not fname:
        fname = f'models/gen{next_gen-1}.h5'
        
    assert os.path.exists(fname), f"{fname} cannot be found. please verify the path you provided"
        
    pvn = PolicyValueNet(filename=fname, name=f'gen{next_gen}', quiet=True)
    
    # get training dataset
    train_dataset = get_training_dataset(next_gen=next_gen, n_gens=n_gens, desired_steps=steps, batch_size=batch_size)
    
    # using custom cross entropy loss for policy head, mse for value head 
    losses = [crossentropy_loss, 'mean_squared_error']
    
    # standard optimizer
    optimizer = tf.optimizers.Adam(lr=0.002)
    
    # compile prior to training
    pvn.model.compile(optimizer=optimizer, loss=losses)
    # train
    history = pvn.model.fit(train_dataset, **kwargs)
    
    pvn.save_model()
    print(f"model weights saved in models/gen{next_gen}.h5")
    
    return history.history, pvn
    
    

    
    