from dataclasses import dataclass

import yaml

# board config
board_config = {'width': 7,
               'height': 6}


# policy-value network
pvn_config = {'block_size': 1,
              'l2_const': 1e-4,
             }

training_config : dict = {}


mcts_config = {'C_PUCT': 4,
              'sims': 100,
               'tau': 1,
              }

selfplay_config = {"tau_start": 1,
                "tau_end": .1,
                "threshold": 10}

@dataclass        
class ConnectFourGameConfig:
    shape: tuple[int,int] = (6, 7)
    n_to_win: int = 4


def load_from_yaml(fpath=str) -> dict:
    """loads game config from a yaml file"""
    with open(fpath, 'r') as f:
        config : dict = yaml.safe_load(f)
        
    return config
    
