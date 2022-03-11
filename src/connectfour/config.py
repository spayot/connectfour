from dataclasses import dataclass


# board config
board_config = {'width': 7,
               'height': 6}


# policy-value network
pvn_config = {'block_size': 1,
              'l2_const': 1e-4,
             }

training_config = {}


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

