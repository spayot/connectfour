# -*- coding: utf-8 -*-
"""
Allows a human player to play against an AlphaZero Agent in the terminal.

Example:
--------

% python3 play_vs_ai.py --temperature 1 --n_simulations 100 --modelpath "models/gen9.h5"

"""
import argparse


import connectfour as c4

def _parse_args():
    """returns the parsed CLI arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modelpath",
                        default='models/gen9.h5',
                        help="the path to the policy-value estimator to power AlphaZero",
                        type=str)
    parser.add_argument("-t", "--temperature", 
                        default=1, 
                        nargs='?', 
                        type=float, 
                        help="the higher the temperature, the more greedily the agent selects the most promising moves.")
    
    parser.add_argument("-n", "--n_simulations", 
                        default=100, 
                        type=int, 
                        help="the number of MCTS simulations to improve the raw evaluator's policy")

    return parser.parse_args()
    
            
    
def main(model_path, temperature, n_sims) -> None:   
    
    runner = c4.human_vs_ai.runner.GameRunner(model_path=model_path, 
                                              temperature=temperature, 
                                              n_sims=n_sims)
    
    # define UI
    ui = c4.human_vs_ai.cli.ConnectFourUI(runner)
    
    # run game
    runner.play_vs_ai(ui)
            

if __name__ == '__main__':
    args = _parse_args()
    main(model_path=args.modelpath, 
         temperature=args.temperature, 
         n_sims=args.n_simulations)
    
    