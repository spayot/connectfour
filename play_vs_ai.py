# -*- coding: utf-8 -*-
"""
Defines Actions and Game States for a connect four game.

TO DO:
- better comment and document the code
- replace with strategies and compete approach
"""
import argparse
import time
import os
import connectfour as c4


def display_state(node):
    os.system('clear')
    print(node.state)

def play_game(model_path: str, tau: float, n_sims: int) -> None:
    evaluator = c4.pvnet.PolicyValueNet(filename=model_path, quiet=True)
    print('evaluator:', evaluator.name)
    az_player = c4.player.AzPlayer(evaluator)

    state = c4.game.ConnectFourGameState(board=None, next_player=c4.game.Player.x)
    node = c4.mcts.MctsNode(state, evaluator)
    while not node.is_terminal_node():
        display_state(node)
        next_player = node.state.next_player
        
        if next_player == c4.game.Player.x:
            try:
                print('evaluator:', evaluator_policy.round(2), round(value, 3))
                print('with mcts:', mcts_policy.round(2))
            except Exception:
                pass
            col = input("your turn! what column do you want to play in [0-6]:")
            col = int(col)
            
            if col not in list(range(7)):
                col = input("improper value. choose a column between 0 and 6:")
            
            action = c4.game.ConnectFourAction(x_coordinate=col, player=next_player)
            
            if not node.state.is_move_legal(action):
                col = input("column is already full. choose another column:")
                action = c4.game.ConnectFourAction(x_coordinate=col, player=next_player)

            state = node.state.move(action)
            
            node = c4.mcts.MctsNode(state, evaluator)


        else:
            action, mcts_policy = az_player.play(node, tau=tau, n_sims=n_sims)
            evaluator_policy, value = evaluator.infer_from_state(node.state)
            

            node = action.take_action()

            # discard rest of tree
            node.parent = None
    
    os.system('clear')
    print(node.state)
    # assess result of the game
    result = node.state.game_result
    
    if result == c4.game.Player.x.value:
        print("congrats! you win!")
    elif result == c4.game.Player.o.value:
        print("good try! but you lost... game over")
    else:
        print('what a draw!')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modelpath",
                        default='models/gen9.h5',
                        help="the policy-value estimator to power AlphaZero",
                        type=str)
    parser.add_argument("-t", "--temperature", 
                        default=1, 
                        nargs='?', 
                        type=float, 
                        help="the higher the temperature, the more greedy the player")
    
    parser.add_argument("-n", "--n_simulations", 
                        default=100, 
                        type=int, 
                        help="the number of MCTS simulations to improve the raw evaluator's policy")

    args = parser.parse_args()
    
    play_game(model_path=args.modelpath, tau=args.temperature, n_sims=args.n_simulations)
    
    
    