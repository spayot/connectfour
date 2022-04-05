# -*- coding: utf-8 -*-
"""
Allows a human player to play against an AlphaZero Agent on the terminal.

% python3 play_vs_ai.py --temperature 1 --n_sims 100 --model_path "models/gen9.h5"

TO DO:
- replace with strategies and compete approach
"""
import argparse
import time
import os
import connectfour as c4


def display_state(node: c4.mcts.MctsNode) -> None:
    """defines how each game state should be displayed in the terminal.
    
    Args:
        node: the MctsNode associated with that game state."""
    os.system('clear')
    print(node.state)
    
    
def assess_user_move(user_input: str, state: c4.game.ConnectFourGameState) -> dict:
    """returns a dictionary with feedback on whether the input
    was valid or not. type: (VALID, INVALID, EXIT).
    if it is invalid, it also includes a feedback message"""
    
    # exit scenario
    if user_input.lower() == 'x':
        exit()
    
    # non-integer values
    if not user_input.isnumeric():
        return {'type': 'INVALID', 'message': 'you need to enter an integer between 0 and 6:  '}
    
    # invalid integer values
    col = int(user_input)
    if (col > 6) | (col < 0):
        return {'type': 'INVALID', 'message': 'you need to enter an integer between 0 and 6:  '}
    
    # checking for illegal moves (column already full)
    action = c4.game.ConnectFourAction(x_coordinate=col, player=state.next_player)
    if not state.is_move_legal(action):
        return {'type': 'INVALID', 'message': 'this column is already full. try again!  '}
    
    return {'type': 'VALID', 'message': None}
    
    
def get_user_input(state: c4.game.ConnectFourGameState) -> int:
    """allows to collect user input for the next move and verify the move is valid.
    allows user to enter a different value if the original input is invalid."""
    
    msg = """your turn! what column do you want to play in [0-6]? type X to exit:  """
    user_input = input(msg)
    is_input_valid = assess_user_move(user_input, state)
    
    while is_input_valid['type'] == 'INVALID':
        # ask for new input with appropriate feedback message
        user_input = input(is_input_valid['message'])
        
        # revalidate the output
        is_input_valid = assess_user_move(user_input, state)
    
    return int(user_input)        
    

    
def play_game(model_path: str, temperature: float, n_sims: int) -> None:
    """interface to play a game against an AlphaZero-like agent.
    
    Args:
        model_path: allows to select which generation to play against
        temperature: the temperature defining how greedy the AI agent will be
        n_sims: the number of MCTS simulations the agent will perform to
            improve on its raw evaluator policy.
            
    Returns:
        None"""
    
    # instantiate evaluator
    evaluator = c4.pvnet.PolicyValueNet.from_file(filename=model_path)
    print('evaluator:', evaluator.name)
    
    # instantiate the AlphaZero agent
    az_player = c4.player.AzPlayer(evaluator)
    
    # initialize the game
    state = c4.game.ConnectFourGameState(board=None, next_player=c4.game.Player.x)
    node = c4.mcts.MctsNode(state, evaluator)
    turns = 0 # a counter of the number of moves so far
    
    while not node.is_terminal_node():
        
        # render current state
        display_state(node)
        
        # display calculated policy when it follows AZ's turn
        if turns % 2 == 0 and turns > 0:
            print('raw evaluator:', evaluator_policy.round(2), f'\texpected value: {value:.3f}')
            print('with mcts:    ', mcts_policy.round(2))
        
        # switch player turn
        next_player = node.state.next_player
        
        # human player turn
        if next_player == c4.game.Player.x:
            
            # get user chosen action
            col = get_user_input(node.state)
            
            
            # note: this approach loses all the previous discoveries from the tree search.
            # better approach in the future: prune tree from all branches corresponding to 
            # actions not chosen by the human player, to inherit the existing tree.
            action = c4.game.ConnectFourAction(x_coordinate=col, player=next_player)
            
            state = node.state.move(action)
            
            node = c4.mcts.MctsNode(state, evaluator) # can be improved on to take further advantage of previous simulations

        # AlphaZero Agent turn
        else:
            # selects action
            chosen_action, mcts_policy = az_player.play_single_turn(node, tau=temperature, n_sims=n_sims)
            # this evaluation is not necessary to play the game, but simply to visualize the
            # raw policies 
            evaluator_policy, value = evaluator.infer_from_state(node.state)
            
            # execute action and prune tree branches related to alternative actions
            node = chosen_action.take_action(prune=True)
            
        turns += 1
    
    display_state(node)
    # assess result of the game
    result = node.state.game_result
    
    if result == c4.game.Player.x.value:
        print(f"congrats! you won in {turns} turns!")
    elif result == c4.game.Player.o.value:
        print(f"good try! but you lost in {turns} turns... game over")
    else:
        print('what a draw! good game.')




if __name__ == '__main__':
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

    args = parser.parse_args()
    
    play_game(model_path=args.modelpath, temperature=args.temperature, n_sims=args.n_simulations)
    
    
    