# -*- coding: utf-8 -*-
"""
Allows a human player to play against an AlphaZero Agent in the terminal.

% python3 play_vs_ai.py --temperature 1 --n_simulations 100 --modelpath "models/gen9.h5"

"""
import argparse
import os
from typing import TypedDict

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
    

class ValidUserInputFeedback(TypedDict):
    type: str
    message: str

    
def _assess_user_move(user_input: str, state: c4.game.ConnectFourGameState) -> ValidUserInputFeedback:
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
    
    
    
class ConnectFourUI:
    """User Interface allowing the game logic to interact with the user,
    both in terms of getting inputs (row to play)
    
    Args:
        runner: GameRunner"""
    def __init__(self, runner):
        self.runner = runner
    
    def display_introduction(self) -> None:
        print("welcome to ConnectFour vs AlphaZero!")
        print('you will play against agent:', self.runner.agent)
        
        
    def display_current_state(self) -> None:
        """defines how each game state should be displayed in the terminal.
    
        Args:
            runner: a GameRunner instance"""
        os.system('clear')
        print(self.runner.current_state)
        if self.runner.turns % 2 == 0 and self.runner.turns > 0:
            print('with mcts:    ', self.runner.mcts_policy.round(2))
        
        
    def get_user_input(self) -> int:
        """allows to collect user input for the next move and verify the move is valid.
        allows user to enter a different value if the original input is invalid."""

        msg = """your turn! what column do you want to play in [0-6]? type X to exit:  """
        user_input = input(msg)
        is_input_valid = _assess_user_move(user_input, self.runner.current_state)

        while is_input_valid['type'] == 'INVALID':
            # ask for new input with appropriate feedback message
            user_input = input(is_input_valid['message'])

            # revalidate the output
            is_input_valid = _assess_user_move(user_input, self.runner.current_state)

        return int(user_input) 
    
    
    def display_final_message(self) -> None:
        self.display_current_state()
        final_message = {
            c4.game.Player.x.value: f"congrats! you won in {self.runner.turns} turns!",
            c4.game.Player.o.value: f"good try! but you lost in {self.runner.turns} turns... game over",
            0: 'what a draw! good game.'
        }
        print(final_message[self.runner.result]) 
            
    
        
class GameRunner:
    def __init__(self, model_path: str, temperature: float, n_sims: int) -> None:
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

        # instantiate the AlphaZero agent
        temperature_schedule = c4.temperature.TemperatureSchedule(temperature,0,temperature)
        
        self.agent = c4.player.AzAgent(evaluator=evaluator, 
                                       temperature_schedule=temperature_schedule)
        
        self.n_sims = n_sims
    
    def initialize_game(self):
        """initialize a new game (both state and agent representation)"""
        self.current_state = c4.game.ConnectFourGameState()
        self.agent.initialize_game() # initialize internal representation

        self.turns: int = 0 # a counter of the number of moves so far

        self.mcts_policy = None
    
    def update(self, action: c4.game.ConnectFourAction):
        """updates game state and Agent representation of the game"""
        # update state    
        self.current_state = self.current_state.move(action)
        self.agent.update(action)    
        self.turns += 1
        
    def play_vs_ai(self, ui: ConnectFourUI):
        """runs a full ConnectFourGame where AzAgent is playing against user.
        
        Args:
            ui: the ConnectFourUI that defines the implementation of anything UI
            related."""
        ui.display_agent()
        
        
        self.initialize_game()
        
        while not self.current_state.is_game_over:

            # render current state
            ui.display_current_state()
            
            # switch player turn
            next_player = self.current_state.next_player

            # human player turn
            if next_player == c4.game.Player.x:

                # get user chosen action
                col = ui.get_user_input()

                action = c4.game.ConnectFourAction(x_coordinate=col, player=next_player)

            else:
                action, self.mcts_policy = self.agent.select_next_action(self.n_sims)


            self.update(action)

        # assess result of the game
        self.result = self.current_state.game_result
        ui.display_final_message()
            

if __name__ == '__main__':
    args = _parse_args()
    runner = GameRunner(model_path=args.modelpath, 
                        temperature=args.temperature, 
                        n_sims=args.n_simulations)
    
    # define UI
    ui = ConnectFourUI(runner)
    
    # run game
    runner.play_vs_ai(ui)
    