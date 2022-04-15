import os
from typing import TypedDict

from .. import game

class ValidUserInputFeedback(TypedDict):
    type: str
    message: str

    
def _assess_user_input(user_input: str, state: game.ConnectFourGameState) -> ValidUserInputFeedback:
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
    action = game.ConnectFourAction(x_coordinate=col, player=state.next_player)
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
        is_input_valid = _assess_user_input(user_input, self.runner.current_state)

        while is_input_valid['type'] == 'INVALID':
            # ask for new input with appropriate feedback message
            user_input = input(is_input_valid['message'])

            # revalidate the output
            is_input_valid = _assess_user_input(user_input, self.runner.current_state)

        return int(user_input) 
    
    
    def display_final_message(self) -> None:
        self.display_current_state()
        final_message = {
            game.Player.x.value: f"congrats! you won in {self.runner.turns} turns!",
            game.Player.o.value: f"good try! but you lost in {self.runner.turns} turns... game over",
            0: 'what a draw! good game.'
        }
        print(final_message[self.runner.result]) 