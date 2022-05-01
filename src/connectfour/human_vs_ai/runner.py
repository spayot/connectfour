"""
defines the GameRunner class that handles the game logic for a human vs Player game.

"""

from typing import Optional

import numpy as np

from .. import game, player, pvnet
from .. import temperature as temp
from . import cli


class GameRunner:
    def __init__(self, model_path: str, temperature: float, n_sims: int) -> None:
        """interface to play a game against an AlphaZero-like agent.

        Args:
            model_path: allows to select which generation to play against
            temperature: the temperature defining how greedy the AI agent will be
            n_sims: the number of MCTS simulations the agent will perform to
                improve on its raw evaluator policy. (the higher, the better the agent)

        """

        # instantiate evaluator
        evaluator = pvnet.PolicyValueNet.from_file(filename=model_path)

        # instantiate the AlphaZero agent
        temperature_schedule = temp.TemperatureSchedule(temperature,0,temperature)
        
        self.agent = player.AzAgent(evaluator=evaluator, 
                                       temperature_schedule=temperature_schedule)
        
        self.n_sims = n_sims
    
    def initialize_game(self) -> None:
        """initialize a new game (both state and agent representation)
        
        Args:
            None
        
        Returns:
            None"""
        self.current_state = game.ConnectFourGameState()
        self.agent.initialize_game() # initialize internal representation

        self.turns: int = 0 # a counter of the number of moves so far

        self.mcts_policy: Optional[np.ndarray] = None
    
    def update(self, action: game.ConnectFourAction) -> None:
        """updates game state and Agent representation of the game.
        
        Args:
            action: the action selected by the player whose turn it is
            
        Returns:
            None
        """
        # update state    
        self.current_state = self.current_state.move(action)
        self.agent.update(action)    
        self.turns += 1
        
    def play_vs_ai(self, ui: cli.ConnectFourUI) -> None:
        """runs a full ConnectFourGame where AzAgent is playing against user.
        
        Args:
            ui: the ConnectFourUI that defines the implementation of anything UI
            related.
            
        Returns:
            None"""
        ui.display_introduction()
        
        
        self.initialize_game()
        
        while not self.current_state.is_game_over:

            # render current state
            ui.display_current_state()
            
            # switch player turn
            next_player = self.current_state.next_player

            # human player turn
            if next_player == game.Player.x:

                # get user chosen action
                col = ui.get_user_input()

                action = game.ConnectFourAction(x_coordinate=col, player=next_player)

            else:
                action, self.mcts_policy = self.agent.select_next_action(self.n_sims)


            self.update(action)

        # assess result of the game 
        self.result = self.current_state.game_result
        ui.display_final_message()