import numpy as np
import pytest

from connectfour import game


class TestPlayer:
    def test_get_next(self):
        player = game.Player.x
        assert player.get_next() == game.Player.o

class TestConnectFourBoard:
    def test_default_init(self):
        board = game.ConnectFourBoard()
        assert board.shape == (6,7), "default initialization should lead to shape (6,7)"

    def test_custom_init_shape(self):
        shape = (12,15)
        board = game.ConnectFourBoard(shape)
        assert board.shape == (12,15), "board shape is not updating as expected"
    
    def test_from_array_constructor_happy_path(self):
        array = np.array([[0,1,0,0],[0,-1,0,1],[1,-1,1,-1]])
        board = game.ConnectFourBoard.from_array(array)
        assert board.shape == array.shape, "from_array creates boards with incorrect shape"

    def test_from_array_constructor_invalid_Values(self):
        with pytest.raises(AssertionError):
            array = np.array([[0,1,0,0],[0,-1,0,1],[1,-1,1,-2]])
            board = game.ConnectFourBoard.from_array(array)
            
    def test_board_updates(self):
        array = np.array([[0,1,0,0],[0,-1,0,1],[1,-1,1,-1]])
        board = game.ConnectFourBoard.from_array(array)
        board[0,0] = 1
        assert board.array[0,0] == 1
        
    def test_copy(self):
        """verifies that changing the copy of a board does not impact
        the original board"""
        board = game.ConnectFourBoard()
        board2 = board.copy()
        board2[1,1] == 1
        assert board[1,1] == 0
        
    def test_repr(self):
        board = game.ConnectFourBoard.from_array(np.array([[0,0,1],[1,-1,-1]]))
        assert str(board) == 'ConnectFourBoard\n  .  .  X\n  X  O  O'
        
        
def generate_test_state(played_positions: list[int]) -> game.ConnectFourGameState:
    """allows to generate a state, based on a succession of moves."""
    board = game.ConnectFourBoard()
    state = game.ConnectFourGameState(board, next_player=game.Player.x)
    for position in played_positions:
        action = game.ConnectFourAction(position, state.next_player)
        state = state.move(action)
    return state


class TestConnectFourGame:
    def test_repr(self):
        state = generate_test_state([1,2])
        assert str(state) == """ConnectFourGameState(board:\n\tConnectFourBoard\n  .  .  .  .  .  .  .\n  .  .  .  .  .  .  .\n  .  .  .  .  .  .  .\n  .  .  .  .  .  .  .\n  .  .  .  .  .  .  .\n  .  X  O  .  .  .  .\n\tnext_player=x"""
        
    def test_horizontal_win(self):
        state = generate_test_state([1,2,1,2,1,5,1])
        assert state.game_result == 1
        assert state.is_game_over()
        
    def test_vertical_win(self):
        state = generate_test_state([0,0,1,1,3,2,5,2,5,3])
        assert state.game_result == -1
        assert state.is_game_over(), "vertical win should lead to a game over status"
        
    def test_diagonal_win(self):
        state = generate_test_state([0,1,1,2,2,3,2,3,3,5,3])
        assert state.game_result == 1, "diagonal wins are not properly identified"
        assert state.is_game_over()
        
    def test_game_is_not_over(self):
        state = generate_test_state([2,3,4,5])
        assert ~state.is_game_over(), f"state.is_game_over() should be False for state:{state}"
        
    def test_move_is_not_legal_top(self):
        state = generate_test_state([1,1,1,1,1,1])
        action = game.ConnectFourAction(1, game.Player.x)
        assert not state.is_move_legal(action)
        
    def test_is_move_legal_should_be_true(self):
        state = generate_test_state([1,1])
        action = game.ConnectFourAction(1, game.Player.x)
        assert state.is_move_legal(action)
        
    def test_move_is_not_legal_wrong_player(self):
        state = generate_test_state([1,1])
        action = game.ConnectFourAction(1, game.Player.o)
        assert not state.is_move_legal(action)
        
    def test_get_legal_actions(self):
        state = generate_test_state([1,1,1,1,1,1])
        legal_actions = state.get_legal_actions()
        coords = [action.x_coordinate for action in legal_actions]
        expected_coords = [0,2,3,4,5,6]
        assert coords == expected_coords, f"for state {state} get_legal_actions should return {expected_coords}, not {coords}"
        
