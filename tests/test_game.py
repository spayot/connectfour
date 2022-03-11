import sys

import numpy as np
import pytest

sys.path.append('src/')

from connectfour import game

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
        assert board[0,0] == 1