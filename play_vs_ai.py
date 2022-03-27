import time
import os
import connectfour as c4


def display_state(node):
    os.system('clear')
    print(node.state)

def play_game() -> None:
    evaluator = c4.pvnet.PolicyValueNet(filename='models/gen9.h5', quiet=True)
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
            action, mcts_policy = az_player.play(node, tau=.1, n_sims=100)
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
    play_game()