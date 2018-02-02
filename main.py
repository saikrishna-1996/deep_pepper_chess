import numpy as np
import random

gamma = 0.9 #discount factor

def initialize():
    board = np.zeros((6,7)) #note that (0,0) corresponds on top left and (5,6) corresponds to bottom right
    return board

def did_white_win(board):
    for i in range(6):
        for j in range(7):
            if board[i,j] == 1:
                if i < 3 and j < 4:     #down right diagonal condition. not possible for i >=3 and j >= 4
                    if board[i+1,j+1] == 1 and board[i+2,j+2] == 1 and board[i+3,j+3] == 1:
                        return 1
                if j < 4: #right condition. not possible for j>=4
                    if board[i,j+1] == 1 and board[i,j+2] == 1 and board[i,j+3] == 1:
                        return 1
                if i < 3: #bottom condition.
                    if board[i+1,j] == 1 and board[i+2,j] == 1 and board[i+3,j] == 1:
                        return 1
                if i < 3 and j >= 3: #bottom left diagonal condition
                    if board[i+1,j-1] == 1 and board[i+2,j-2] == 1 and board[i+3,j-3] == 1:
                        return 1

    return 0

def did_black_win(board):
    for i in range(6):
        for j in range(7):
            if board[i,j] == -1:
                #print(i)
                #print(j)
                if i < 3 and j < 4:     #down right diagonal condition. not possible for i >=3 and j >= 4
                    if board[i+1,j+1] == -1 and board[i+2,j+2] == -1 and board[i+3,j+3] == -1:
                        return -1
                if j < 4: #right condition. not possible for j>=4
                    if board[i,j+1] == -1 and board[i,j+2] == -1 and board[i,j+3] == -1:
                        return -1
                if i < 3: #bottom condition.
                    if board[i+1,j] == -1 and board[i+2,j] == -1 and board[i+3,j] == -1:
                        return -1
                if i < 3 and j >= 3: #bottom left diagonal condition
                    if board[i+1,j-1] == -1 and board[i+2,j-2] == -1 and board[i+3,j-3] == -1:
                        return -1

    return 0


def mcts_thinker(board):
    mess_with_me = board.copy()
    num_simulations = 50
    #total_reward = 0
    best_reward = -100
    best_move = 0
    for j in range (7): #I have 7 possible moves

        total_reward = 0

        turn = -1 # 1 for white and -1 for black
        for lol in range(num_simulations):
            num_turns = 0
            mess_with_me = board.copy()
            if mess_with_me[0,j] != 0:
                break

            #mess_with_me = board.copy()
            if mess_with_me[0,j] == 0:
                for i in range(6):
                    if mess_with_me[5-i,j] == 0:
                        mess_with_me[5-i,j] == -1
                        turn = 1
                        break
            while(did_black_win(mess_with_me) == 0 or did_white_win(mess_with_me) == 0):
                num_turns = num_turns + 1
                col = random.randint(0,6)
                if turn == 1:

                    if mess_with_me[0,col] == 0:
                        for i in range(6):
                            if mess_with_me[5-i,col] == 0:
                                mess_with_me[5-i,col] = 1
                                break
                    if(did_white_win(mess_with_me) == 1):
                        total_reward = total_reward - 1.*(gamma**num_turns)
                        #print(lol)
                        print("white wins in simulation\n")
                        #print(mess_with_me)
                        break
                    else:
                        turn = -1

                else:
                    if mess_with_me[0,col] == 0:
                        for i in range(6):
                            if mess_with_me[5-i,col] == 0:
                                mess_with_me[5-i,col] = -1
                                break

                    if(did_black_win(mess_with_me) == -1):
                        total_reward = total_reward + 1.*(gamma**num_turns)
                        #print(lol)
                        print("Black wins in simulation\n")
                        #print(mess_with_me)
                        break
                    else:
                        turn = 1
        print(j)
        print(total_reward)
        print(best_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_move = j

        print("best move is %d\n",best_move)

    if board[0,best_move] != 0:
        print("Draw?\n")
    else:
        for i in range(6):
            if board[5-i,best_move] == 0:
                return 5-i, best_move
                #board[5-i,j] = -1
                #break


board = initialize()
while(did_black_win(board) == 0 or did_white_win(board) == 0):
    movestring = input('Enter Whites move:   ')
    move = int(movestring)      #just enter the column number as input
    if board[0, move] != 0:
        print("Invalid move\n")
    for lol in range(6):
        if board[5-lol,move] == 0:
            board[5-lol, move] = 1
            break

    print(board)
    if(did_white_win(board) == 1):
        print("White wins\n ")
        break

    row2, col2 = mcts_thinker(board)
    if board[row2,col2] == 0:
        board[row2,col2] = -1
    else:
        print("illegal move, piece already existant\n")
    print(board)
    if(did_black_win(board) == -1):
        print("Black wins\n ")
        break
