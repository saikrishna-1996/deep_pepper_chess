import numpy as np
import random

def initialize():
    board = np.zeros((6,7))
    return board

def did_ivana_win(board):
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

def did_sai_win(board):
    for i in range(6):
        for j in range(7):
            if board[i,j] == -1:
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
    total_reward = 0
    best_reward = -100
    best_move = 0
    for j in range (7): #I have 7 possible moves

        #if you can't make that move, you are lost. (incorrect logic?)
        if mess_with_me[0,j] != 0:
            total_reward = total_reward - 1;

        #if the move is possible, make the move
        else:
            for i in range(6):
                if mess_with_me[5-i,j] == 0:
                    mess_with_me[5-i,j] == -1

        #simulating randomly from now onwards
        turn = 1 # 1 for ivana and -1 for sai
        for lol in range(num_simulations):
            while(did_sai_win(mess_with_me) == 0 or did_ivana_win(mess_with_me) == 0):

                col = random.randint(0,6)
                if turn == 1:
                    if mess_with_me[0,col] != 0:
                        total_reward = total_reward + 1
                    else:
                        for i in range(6):
                            if mess_with_me[5-i,col] == 0:
                                mess_with_me[5-i,col] = 1
                                break
                    if(did_ivana_win(mess_with_me) == 1):
                        total_reward = total_reward - 1
                        break
                    else:
                        turn = -1

                else:
                    if mess_with_me[0,col] != 0:
                        total_reward = total_reward - 1
                    else:
                        for i in range(6):
                            if mess_with_me[5-i,col] == 0:
                                mess_with_me[5-i,col] = -1
                                break

                    if(did_sai_win(mess_with_me) == -1):
                        total_reward = total_reward + 1
                        break
                    else:
                        turn = 1

        if total_reward > best_reward:
            best_reward = total_reward
            best_move = j

    if board[0,best_move] != 0:
        print("Draw?\n")
    else:
        for i in range(6):
            if board[5-i,best_move] == 0:
                return 5-i, best_move
                #board[5-i,j] = -1
                #break


board = initialize()
while(did_sai_win(board) == 0 or did_ivana_win(board) == 0):
    movestring = input('Enter ivanas move:   ')
    move = int(movestring)
    col = move%10
    #print(col)
    row = (move-col)/10
    row = int(row)
    if board[row,col] == 0:
        board[row,col] = 1
    else:
        print("illegal move, piece already existant\n")
    print(board)
    if(did_ivana_win(board) == 1):
        print("Ivana wins\n ")
        break

    #movestring2 = input('Enter sais move:    ')
    #move2 = int(movestring2)
    #col2 = move2%10
    #row2 = (move2-col2)/10
    #row2 = int(row2)
    row2, col2 = mcts_thinker(board)
    if board[row2,col2] == 0:
        board[row2,col2] = -1
    else:
        print("illegal move, piece already existant\n")
    print(board)
    if(did_sai_win(board) == -1):
        print("Sai wins\n ")
        break
