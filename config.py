import chess

def make_all_moves_map():
    ALLMOVESMAP = {}
    k=0
    for i in range(0,64):
        for j in range(0,64):
            if (i!=j):
                ALLMOVESMAP[chess.Move(i,j).uci()] = k
                k+=1
    return ALLMOVESMAP

ALLMOVESMAP = make_all_moves_map()