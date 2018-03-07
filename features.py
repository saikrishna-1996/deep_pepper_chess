import chess
import chess.uci
import numpy as np

def get_north_mobility(board, pos):
    cpos = int(pos)
    bean = 0
    cpos = cpos + 8
    while(cpos <= 63):
        #cpos = cpos + 8
        if(board.piece_type_at(cpos) == 0):
            bean = bean + 1
        else:
            break
        cpos = cpos + 8
    return bean

def get_south_mobility(board, pos):
    cpos = int(pos)
    bean = 0
    cpos = cpos - 8
    while(cpos >= 0):
        #cpos = cpos - 8
        if(board.piece_type_at(cpos) == 0):
            bean = bean + 1
        else:
            break
        cpos = cpos - 8
    return bean

def get_east_mobility(board, pos):
    cpos = int(pos)
    row = int(int(cpos)/int(8))
    col = cpos%8
    bean = 0
    col = col + 1
    while(col < 8):
        #col = col + 1
        cpos = row*8 + col
        if(board.piece_type_at(cpos) == 0):
            bean = bean + 1
        else:
            break
        col = col + 1
    return bean

def get_west_mobility(board, pos):
    cpos = int(pos)
    row = int(int(cpos)/int(8))
    col = cpos%8
    bean = 0
    col = col - 1
    while(col >= 0):
        #col = col-1
        cpos = row*8 + col
        if(board.piece_type_at(cpos) == 0):
            bean = bean + 1
        else:
            break
        col = col - 1
    return bean

def get_north_east_mobility(board, pos):
    #print(pos)
    cpos = int(pos)
    row = int(int(cpos)/int(8))
    #print(row)
    col = cpos%8
    bean = 0
    row = row + 1
    col = col + 1
    while(col < 8 & row < 8):
        #row = row + 1
        #col = col + 1
        cpos = row*8 + col
        #print(cpos)
        if(board.piece_type_at(cpos) == 0):
            bean = bean + 1
        else:
            break
        row = row + 1
        col = col + 1
    return bean

def get_south_east_mobility(board, pos):
    cpos = int(pos)
    row = int(int(cpos)/int(8))
    col = cpos%8
    bean = 0
    row = row - 1
    col = col + 1
    while(col < 8 & row >= 0):
        #row = row - 1
        #col = col + 1
        cpos = row*8 + col
        if(board.piece_type_at(cpos) == 0):
            bean = bean + 1
        else:
            break
        row = row - 1
        col = col + 1
    return bean

def get_south_west_mobility(board, pos):
    cpos = int(pos)
    row = int(int(cpos)/int(8))
    col = cpos%8
    bean = 0
    row = row - 1
    col = col - 1
    while(col >= 0 & row >= 0):
        #row = row - 1
        #col = col - 1
        cpos = row*8 + col
        if(board.piece_type_at(cpos) == 0):
            bean = bean + 1
        else:
            break
        row = row - 1
        col = col - 1
    return bean

def get_north_west_mobility(board, pos):
    cpos = int(pos)
    row = int(int(cpos)/int(8))
    col = cpos%8
    bean = 0
    row = row + 1
    col = col - 1
    while(col >= 0 & row < 8):
        #row = row + 1
        #col = col - 1
        cpos = row*8 + col
        print(bean, row, col, cpos)
        if(board.piece_type_at(cpos) == 0):
            bean = bean + 1
        else:
            break
        row = row + 1
        col = col - 1
    return bean



def BoardToFeature(board):

    feature = np.zeros(384)
    "START: GLOBAL FEATURES"

    #side-to-move
    feature[0] = board.turn

    #castling rights
    feature[1] = board.has_kingside_castling_rights(chess.WHITE)
    feature[2] = board.has_queenside_castling_rights(chess.WHITE)
    feature[3] = board.has_kingside_castling_rights(chess.BLACK)
    feature[4] = board.has_queenside_castling_rights(chess.BLACK)

    #material configuration
    feature[5] = len(board.pieces(chess.KING, chess.WHITE))
    feature[6] = len(board.pieces(chess.QUEEN, chess.WHITE))
    feature[7] = len(board.pieces(chess.ROOK, chess.WHITE))
    feature[8] = len(board.pieces(chess.BISHOP, chess.WHITE))
    feature[9] = len(board.pieces(chess.KNIGHT, chess.WHITE))
    feature[10] = len(board.pieces(chess.PAWN, chess.WHITE))
    feature[11] = len(board.pieces(chess.KING, chess.BLACK))
    feature[12] = len(board.pieces(chess.QUEEN, chess.BLACK))
    feature[13] = len(board.pieces(chess.ROOK, chess.BLACK))
    feature[14] = len(board.pieces(chess.BISHOP, chess.BLACK))
    feature[15] = len(board.pieces(chess.KNIGHT, chess.BLACK))
    feature[16] = len(board.pieces(chess.PAWN, chess.BLACK))

    "START: PIECE LIST"
    #should probably take extra care for bishops

    #white king
    feature[17] = 1
    wk_squares = board.pieces(chess.KING, chess.WHITE)
    pos = list(wk_squares)[0]
    row = pos/8
    col = pos%8
    feature[18] = float(row)/8.0
    feature[19] = float(col)/8.0

    white_attack = list(board.attackers(chess.WHITE, pos))
    min_val = 0
    for lol in range(len(white_attack)):
        curr_val = board.piece_type_at(white_attack[lol])
        if curr_val < min_val:
            min_val = curr_val
    feature[20] = min_val

    black_attack = list(board.attackers(chess.BLACK, pos))
    min_val = 0
    for lol in range(len(black_attack)):
        curr_val = board.piece_type_at(black_attack[lol])
        if curr_val < min_val:
            min_val = curr_val
    feature[21] = min_val

    #white queen
    if feature[6] < 1:
        feature[22] = 0
        feature[177:184] = 0
    else:
        feature[22] = 1
        wq_squares = board.pieces(chess.KING, chess.WHITE)
        pos = list(wq_squares)[0]
        row = pos/8
        col = pos%8

        #since you got the queen pose, add the mobility of queen
        feature[177] = get_north_mobility(board, pos)
        feature[178] = get_north_east_mobility(board, pos)
        feature[179] = get_east_mobility(board, pos)
        feature[180] = get_south_east_mobility(board, pos)
        feature[181] = get_south_mobility(board, pos)
        feature[182] = get_south_west_mobility(board, pos)
        feature[183] = get_west_mobility(board, pos)
        feature[184] = get_north_west_mobility(board, pos)


        feature[23] = float(row)/8.0
        feature[24] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[25] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[26] = min_val

    #white rooks
    wr_squares = list(board.pieces(chess.ROOK, chess.WHITE))
    wr_num = len(wr_squares)
    if wr_num == 0:
        feature[27:36] = 0
        feature[185:192] = 0
    elif wr_num == 1:
        feature[32:36] = 0
        feature[189:192] = 0
        feature[27] = 1
        pos = wr_squares[0]
        feature[185] = get_north_mobility(board, pos)
        feature[186] = get_east_mobility(board, pos)
        feature[187] = get_south_mobility(board, pos)
        feature[188] = get_west_mobility(board, pos)
        row = pos/8
        col = pos%8
        feature[28] = float(row)/8.0
        feature[29] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
    	    curr_val = board.piece_type_at(white_attack[lol])
    	    if curr_val < min_val:
    	        min_val = curr_val
        feature[30] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[31] = min_val

    else:
        feature[27] = 1
        pos = wr_squares[0]
        feature[185] = get_north_mobility(board, pos)
        feature[186] = get_east_mobility(board, pos)
        feature[187] = get_south_mobility(board, pos)
        feature[188] = get_west_mobility(board, pos)
        row = pos/8
        col = pos%8
        feature[28] = float(row)/8.0
        feature[29] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[30] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[31] = min_val

        feature[32] = 1
        pos = wr_squares[1]
        feature[189] = get_north_mobility(board, pos)
        feature[190] = get_east_mobility(board, pos)
        feature[191] = get_south_mobility(board, pos)
        feature[192] = get_west_mobility(board, pos)
        row = pos/8
        col = pos%8
        feature[33] = float(row)/8.0
        feature[34] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[35] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[36] = min_val


    #white bishops
    wb_squares = list(board.pieces(chess.BISHOP, chess.WHITE))
    wb_num = len(wb_squares)
    if wb_num == 0:
        feature[37:46] = 0
        feature[193:200] = 0
    elif wb_num == 1:
        feature[42:46] = 0
        feature[197:200] = 0
        feature[37] = 1
        pos = wb_squares[0]
        feature[193] = get_north_east_mobility(board, pos)
        feature[194] = get_south_east_mobility(board, pos)
        feature[195] = get_south_west_mobility(board, pos)
        feature[196] = get_north_west_mobility(board, pos)
        row = pos/8
        col = pos%8
        feature[38] = float(row)/8.0
        feature[39] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[40] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[41] = min_val

    else:
        feature[37] = 1
        pos = wb_squares[0]
        feature[193] = get_north_east_mobility(board, pos)
        feature[194] = get_south_east_mobility(board, pos)
        feature[195] = get_south_west_mobility(board, pos)
        feature[196] = get_north_west_mobility(board, pos)
        row = pos/8
        col = pos%8
        feature[38] = float(row)/8.0
        feature[39] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[40] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[41] = min_val

        feature[42] = 1
        pos = wb_squares[1]
        feature[197] = get_north_east_mobility(board, pos)
        feature[198] = get_south_east_mobility(board, pos)
        feature[199] = get_south_west_mobility(board, pos)
        feature[200] = get_north_west_mobility(board, pos)
        row = pos/8
        col = pos%8
        feature[43] = float(row)/8.0
        feature[44] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[45] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[46] = min_val


    #white knights
    wn_squares = list(board.pieces(chess.KNIGHT, chess.WHITE))
    wn_num = len(wn_squares)
    if wn_num == 0:
        feature[47:56] = 0
    elif wn_num == 1:
        feature[52:56] = 0
        feature[47] = 1
        pos = wn_squares[0]
        row = pos/8
        col = pos%8
        feature[48] = float(row)/8.0
        feature[49] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[50] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[51] = min_val

    else:
        feature[47] = 1
        pos = wn_squares[0]
        row = pos/8
        col = pos%8
        feature[48] = float(row)/8.0
        feature[49] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[50] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[51] = min_val

        feature[52] = 1
        pos = wn_squares[1]
        row = pos/8
        col = pos%8
        feature[53] = float(row)/8.0
        feature[54] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[55] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[56] = min_val


    #white pawns
    wp_squares = list(board.pieces(chess.PAWN, chess.WHITE))
    num_wp = len(wp_squares)
    fc = 57
    for lol in range(8):
        if lol > num_wp:
            feature[fc:fc+5] = 0
        else:
            feature[fc] = 1
            pos = wp_squares[lol]
            row = pos/8
            col = pos%8
            feature[fc+1] = float(row)/8.0
            feature[fc+2] = float(col)/8.0

            white_attack = list(board.attackers(chess.WHITE, pos))
            min_val = 0
            for lol2 in range(len(white_attack)):
                curr_val = board.piece_type_at(white_attack[lol2])
                if curr_val < min_val:
                    min_val = curr_val
            feature[fc+3] = min_val

            black_attack = list(board.attackers(chess.BLACK, pos))
            min_val = 0
            for lol2 in range(len(black_attack)):
                curr_val = board.piece_type_at(black_attack[lol2])
                if curr_val < min_val:
                    min_val = curr_val
            feature[fc+4] = min_val

        fc = fc+5


    #black king
    feature[97] = 1
    bk_squares = board.pieces(chess.KING, chess.BLACK)
    pos = list(bk_squares)[0]
    row = pos/8
    col = pos%8
    feature[98] = float(row)/8.0
    feature[99] = float(col)/8.0

    white_attack = list(board.attackers(chess.WHITE, pos))
    min_val = 0
    for lol in range(len(white_attack)):
        curr_val = board.piece_type_at(white_attack[lol])
        if curr_val < min_val:
            min_val = curr_val
    feature[100] = min_val

    black_attack = list(board.attackers(chess.BLACK, pos))
    min_val = 0
    for lol in range(len(black_attack)):
        curr_val = board.piece_type_at(black_attack[lol])
        if curr_val < min_val:
            min_val = curr_val
    feature[101] = min_val

    #black queen
    if feature[12] == 0:  #check this
        feature[102] = 0
        feature[201:208] = 0
    else:
        feature[102] = 1
        bq_squares = board.pieces(chess.QUEEN, chess.BLACK)
        pos = list(bq_squares)[0]
        feature[201] = get_north_mobility(board, pos)
        feature[202] = get_north_east_mobility(board, pos)
        feature[203] = get_east_mobility(board, pos)
        feature[204] = get_south_east_mobility(board, pos)
        feature[205] = get_south_mobility(board, pos)
        feature[206] = get_south_west_mobility(board, pos)
        feature[207] = get_west_mobility(board, pos)
        feature[208] = get_north_west_mobility(board, pos)

        row = pos/8
        col = pos%8
        feature[103] = float(row)/8.0
        feature[104] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[105] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[106] = min_val


    #black rooks
    br_squares = list(board.pieces(chess.ROOK, chess.BLACK))
    br_num = len(br_squares)
    if br_num == 0:
        feature[107:116] = 0
        feature[209:216] = 0
    elif br_num == 1:
        feature[112:116] = 0
        feature[213:216] = 0
        feature[107] = 1
        pos = br_squares[0]
        feature[209] = get_north_mobility(board, pos)
        feature[210] = get_east_mobility(board, pos)
        feature[211] = get_south_mobility(board, pos)
        feature[212] = get_west_mobility(board, pos)

        row = pos/8
        col = pos%8
        feature[108] = float(row)/8.0
        feature[109] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[110] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[111] = min_val

    else:
        feature[107] = 1
        pos = br_squares[0]
        feature[209] = get_north_mobility(board, pos)
        feature[210] = get_east_mobility(board, pos)
        feature[211] = get_south_mobility(board, pos)
        feature[212] = get_west_mobility(board, pos)

        row = pos/8
        col = pos%8
        feature[108] = float(row)/8.0
        feature[109] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[110] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[111] = min_val

        feature[112] = 1
        pos = br_squares[1]
        feature[213] = get_north_mobility(board, pos)
        feature[214] = get_east_mobility(board, pos)
        feature[215] = get_south_mobility(board, pos)
        feature[216] = get_west_mobility(board, pos)

        row = pos/8
        col = pos%8
        feature[113] = float(row)/8.0
        feature[114] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[115] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[116] = min_val


    #black bishops
    bb_squares = list(board.pieces(chess.BISHOP, chess.BLACK))
    bb_num = len(bb_squares)
    if bb_num == 0:
        feature[117:126] = 0
        feature[217:224] = 0
    elif bb_num == 1:
        feature[122:126] = 0
        feature[221:224] = 0
        feature[117] = 1
        pos = bb_squares[0]
        feature[217] = get_north_east_mobility(board, pos)
        feature[218] = get_south_east_mobility(board, pos)
        feature[219] = get_south_west_mobility(board, pos)
        feature[220] = get_north_west_mobility(board, pos)

        row = pos/8
        col = pos%8
        feature[118] = float(row)/8.0
        feature[119] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
    	        min_val = curr_val
        feature[120] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[121] = min_val

    else:
        feature[117] = 1
        pos = bb_squares[0]
        feature[217] = get_north_east_mobility(board, pos)
        feature[218] = get_south_east_mobility(board, pos)
        feature[219] = get_south_west_mobility(board, pos)
        feature[220] = get_north_west_mobility(board, pos)

        row = pos/8
        col = pos%8
        feature[118] = float(row)/8.0
        feature[119] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
    	        min_val = curr_val
        feature[120] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[121] = min_val

        feature[122] = 1
        pos = bb_squares[1]
        feature[221] = get_north_east_mobility(board, pos)
        feature[222] = get_south_east_mobility(board, pos)
        feature[223] = get_south_west_mobility(board, pos)
        feature[224] = get_north_west_mobility(board, pos)

        row = pos/8
        col = pos%8
        feature[123] = float(row)/8.0
        feature[124] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[125] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[126] = min_val


    #black knights
    bn_squares = list(board.pieces(chess.KNIGHT, chess.BLACK))
    bn_num = len(bn_squares)
    if bn_num == 0:
        feature[127:136] = 0
    elif wn_num == 1:
        feature[132:136] = 0
        feature[127] = 1
        pos = bn_squares[0]
        row = pos/8
        col = pos%8
        feature[128] = float(row)/8.0
        feature[129] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
    	        min_val = curr_val
        feature[130] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[131] = min_val

    else:
        feature[127] = 1
        pos = bn_squares[0]
        row = pos/8
        col = pos%8
        feature[128] = float(row)/8.0
        feature[129] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[130] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[131] = min_val

        feature[132] = 1
        pos = bn_squares[1]
        row = pos/8
        col = pos%8
        feature[133] = float(row)/8.0
        feature[134] = float(col)/8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 0
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[135] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 0
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[136] = min_val


    #black pawns
    bp_squares = list(board.pieces(chess.PAWN, chess.BLACK))
    num_bp = len(bp_squares)
    fc = 137
    for lol in range(8):
        if lol > num_bp:
            feature[fc:fc+5] = 0
        else:
            feature[fc] = 1
            pos = bp_squares[lol]
            row = pos/8
            col = pos%8
            feature[fc+1] = float(row)/8.0
            feature[fc+2] = float(col)/8.0

            white_attack = list(board.attackers(chess.WHITE, pos))
            min_val = 0
            for lol2 in range(len(white_attack)):
                curr_val = board.piece_type_at(white_attack[lol2])
                if curr_val < min_val:
                    min_val = curr_val
            feature[fc+3] = min_val

            black_attack = list(board.attackers(chess.BLACK, pos))
            min_val = 0
            for lol2 in range(len(black_attack)):
                curr_val = board.piece_type_at(black_attack[lol2])
                if curr_val < min_val:
                    min_val = curr_val
            feature[fc+4] = min_val

        fc = fc+5


    "START: Attack and Defend Maps (square-centric features)"

    #White-attacker
    for i in range(63):
        shooters = list(board.attackers(chess.WHITE, i))
        min_val = 0
        for j in range(len(shooters)):
            curr_val = board.piece_type_at(shooters[j])
            if curr_val < min_val:
                min_val = curr_val
        feature[225+i] = min_val

    #Black-attacker
    for j in range(63):
        shooters = list(board.attackers(chess.BLACK, i))
        min_val = 0
        for j in range(len(shooters)):
            curr_val = board.piece_type_at(shooters[j])
            if curr_val < min_val:
                min_val = curr_val
        feature[289+i] = min_val

    #We should make sure that bishops are not messed up.


board = chess.Board()
featuress = BoardToFeature(board)
print(featuress)
