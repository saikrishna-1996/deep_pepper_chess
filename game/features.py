import chess
import chess.uci
import numpy as np

# input: board
# output: feature representation of the board


# get_north_mobility(), get_south_mobility(), get_east_mobility(), get_west_mobility(), get_north_east_mobility(), get_south_east_mobility(), get_south_west_mobility(), get_north_west_mobility() gives
# the mobility of sliding pieces (queen or rook or bishop) in each of those directions. Mobility is defined as the number of squares the particular sliding piece under consideration can move in the
# above mentioned directions before encountering any piece.

def get_north_mobility(board, pos):
    pos_copy = int(pos)
    count_temp = 0
    pos_copy = pos_copy + 8
    while pos_copy <= 63:
        if board.piece_type_at(pos_copy) == 0:
            count_temp = count_temp + 1
        else:
            break
        pos_copy = pos_copy + 8
    return count_temp


def get_south_mobility(board, pos):
    pos_copy = int(pos)
    count_temp = 0
    pos_copy = pos_copy - 8
    while pos_copy >= 0:
        if board.piece_type_at(pos_copy) == 0:
            count_temp = count_temp + 1
        else:
            break
        pos_copy = pos_copy - 8
    return count_temp


def get_east_mobility(board, pos):
    pos_copy = int(pos)
    row = int(int(pos_copy) / int(8))
    col = pos_copy % 8
    count_temp = 0
    col = col + 1
    while col < 8:
        pos_copy = row * 8 + col
        if board.piece_type_at(pos_copy) == 0:
            count_temp = count_temp + 1
        else:
            break
        col = col + 1
    return count_temp


def get_west_mobility(board, pos):
    pos_copy = int(pos)
    row = int(int(pos_copy) / int(8))
    col = pos_copy % 8
    count_temp = 0
    col = col - 1
    while col >= 0:
        pos_copy = row * 8 + col
        if board.piece_type_at(pos_copy) == 0:
            count_temp = count_temp + 1
        else:
            break
        col = col - 1
    return count_temp


def get_north_east_mobility(board, pos):
    pos_copy = int(pos)
    row = int(int(pos_copy) / int(8))
    col = pos_copy % 8
    count_temp = 0
    row = row + 1
    col = col + 1
    while col < 8 and row < 8:
        pos_copy = row * 8 + col
        if board.piece_type_at(pos_copy) == 0:
            count_temp = count_temp + 1
        else:
            break
        row = row + 1
        col = col + 1
    return count_temp


def get_south_east_mobility(board, pos):
    pos_copy = int(pos)
    row = int(int(pos_copy) / int(8))
    col = pos_copy % 8
    count_temp = 0
    row = row - 1
    col = col + 1
    while col < 8 and row >= 0:
        pos_copy = row * 8 + col
        if board.piece_type_at(pos_copy) == 0:
            count_temp = count_temp + 1
        else:
            break
        row = row - 1
        col = col + 1
    return count_temp


def get_south_west_mobility(board, pos):
    pos_copy = int(pos)
    row = int(int(pos_copy) / int(8))
    col = pos_copy % 8
    count_temp = 0
    row = row - 1
    col = col - 1
    while col >= 0 and row >= 0:
        pos_copy = row * 8 + col
        if board.piece_type_at(pos_copy) == 0:
            count_temp = count_temp + 1
        else:
            break
        row = row - 1
        col = col - 1
    return count_temp


def get_north_west_mobility(board, pos):
    pos_copy = int(pos)
    row = int(int(pos_copy) / int(8))
    col = pos_copy % 8
    count_temp = 0
    row = row + 1
    col = col - 1
    while col >= 0 and row < 8:
        pos_copy = row * 8 + col
        if board.piece_type_at(pos_copy) == 0:
            count_temp = count_temp + 1
        else:
            break
        row = row + 1
        col = col - 1
    return count_temp


def board_to_feature(board):
    feature = np.zeros(384)
    "START: GLOBAL FEATURES"

    # side-to-move
    feature[0] = board.turn

    # castling rights (whether each side has kingside / queenside castling rights)
    feature[1] = board.has_kingside_castling_rights(chess.WHITE)
    feature[2] = board.has_queenside_castling_rights(chess.WHITE)
    feature[3] = board.has_kingside_castling_rights(chess.BLACK)
    feature[4] = board.has_queenside_castling_rights(chess.BLACK)

    # material configuration (number of pieces of each type present on the board)
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
    ## should probably take extra care for bishops

    ## In piece lists, we have 5 slots alloted for each of 1 king, 1 queen, 2 rooks, 2 bishops, 2 knights and 8 pawns for white and black.

    ## 17-176: For every piece, we determine whether the piece is existant (1 slot), normalized x-coordinate (1 slot), normalized y-coordinate (1 slot), lowest valued attacker of that piece (1 slot), and lowest valued defender of that piece (1 slot). So, we use 5 slots for every piece. And, there are a total of 32 pieces. (2 kings, 2 queens, 4 rooks, 4 bishops, 4 knights, 16 pawns). So, a total of 160 slots.

    ## 177-224: For each sliding piece, this encodes how far they can go in each direction. So 8 slots for a queen, 4 slots for a rook and 4 slots for a bishop. Since there are 2 queens, 4 rooks and 4 bishops, we need a total of 48 slots.

    ## For king, knights and pawns (which are classified as non-sliding pieces according to Giraffe), we don't encode how far they move in each direction. So, for determining the corresponding features for a non-sliding piece, we make a list of all the existing pieces of that particular type (for example, a maximum of 2 knights), and then for each of those pieces, we compute it's normalized x-coordinate and y-coordinate by using "float(row)/ 8.0", "float(col)/ 8.0" and then compute
    # the least valued attacker and defender of that square (where the piece is present) by looping over all the attackers and defenders respectively of that square.

    ## For queen, rooks and bishops (which are classified as sliding pieces according to Giraffe), we also encode how far they can move in their possible directions before they encounter an obstacle. All this is computed in the mobility functions defined at the beginning of this code. For example, a rook can move north or east or west or south. So, we compute it's mobility in all the 4 directions, where as a queen can move in 8 directions.


    # white king
    feature[17] = 1
    wk_squares = board.pieces(chess.KING, chess.WHITE)
    pos = list(wk_squares)[0]
    row = pos / 8
    col = pos % 8
    feature[18] = float(row) / 8.0
    feature[19] = float(col) / 8.0

    white_attack = list(board.attackers(chess.WHITE, pos))
    min_val = 10
    for lol in range(len(white_attack)):
        curr_val = board.piece_type_at(white_attack[lol])
        if curr_val < min_val:
            min_val = curr_val
    feature[20] = min_val

    black_attack = list(board.attackers(chess.BLACK, pos))
    min_val = 10
    for lol in range(len(black_attack)):
        curr_val = board.piece_type_at(black_attack[lol])
        if curr_val < min_val:
            min_val = curr_val
    feature[21] = min_val

    # white queen
    if feature[6] < 1:
        feature[22] = 0
        feature[177:184] = 0
    else:
        feature[22] = 1
        wq_squares = board.pieces(chess.KING, chess.WHITE)
        pos = list(wq_squares)[0]
        row = pos / 8
        col = pos % 8

        # since you got the queen pose, add the mobility of queen
        feature[177] = get_north_mobility(board, pos)
        feature[178] = get_north_east_mobility(board, pos)
        feature[179] = get_east_mobility(board, pos)
        feature[180] = get_south_east_mobility(board, pos)
        feature[181] = get_south_mobility(board, pos)
        feature[182] = get_south_west_mobility(board, pos)
        feature[183] = get_west_mobility(board, pos)
        feature[184] = get_north_west_mobility(board, pos)

        feature[23] = float(row) / 8.0
        feature[24] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[25] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[26] = min_val

    # white rooks
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
        row = pos / 8
        col = pos % 8
        feature[28] = float(row) / 8.0
        feature[29] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[30] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
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
        row = pos / 8
        col = pos % 8
        feature[28] = float(row) / 8.0
        feature[29] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[30] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
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
        row = pos / 8
        col = pos % 8
        feature[33] = float(row) / 8.0
        feature[34] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[35] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[36] = min_val

    # white bishops
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
        row = pos / 8
        col = pos % 8
        feature[38] = float(row) / 8.0
        feature[39] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[40] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
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
        row = pos / 8
        col = pos % 8
        feature[38] = float(row) / 8.0
        feature[39] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[40] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
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
        row = pos / 8
        col = pos % 8
        feature[43] = float(row) / 8.0
        feature[44] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[45] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[46] = min_val

    # white knights
    wn_squares = list(board.pieces(chess.KNIGHT, chess.WHITE))
    wn_num = len(wn_squares)
    if wn_num == 0:
        feature[47:56] = 0
    elif wn_num == 1:
        feature[52:56] = 0
        feature[47] = 1
        pos = wn_squares[0]
        row = pos / 8
        col = pos % 8
        feature[48] = float(row) / 8.0
        feature[49] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[50] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[51] = min_val

    else:
        feature[47] = 1
        pos = wn_squares[0]
        row = pos / 8
        col = pos % 8
        feature[48] = float(row) / 8.0
        feature[49] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[50] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[51] = min_val

        feature[52] = 1
        pos = wn_squares[1]
        row = pos / 8
        col = pos % 8
        feature[53] = float(row) / 8.0
        feature[54] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[55] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[56] = min_val

    # white pawns
    wp_squares = list(board.pieces(chess.PAWN, chess.WHITE))
    num_wp = len(wp_squares)
    fc = 57
    for lol in range(8):
        if lol > num_wp - 1:
            feature[fc:fc + 5] = 0
        else:
            feature[fc] = 1
            pos = wp_squares[lol]
            row = pos / 8
            col = pos % 8
            feature[fc + 1] = float(row) / 8.0
            feature[fc + 2] = float(col) / 8.0

            white_attack = list(board.attackers(chess.WHITE, pos))
            min_val = 10
            for lol2 in range(len(white_attack)):
                curr_val = board.piece_type_at(white_attack[lol2])
                if curr_val < min_val:
                    min_val = curr_val
            feature[fc + 3] = min_val

            black_attack = list(board.attackers(chess.BLACK, pos))
            min_val = 10
            for lol2 in range(len(black_attack)):
                curr_val = board.piece_type_at(black_attack[lol2])
                if curr_val < min_val:
                    min_val = curr_val
            feature[fc + 4] = min_val

        fc = fc + 5

    # black king
    feature[97] = 1
    bk_squares = board.pieces(chess.KING, chess.BLACK)
    pos = list(bk_squares)[0]
    row = pos / 8
    col = pos % 8
    feature[98] = float(row) / 8.0
    feature[99] = float(col) / 8.0

    white_attack = list(board.attackers(chess.WHITE, pos))
    min_val = 10
    for lol in range(len(white_attack)):
        curr_val = board.piece_type_at(white_attack[lol])
        if curr_val < min_val:
            min_val = curr_val
    feature[100] = min_val

    black_attack = list(board.attackers(chess.BLACK, pos))
    min_val = 10
    for lol in range(len(black_attack)):
        curr_val = board.piece_type_at(black_attack[lol])
        if curr_val < min_val:
            min_val = curr_val
    feature[101] = min_val

    # black queen
    if feature[12] == 0:  # check this
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

        row = pos / 8
        col = pos % 8
        feature[103] = float(row) / 8.0
        feature[104] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[105] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[106] = min_val

    # black rooks
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

        row = pos / 8
        col = pos % 8
        feature[108] = float(row) / 8.0
        feature[109] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[110] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
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

        row = pos / 8
        col = pos % 8
        feature[108] = float(row) / 8.0
        feature[109] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[110] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
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

        row = pos / 8
        col = pos % 8
        feature[113] = float(row) / 8.0
        feature[114] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[115] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[116] = min_val

    # black bishops
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

        row = pos / 8
        col = pos % 8
        feature[118] = float(row) / 8.0
        feature[119] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[120] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
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

        row = pos / 8
        col = pos % 8
        feature[118] = float(row) / 8.0
        feature[119] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[120] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
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

        row = pos / 8
        col = pos % 8
        feature[123] = float(row) / 8.0
        feature[124] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[125] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[126] = min_val

    # black knights
    bn_squares = list(board.pieces(chess.KNIGHT, chess.BLACK))
    bn_num = len(bn_squares)
    if bn_num == 0:
        feature[127:136] = 0
    elif bn_num == 1:
        feature[132:136] = 0
        feature[127] = 1
        pos = bn_squares[0]
        row = pos / 8
        col = pos % 8
        feature[128] = float(row) / 8.0
        feature[129] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[130] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[131] = min_val

    else:
        feature[127] = 1
        pos = bn_squares[0]
        row = pos / 8
        col = pos % 8
        feature[128] = float(row) / 8.0
        feature[129] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[130] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[131] = min_val

        feature[132] = 1
        pos = bn_squares[1]
        row = pos / 8
        col = pos % 8
        feature[133] = float(row) / 8.0
        feature[134] = float(col) / 8.0

        white_attack = list(board.attackers(chess.WHITE, pos))
        min_val = 10
        for lol in range(len(white_attack)):
            curr_val = board.piece_type_at(white_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[135] = min_val

        black_attack = list(board.attackers(chess.BLACK, pos))
        min_val = 10
        for lol in range(len(black_attack)):
            curr_val = board.piece_type_at(black_attack[lol])
            if curr_val < min_val:
                min_val = curr_val
        feature[136] = min_val

    # black pawns
    bp_squares = list(board.pieces(chess.PAWN, chess.BLACK))
    num_bp = len(bp_squares)
    fc = 137
    for lol in range(8):
        if lol > num_bp - 1:
            feature[fc:fc + 5] = 0
        else:
            feature[fc] = 1
            pos = bp_squares[lol]
            row = pos / 8
            col = pos % 8
            feature[fc + 1] = float(row) / 8.0
            feature[fc + 2] = float(col) / 8.0

            white_attack = list(board.attackers(chess.WHITE, pos))
            min_val = 10
            for lol2 in range(len(white_attack)):
                curr_val = board.piece_type_at(white_attack[lol2])
                if curr_val < min_val:
                    min_val = curr_val
            feature[fc + 3] = min_val

            black_attack = list(board.attackers(chess.BLACK, pos))
            min_val = 10
            for lol2 in range(len(black_attack)):
                curr_val = board.piece_type_at(black_attack[lol2])
                if curr_val < min_val:
                    min_val = curr_val
            feature[fc + 4] = min_val

        fc = fc + 5

    "START: Attack and Defend Maps (square-centric features)"
    ## For each square, we encode the values of the lowest valued attacker and lowest valued defender of that square. Since we have 64 squares, there are a total of 128 slots used here. So, in the slots 225-288, we will encode the least-valued white piece attacking each of the sqaures on the board. For the slots 289-352, we encode the least-valued black piece attacking each of the squares on the board.

    # White-attacker
    for i in range(63): # iterating through every square
        attackers_list = list(board.attackers(chess.WHITE, i)) # list of all white pieces attacking that particular square
        min_val = 10    # if there is no white piece attacking that square, we will store value of 10
        for j in range(len(attackers_list)):
            curr_val = board.piece_type_at(attackers_list[j])
            if curr_val < min_val:
                min_val = curr_val
        feature[225 + i] = min_val

    # Black-attacker
    for i in range(63): # iterating through every square
        attackers_list = list(board.attackers(chess.BLACK, i)) # list of all black pieces attacking that particular square
        min_val = 10    # if there is no black piece attacking that square, we will store value of 10
        for j in range(len(attackers_list)):
            curr_val = board.piece_type_at(attackers_list[j])
            if curr_val < min_val:
                min_val = curr_val
        feature[289 + i] = min_val

    ## Done with computing the feature vector.
    return feature
