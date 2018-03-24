import chess
import chess.polyglot

board = chess.Board()

with chess.polyglot.open_reader("./komodo_bin/strong/komodo.bin") as reader:
    for entry in reader.find_all(board):
        print(entry.move(), entry.weight, entry.learn)
