
python3 pgn2boards.py --player-file 'miniature.pgn' &&

for index in $(seq 0 10 1000000);#
do
python3 board_labeling.py --board-index $index --dump-path '/u/gottipav/deep_pepper_chess/game/' --player-file 'miniature.pgn'
echo $index
done
&&
python3 joining_boards.py --load-path '/u/gottipav/deep_pepper_chess/game/'
