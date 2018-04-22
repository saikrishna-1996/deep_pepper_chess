
python3 pgn2boards.py --player-file 'Karpov.pgn'  &&

for index in $(seq 0 10 1000000);#
do
python3 board_labeling.py --board-index $index --dump-path '/media/ahmad/ahmad/topalov_files/' --player-file 'Karpov.pgn'
echo $index
done
&&
python3 joining_boards.py --load-path '/media/ahmad/ahmad/Karpov_files/'
