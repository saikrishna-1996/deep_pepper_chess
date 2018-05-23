# Deep Pepper

MCTS-based algorithm for parallel training of a chess engine. Adapted from existing deep learning game engines such as Giraffe and AlphaZero, Deep Pepper is a clean-room implementation of a chess engine that leverages Stockfish for the opening and closing book, and learns a policy entirely through self-play.

## Technologies Used

We use the following technologies to train the model and interface with the Stockfish Chess engine.

* [python-chess](https://github.com/niklasf/python-chess) - For handling the chess environment and gameplay.
* [pytorch](https://github.com/pytorch/pytorch) - For training and inference.
* [Stockfish](https://github.com/official-stockfish/stockfish) - For value function and endgame evaluation.
* [Tensorboard](https://github.com/tensorflow/tensorboard) - For visualizing training progress.
 
## Setup Instructions

1. Run `pip install -r requirements.txt` to install the necessary dependencies.
2. Run `python launch_script.py` to start training the Chess Engine.

## Acknowledgements

* [Giraffe](https://arxiv.org/abs/1509.01549)
* [Alpha Zero](https://arxiv.org/pdf/1712.01815.pdf)
* [StockFish](https://stockfishchess.org)
