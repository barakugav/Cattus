

# Cattus
[![CI](https://github.com/poja/Cattus/actions/workflows/ci.yaml/badge.svg)](https://github.com/poja/Cattus/actions/workflows/ci.yaml)

Cattus is a chess engine based on DeepMind [AlphaZero paper](https://arxiv.org/abs/1712.01815), written in Rust. It uses a neural network to evaluate positions, and MCTS as a search algorithm.

The neural network is trained by self-play of the engine itself, in iterations. The initial network may perform badly on a state, but using the search algorithm on top of the network output, a more accurate evaluation of the state is achieved, which is then fed into the network as training data.

## UCI

The engine exectuable support the [Universal Chess Interface](https://en.wikipedia.org/wiki/Universal_Chess_Interface).

By using [python-chess](https://pypi.org/project/python-chess/0.15.0/) the engine can be used in Python:
```python
import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci(["cattus.exe", "--sim-num", "10000"])

# Let Cattus play against itself
board = chess.Board()
while not board.is_game_over() and not board.can_claim_draw():
	result = engine.play(board, chess.engine.Limit(time=20))
	board.push(result.move)

engine.quit()
```

## Training

To run the training process, first one should install the python project:
1. From a directory of the project
   ```
   pip install Cattus
   ```
2. Or straight from Github:
   ```
   pip install git+https://github.com/poja/Cattus.git
   ```


Then start the training process:
```bash
python training/bin/main.py --config training/config/chess_dev.yaml
```
