import json
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from cattus_train.chess import Chess
from cattus_train.data_set import DataSet
from cattus_train.hex import Hex
from cattus_train.tictactoe import TicTacToe
from cattus_train.trainable_game import Game

TESTS_DIR = Path(__file__).parent.resolve()
SELF_PLAY_TOP = TESTS_DIR.parent / "self-play"

logging.basicConfig(level=logging.DEBUG, format="[Serialize Encode Test]: %(message)s")
logger = logging.getLogger(__name__)


def test_ttt_serialize_encode():
    _test_serialize_encode(
        "tictactoe",
        TicTacToe(),
        [
            "___x__o_ox",
            "o_xx_x__ox",
            "o__xo__xxx",
            "o___x_o__x",
            "oo__x____x",
            "oo__o__oxx",
        ],
    )


def test_hex_serialize_encode():
    _test_serialize_encode(
        "hex11",
        Hex(11),
        [
            (
                "reeeeeeeeee"
                "eeeeeeeeeee"
                "eeeeeeeeeee"
                "eeeeeeeeeee"
                "eeeeeeeeeee"
                "eeeeeeeeeee"
                "eeeeeeeeeee"
                "eeeeeeeeeee"
                "eeeeeeeeeee"
                "eeeeeeeeeee"
                "eeeeeeeeeee"
                "r"
            ),
            (
                "rererererer"
                "ererererere"
                "rererererer"
                "ererererere"
                "rererererer"
                "ererererere"
                "rererererer"
                "ererererere"
                "rererererer"
                "ererererere"
                "rererererer"
                "r"
            ),
            (
                "reeeeeeeeee"
                "ereeeeeeeee"
                "eereeeeeeee"
                "eeereeeeeee"
                "eeeereeeeee"
                "eeeeereeeee"
                "eeeeeereeee"
                "eeeeeeereee"
                "eeeeeeeeree"
                "eeeeeeeeere"
                "eeeeeeeeeer"
                "r"
            ),
        ],
    )


def test_chess_serialize_encode():
    _test_serialize_encode(
        "chess",
        Chess(),
        [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "nnqrkbbr/pppppppp/8/8/8/8/PPPPPPPP/NNQRKBBR w - - 0 1",
            "8/1p6/3QR3/6k1/1P2b3/2P3K1/6b1/6r1/ w - - 0 1",
            "4k2r/6r1/8/8/8/8/3R4/R3K3 w Qk - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1",
        ],
    )


def _test_serialize_encode(game_name: str, game: Game, positions):
    for position in positions:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            serialize_file = tmp_dir / "serialize_res.json"
            encode_file = tmp_dir / "encode_res.json"

            subprocess.check_call(
                [
                    "cargo",
                    "run",
                    "--profile=dev",
                    "-q",
                    "--features=onnx-ort",
                    "--bin=test_encode",
                    "--",
                    f"--game={game_name}",
                    f"--position={position}",
                    f"--outfile={encode_file}",
                ],
                cwd=SELF_PLAY_TOP,
            )
            subprocess.check_call(
                [
                    "cargo",
                    "run",
                    "--profile=dev",
                    "-q",
                    "--features=onnx-ort",
                    "--bin=test_serialize",
                    "--",
                    f"--game={game_name}",
                    f"--position={position}",
                    f"--outfile={serialize_file}",
                ],
                cwd=SELF_PLAY_TOP,
            )

            packed_entry = game.load_data_entry(serialize_file)
            tensors_entry = DataSet.unpack_planes(packed_entry, game)
            planes_py = tensors_entry.planes

            with open(encode_file, "r") as encode_file:
                rust_tensor_data = json.load(encode_file)

            planes_rust_shape = tuple(rust_tensor_data["shape"])
            if planes_rust_shape != planes_py.shape:
                raise ValueError("planes tensor shape mismatch", planes_rust_shape, planes_py.shape)

            planes_py = planes_py.flatten()
            planes_rust = np.array(rust_tensor_data["data"], dtype=np.float32)
            if (planes_rust != planes_py).any():
                raise ValueError(
                    f"Planes tensor mismatch."
                    f"(Game: {game_name}, Position: {position}, "
                    f"Rust planes: {planes_rust}, Data parser planes: {planes_py})"
                )


if __name__ == "__main__":
    test_ttt_serialize_encode()
    test_hex_serialize_encode()
    test_chess_serialize_encode()
    logger.info("test passed")
