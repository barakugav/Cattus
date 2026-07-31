import json
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import chess
import chess.engine
from cattus_train.config import MctsConfig, OnnxOrtConfig

TESTS_DIR = Path(__file__).parent.resolve()
CATTUS_ENGINE_TOP = TESTS_DIR.parent.parent / "engine"


def test_works_with_python_library_chess():
    subprocess.check_call(
        ["cargo", "build", "--features=onnx-ort,stockfish", "--bin=cattus", "-q", "--profile=release"],
        cwd=CATTUS_ENGINE_TOP,
    )
    cattus_exe = CATTUS_ENGINE_TOP / "target" / "release" / "cattus"
    assert cattus_exe.exists()

    engine_cfg = {
        "mcts": asdict(
            MctsConfig(
                sim_num=100,
                explore_factor=1.41421,
                temperature_policy=[(9999, 1.0)],
                prior_noise_alpha=0.03,
                prior_noise_epsilon=0.25,
                cache_size=1000000,
            )
        ),
        "model": {
            "model_path": "None",
            "inference": asdict(OnnxOrtConfig()),
            "batch_size": 1,
        },
        "threads": 1,
    }

    with tempfile.TemporaryDirectory() as tempdir:
        config_path = Path(tempdir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(engine_cfg, f, indent=2)

        engine = chess.engine.SimpleEngine.popen_uci(
            [str(cattus_exe), f"--config-file={config_path}"],
            cwd=CATTUS_ENGINE_TOP,
        )

        board = chess.Board()
        while not board.is_game_over() and not board.can_claim_draw():
            result = engine.play(board, chess.engine.Limit(time=20))
            board.push(result.move)

            for r in reversed(range(8)):
                for f in range(8):
                    sq = chess.square(f, r)
                    piece = board.piece_at(sq)
                    s = "·" if piece is None else str(piece)
                    print(s, " ", end="")
                print()
            print()
            print()

        engine.quit()


if __name__ == "__main__":
    test_works_with_python_library_chess()
