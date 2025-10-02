import argparse
import copy
import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import yaml
from cattus_train import Config
from cattus_train.chess import Chess
from cattus_train.config import (
    ExecutorchConfig,
    OnnxOrtConfig,
    OnnxTractConfig,
    TorchPyConfig,
    TorchTchRsConfig,
)
from cattus_train.self_play import compile_selfplay_exe, runtime_env

CATTUS_TOP = Path(__file__).parent.parent.parent.resolve()
SELF_PLAY_CRATE_TOP = CATTUS_TOP / "training" / "self-play"
GAME = Chess()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CATTUS_TOP / "training" / "config" / "chess_dev.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    cfg = Config(**yaml.safe_load(args.config.read_text()))
    cfg.engine.mcts.sim_num = 4
    cfg.engine.threads = 1
    cfg.engine.model.batch_size = 1
    cfg.self_play.games_num = 2

    inference_configs = [
        ExecutorchConfig(backend="none"),
        ExecutorchConfig(backend="xnnpack"),
        # ExecutorchConfig(backend="mps"),
        TorchPyConfig(),
        TorchTchRsConfig(),
        OnnxTractConfig(),
        OnnxOrtConfig(),
    ]

    summary = []
    for inference_config in inference_configs:
        with tempfile.TemporaryDirectory() as tempdir:
            print("\n# Benchmarking inference config:", inference_config)

            current_cfg = copy.deepcopy(cfg)
            current_cfg.engine.model.inference = inference_config
            current_cfg.self_play.engine_overrides.pop("inference", None)

            model_path = Path(tempdir) / "model"
            export_model(current_cfg, model_path)

            executable = compile_selfplay_exe(
                "chess", inference_config, profile="profiling"
            )
            t0 = time.time()
            score = bench_selfplay(executable, model_path, current_cfg)
            t1 = time.time()

            summary.append((inference_config, score, t1 - t0))

    print("\n # Summary:")
    for inference_config, score, duration in summary:
        print("cfg:", inference_config)
        print("score:", score)
        print("duration:", duration)
        print()


def bench_selfplay(executable: Path, model_path: Path, cfg: Config) -> float:
    with tempfile.TemporaryDirectory() as tempdir_:
        tempdir = Path(tempdir_)

        cfg_file = tempdir / "config.json"
        engine_cfg = cfg.self_play_engine_cfg()
        with open(cfg_file, "w") as f:
            json.dump(asdict(engine_cfg), f, indent=2)

        data_entries_dir = tempdir / "data_entries"
        summary_file = tempdir / "summary.json"
        subprocess.check_call(
            [
                # *("samply", "record"),
                executable,
                f"--model1-path={model_path}",
                f"--model2-path={model_path}",
                f"--games-num={cfg.self_play.games_num}",
                f"--out-dir1={data_entries_dir}",
                f"--out-dir2={data_entries_dir}",
                f"--summary-file={summary_file}",
                f"--config-file={cfg_file}",
            ],
            env=runtime_env(engine_cfg, os.environ.copy()),
            cwd=SELF_PLAY_CRATE_TOP,
        )

        with open(summary_file, "r") as f:
            summary = json.load(f)
    return summary["metrics"]["model.run_duration"]


def export_model(cfg: Config, path: Path):
    from cattus_train.self_play import export_model as export_model_impl

    model = GAME.create_model(cfg.model.type, cfg.model.__dict__.copy())

    model_cfg = cfg.self_play_engine_cfg().model
    self_play_input_shape = GAME.model_input_shape(cfg.model.type)
    self_play_input_shape = (model_cfg.batch_size,) + self_play_input_shape[1:]
    export_model_impl(model, path, model_cfg.inference, self_play_input_shape)


if __name__ == "__main__":
    main()
