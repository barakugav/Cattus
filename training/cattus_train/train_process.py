import copy
import json
import logging
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cattus_train.chess import Chess
from cattus_train.config import Config
from cattus_train.data_set import DataSet
from cattus_train.hex import Hex
from cattus_train.self_play import compile_selfplay_exe, export_model, exported_model_suffix, runtime_env
from cattus_train.tictactoe import TicTacToe
from cattus_train.trainable_game import Game

CATTUS_TOP = Path(__file__).parent.parent.parent.resolve()
CATTUS_TRAIN_TOP = CATTUS_TOP / "training"
SELF_PLAY_CRATE_DIR = CATTUS_TRAIN_TOP / "self-play"


class TrainProcess:
    def __init__(self, cfg: Config):
        cfg = copy.deepcopy(cfg)
        self.cfg: Config = cfg

        if cfg.working_area.parts[0] == "{CATTUS_TRAIN_TOP}":
            cfg.working_area = CATTUS_TRAIN_TOP.joinpath(*cfg.working_area.parts[1:])
        cfg.working_area = Path().joinpath(
            *[p.replace("{GAME_NAME}", cfg.game) for p in cfg.working_area.resolve().parts]
        )
        cfg.games_dir = cfg.games_dir or cfg.working_area / "games"
        cfg.models_dir = cfg.working_area / "models"
        cfg.metrics_dir = cfg.working_area / "metrics"
        cfg.working_area.mkdir(parents=True, exist_ok=True)
        cfg.games_dir.mkdir(parents=True, exist_ok=True)
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        cfg.metrics_dir.mkdir(parents=True, exist_ok=True)

        self._self_play_engine_cfg = cfg.self_play_engine_cfg()
        self._model_compare_engine_cfg = cfg.model_compare_engine_cfg()

        self._game: Game
        if cfg.game == "tictactoe":
            self._game = TicTacToe()
        elif re.match("hex[0-9]+", cfg.game):
            size = int(re.findall("hex([0-9]+)", cfg.game)[0])
            self._game = Hex(size)
        elif cfg.game == "chess":
            self._game = Chess()
        else:
            raise ValueError("Unknown game argument in config file.")
        self.temp_dir_ = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_.name)
        self._self_play_exec_path: Path = self.temp_dir / "bin" / "self_play"
        self._model_compare_exec_path: Path = self.temp_dir / "bin" / "model_compare"

        self._net_type: str = cfg.model.type
        match cfg.model.base:
            case "[none]":
                self._base_model_path = self._save_model(self._create_model())
            case "[latest]":
                logging.warning("Choosing latest model based on directory name format")
                all_models = list(cfg.models_dir.iterdir())
                if len(all_models) == 0:
                    raise ValueError("Model [latest] was requested, but no existing models were found.")
                self._base_model_path = sorted(all_models)[-1]
            case Path():
                self._base_model_path = cfg.model.base
            case _:
                raise ValueError("Unknown base model argument in config file.")

        if cfg.training.device is None:
            if torch.cuda.is_available():
                cfg.training.device = "cuda"
            elif torch.backends.mps.is_available():
                cfg.training.device = "mps"
            else:
                cfg.training.device = "cpu"

        self._lr_scheduler = LearningRateScheduler(cfg.training.learning_rate)

    def run_training_loop(self, run_id=None):
        if run_id is None:
            run_id = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        self._run_id = run_id
        metrics_filename = self.cfg.metrics_dir / f"{self._run_id}.csv"

        # raise ValueError(self._base_model_path)
        best_model = (self._load_model(self._base_model_path), self._base_model_path)
        latest_models = [best_model]
        if self.cfg.model_num > 1:
            for _ in range(self.cfg.model_num - 1):
                latest_models.append(copy.deepcopy(best_model))

        logging.info("Starting training process with config:")
        logging.info(f"Configuration:\n{dic2str(asdict(self.cfg))}")
        logging.info("run ID:\t%s", self._run_id)
        logging.info("base model:\t%s", self._base_model_path)
        logging.info("metrics file:\t%s", metrics_filename)

        logging.info("Building Self-play executable...")
        profile = "dev" if self.cfg.debug else "release"
        self._self_play_exec_path.parent.mkdir(parents=True, exist_ok=True)
        self._model_compare_exec_path.parent.mkdir(parents=True, exist_ok=True)
        temp_self_play_exec_path = compile_selfplay_exe(
            self.cfg.game, self._self_play_engine_cfg.model.inference, profile=profile
        )
        shutil.copy(temp_self_play_exec_path, self._self_play_exec_path)
        temp_model_compare_exec_path = compile_selfplay_exe(
            self.cfg.game, self._model_compare_engine_cfg.model.inference, profile=profile
        )
        shutil.copy(temp_model_compare_exec_path, self._model_compare_exec_path)

        for iter_num in range(self.cfg.iterations):
            logging.info(f"Training iteration {iter_num}")
            self._metrics = {}

            # Generate training data using the best model
            self._self_play(best_model[1])

            # Train latest models from training data
            latest_models = self._train(latest_models, iter_num)

            # Compare latest model to the current best, and switch if better
            best_model = self._compare_models(best_model, latest_models)

            # Write iteration metrics
            self._write_metrics(metrics_filename)

    def _self_play(self, model_dir: Path):
        logging.info("Self playing using model: %s", model_dir)

        games_dir = self.cfg.games_dir / self._run_id
        data_entries_dir = games_dir / datetime.now().strftime("%y%m%d_%H%M%S_%f")

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_file = Path(tmp_dir) / "config.json"
            engine_cfg = self._self_play_engine_cfg
            with open(cfg_file, "w") as f:
                json.dump(asdict(engine_cfg), f, indent=2)
            summary_file = Path(tmp_dir) / "selfplay_summary.json"

            self_play_start_time = time.time()
            model_path = model_dir / "self_play" / f"model.{exported_model_suffix(engine_cfg.model.inference)}"
            subprocess.check_call(
                [
                    self._self_play_exec_path,
                    f"--model1-path={model_path}",
                    f"--model2-path={model_path}",
                    f"--games-num={self.cfg.self_play.games_num}",
                    f"--out-dir1={data_entries_dir}",
                    f"--out-dir2={data_entries_dir}",
                    f"--summary-file={summary_file}",
                    f"--config-file={cfg_file}",
                ],
                env=runtime_env(engine_cfg, os.environ.copy()),
                cwd=SELF_PLAY_CRATE_DIR,
            )
            self._metrics["self_play_duration"] = time.time() - self_play_start_time

            with open(summary_file, "r") as f:
                summary = json.load(f)
            self._metrics.update(
                {
                    "net_activations_count": summary["metrics"]["model.activation_count"],
                    "net_run_duration_average_us": summary["metrics"]["model.run_duration"],
                    # "batch_size_average": summary["metrics"]["batch_size_average"],
                    # "search_count": summary["metrics"]["search_count"],
                    "search_duration": summary["metrics"]["mcts.search_duration"],
                    "cache_hit_ratio": summary["metrics"]["cache.hits"]
                    / (summary["metrics"]["cache.hits"] + summary["metrics"]["cache.misses"]),
                }
            )

    def _train(self, models, iter_num) -> list[tuple[nn.Module, Path]]:
        train_data_dir = (
            self.cfg.games_dir if self.cfg.training.use_train_data_across_runs else self.cfg.games_dir / self._run_id
        )

        lr = self._lr_scheduler.get_lr(iter_num)
        logging.debug("Training models with learning rate %f", lr)

        losses = [None] * len(models)
        value_accuracies = [None] * len(models)
        policy_accuracies = [None] * len(models)
        training_durations = [None] * len(models)
        trained_models = [None] * len(models)

        def train_models(model_list: list[tuple[int, nn.Module]]):
            for m_idx, model in model_list:
                data_set = DataSet(
                    self._game, train_data_dir, self.cfg.training, torch.device(self.cfg.training.device)
                )
                data_loader = DataLoader(data_set, batch_size=self.cfg.training.batch_size)

                def mask_illegal_moves(output, target):
                    output = torch.where(target >= 0, output, -1e10)
                    target = nn.ReLU()(target)
                    return output, target

                def loss_cross_entropy(output, target):
                    output, target = mask_illegal_moves(output, target)
                    return F.cross_entropy(output, target)

                def loss_fn(outputs, targets):
                    policy_output, value_output = outputs
                    policy_target, value_target = targets
                    policy_loss = loss_cross_entropy(policy_output, policy_target)
                    value_loss = F.mse_loss(value_output.squeeze(), value_target)
                    return policy_loss + value_loss

                model.train()
                model.to(self.cfg.training.device)
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                final_batch = None
                training_start_time = time.time()
                for x, y in data_loader:
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    optimizer.step()
                    final_batch = (x, y)
                train_duration = time.time() - training_start_time
                model = model.to("cpu")

                def policy_head_accuracy(output, target):
                    output, target = mask_illegal_moves(output, target)
                    predicted = torch.argmax(output, dim=1)
                    target = torch.argmax(target, dim=1)
                    return (predicted == target).float().mean()

                def value_head_accuracy(output, target):
                    # Both the target and output should be in range [-1,1]
                    return 1 - torch.abs(target - output).mean() / 2

                with torch.no_grad():
                    model.eval()
                    final_x, final_y = final_batch
                    final_x = final_x.to("cpu")
                    final_y = (final_y[0].to("cpu"), final_y[1].to("cpu"))
                    final_outputs = model(final_x)
                    losses[m_idx] = loss_fn(final_outputs, final_y).detach().item()
                    policy_accuracies[m_idx] = policy_head_accuracy(final_outputs[0], final_y[0]).detach().item()
                    value_accuracies[m_idx] = value_head_accuracy(final_outputs[1], final_y[1]).detach().item()
                    training_durations[m_idx] = train_duration

                trained_models[m_idx] = (model, self._save_model(model))

        models = [(idx, model) for idx, (model, _model_path) in enumerate(models)]
        workers_num = min(self.cfg.training.threads, len(models))
        workers_num = workers_num if self.cfg.training.device == "cpu" else 1
        if workers_num > 1:
            # Divide the training into jobs
            jobs_per_cpu = len(models) // workers_num
            indices = np.arange(0, len(models) + jobs_per_cpu, jobs_per_cpu)
            jobs = [models[i:j] for i, j in zip(indices[:-1], indices[1:])]

            # Execute all jobs
            with ThreadPool(len(jobs)) as pool:
                pool.map(train_models, jobs)
        else:
            # Train on a single CPU thread
            train_models(models)

        for model_idx in range(len(models)):
            logging.info(f"Model{model_idx} training metrics:")
            logging.info(f"\tLoss            {losses[model_idx]:.4f}")
            logging.info(f"\tValue accuracy  {value_accuracies[model_idx]:.4f}")
            logging.info(f"\tPolicy accuracy {policy_accuracies[model_idx]:.4f}")

            self._metrics.update(
                {
                    f"loss_{model_idx}": losses[model_idx],
                    f"value_accuracy_{model_idx}": value_accuracies[model_idx],
                    f"policy_accuracy_{model_idx}": policy_accuracies[model_idx],
                }
            )
        self._metrics["training_duration"] = sum(training_durations)

        return trained_models

    def _compare_models(self, best_model, latest_models) -> tuple[nn.Module, Path]:
        if self.cfg.self_play.model_compare.games_num == 0:
            assert len(latest_models) == 1, "Model comparison can be skipped only when one model is trained"
            logging.debug("Skipping model comparison, considering latest model as best")
            return latest_models[0]

        logging.info("Comparing latest models to best model...")
        for model_idx, (latest_model, latest_model_path) in enumerate(latest_models):
            # Compare the best model to the latest/trained model
            with tempfile.TemporaryDirectory() as tmp_dir:
                # take the opportunity to generate more games to main games directory
                games_dir = self.cfg.games_dir / self._run_id
                best_games_dir = games_dir / datetime.now().strftime("%y%m%d_%H%M%S_%f")
                trained_games_dir = Path(tmp_dir) / "games"

                compare_start_time = time.time()
                best_wr, trained_wr = self._compare_model_impl(
                    best_model[1], latest_model_path, best_games_dir, trained_games_dir
                )
                winning, losing = trained_wr, 1 - best_wr
                self._metrics["model_compare_duration"] = time.time() - compare_start_time
                self._metrics[f"trained_model_win_rate_{model_idx}"] = winning
                logging.debug(f"Trained model winning rate: {winning}")

                if winning > self.cfg.self_play.model_compare.switching_winning_threshold:
                    best_model = (latest_model, latest_model_path)
                    # In case the new model is the new best model, take the opportunity and use the new games generated
                    # by the comparison stage as training data in future training steps
                    for filename in os.listdir(trained_games_dir):
                        shutil.move(trained_games_dir / filename, best_games_dir)
                elif len(latest_models) == 1 and losing > self.cfg.self_play.model_compare.warning_losing_threshold:
                    logging.warning("New model is worse than previous one, losing ratio: %f", losing)
        return best_model

    def _compare_model_impl(self, model1_dir: Path, model2_dir: Path, model1_games_dir: Path, model2_games_dir: Path):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_file = Path(tmp_dir) / "config.json"
            engine_cfg = self._model_compare_engine_cfg
            with open(cfg_file, "w") as f:
                json.dump(asdict(engine_cfg), f, indent=2)

            model_suffix = exported_model_suffix(engine_cfg.model.inference)
            model1_path = model1_dir / "model_compare" / f"model.{model_suffix}"
            model2_path = model2_dir / "model_compare" / f"model.{model_suffix}"
            compare_res_file = Path(tmp_dir) / "compare_result.json"
            subprocess.check_call(
                [
                    self._model_compare_exec_path,
                    f"--model1-path={model1_path}",
                    f"--model2-path={model2_path}",
                    f"--games-num={self.cfg.self_play.model_compare.games_num}",
                    f"--out-dir1={model1_games_dir}",
                    f"--out-dir2={model2_games_dir}",
                    f"--summary-file={compare_res_file}",
                    f"--config-file={cfg_file}",
                ],
                env=runtime_env(engine_cfg, os.environ.copy()),
                cwd=SELF_PLAY_CRATE_DIR,
            )
            with open(compare_res_file, "r") as res_file:
                res = json.load(res_file)
            w1, w2, d = res["player1_wins"], res["player2_wins"], res["draws"]
            total_games = w1 + w2 + d
            return w1 / total_games, w2 / total_games

    def _create_model(self) -> nn.Module:
        return self._game.create_model(self._net_type, self.cfg.model.__dict__.copy())

    def _save_model(self, model: nn.Module) -> Path:
        model_time = datetime.now().strftime("%y%m%d_%H%M%S_%f") + "_{0:04x}".format(random.randint(0, 1 << 16))
        model_dir = self.cfg.models_dir / f"model_{model_time}"
        model_dir.mkdir(parents=True)

        # Save model as state dict for training
        torch.save(model.state_dict(), model_dir / "model.pt")

        #### Export model for self play
        self_play_model_cfg = self._self_play_engine_cfg.model
        model_compare_model_cfg = self._model_compare_engine_cfg.model
        self_play_model_path = model_dir / "self_play" / f"model.{exported_model_suffix(self_play_model_cfg.inference)}"
        model_compare_model_path = (
            model_dir / "model_compare" / f"model.{exported_model_suffix(model_compare_model_cfg.inference)}"
        )
        self_play_model_path.parent.mkdir(parents=True)
        model_compare_model_path.parent.mkdir(parents=True)
        # Export using a batch size different from training
        self_play_input_shape = self._game.model_input_shape(self._net_type)
        self_play_input_shape = (self_play_model_cfg.batch_size,) + self_play_input_shape[1:]
        export_model(model, self_play_model_path, self_play_model_cfg.inference, self_play_input_shape)
        if self_play_model_cfg.inference != model_compare_model_cfg.inference:
            model_compare_input_shape = self._game.model_input_shape(self._net_type)
            model_compare_input_shape = (model_compare_model_cfg.batch_size,) + model_compare_input_shape[1:]
            export_model(
                model,
                model_compare_model_path,
                model_compare_model_cfg.inference,
                model_compare_input_shape,
            )
        else:
            if self_play_model_path.is_file():
                shutil.copyfile(self_play_model_path, model_compare_model_path)
            else:
                shutil.copytree(self_play_model_path, model_compare_model_path)

        return model_dir

    def _load_model(self, model_dir: Path) -> nn.Module:
        model = self._create_model()
        state_dict = torch.load(model_dir / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def _write_metrics(self, filename: Path):
        per_model_columns = [
            # "value_loss",
            # "policy_loss",
            "loss",
            "value_accuracy",
            "policy_accuracy",
            "trained_model_win_rate",
        ]
        per_model_columns = [[f"{col}_{m_idx}" for col in per_model_columns] for m_idx in range(self.cfg.model_num)]
        columns = [
            "net_run_duration_average_us",
            "batch_size_average",
            "net_activations_count",
            "search_duration_average_ms",
            "search_count",
            "cache_hit_ratio",
            "self_play_duration",
            "training_duration",
            "model_compare_duration",
        ] + sum(per_model_columns, [])

        values = [str(self._metrics.get(metric, "")) for metric in columns]

        # write columns
        if not filename.exists():
            with open(filename, "w") as metrics_file:
                metrics_file.write(",".join(columns) + "\n")

        # write values
        with open(filename, "a") as metrics_file:
            metrics_file.write(",".join(values) + "\n")


class LearningRateScheduler:
    def __init__(self, learning_rates: list[list[float]]):
        assert len(learning_rates) > 0

        thresholds = []
        for idx, elm in enumerate(learning_rates[:-1]):
            assert len(elm) == 2
            if idx > 0:
                # assert the iters thresholds are ordered
                assert elm[0] > learning_rates[idx - 1][0]
            thresholds.append((elm[0], elm[1]))
        self._thresholds = thresholds

        # last elm, no iter threshold
        final_lr = learning_rates[-1]
        assert len(final_lr) == 1
        self.final_lr = final_lr[0]

    def get_lr(self, training_iter):
        for threshold, lr in self._thresholds:
            if training_iter < threshold:
                return lr
        return self.final_lr


def dic2str(d, indent=0):
    s = ""
    for key, value in d.items():
        if isinstance(value, dict):
            s += "  " * indent + str(key) + ":\n"
            s += dic2str(value, indent + 1)
        else:
            s += "  " * indent + str(key) + ": \t" + str(value) + "\n"
    return s
