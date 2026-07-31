import logging
import os
import tempfile
from pathlib import Path

import cattus_train
import yaml

logger = logging.getLogger(__name__)


def test_ttt_training():
    logging.basicConfig(level=logging.DEBUG, format="[TicTactToe Training Test]: %(message)s")

    inference_engine = os.getenv("CATTUS_TEST_INFERENCE_ENGINE", "onnx-ort")
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = f"""
game: tictactoe
iterations: 3
debug: false
working_area: {tmp_dir}
model:
    base: null
    type: ConvNetV1
    residual_block_num: 5
    residual_filter_num: 8
    value_head_conv_output_channels_num: 8
    policy_head_conv_output_channels_num: 8
engine:
    mcts:
        sim_num: 600
        explore_factor: 1.41421
        temperature_policy:
            - [9999, 0.0]
        prior_noise_alpha: 0.0
        prior_noise_epsilon: 0.2
        cache_size: 1000000
    model:
        batch_size: 4
        inference:
            engine: {inference_engine}
    threads: 8
self_play:
    engine_overrides:
        mcts:
            temperature_policy:
                - [5,    1.0]
                - [9999, 0.0]
    games_num: 100
    model_compare:
        games_num: 0
        switching_winning_threshold: 0.55
        warning_losing_threshold: 0.55
training:
    latest_data_entries: 3000
    epoch_size: 3000
    batch_size: 16
    learning_rate:
        - [       0.001]
    # l2reg: 0.00005
    use_train_data_across_runs: false
    device: null
"""

        logger.info("Running self play and generating new models...")
        cattus_train.train(cattus_train.Config(**yaml.safe_load(config)), run_id="test")

        logger.info("Checking quality of training...")
        metrics = _get_metrics(tmp_dir)
        assert float(metrics["model0/test/loss"]) > 0
        assert float(metrics["model0/test/value_accuracy"]) > 0.6
        assert float(metrics["model0/test/policy_accuracy"]) > 0.4
        logger.info("Training quality is sufficient")


def _get_metrics(working_area):
    path = Path(working_area) / "metrics" / "test.csv"
    with path.open("r") as f:
        lines = f.readlines()
        columns = lines[0].split(",")
        last_metric = lines[-1].split(",")
        metrics = {}
        for i in range(len(columns)):
            metrics[columns[i]] = last_metric[i]
        return metrics


if __name__ == "__main__":
    test_ttt_training()
