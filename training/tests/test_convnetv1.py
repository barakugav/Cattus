import logging
import os
import tempfile

import yaml

import cattus_train


def _test_convnetv1(game_name):
    inference_engine = os.getenv("CATTUS_TEST_INFERENCE_ENGINE", "onnx-ort")
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = f"""
game: "{game_name}"
iterations: 2
debug: true
working_area: {tmp_dir}
model:
    base: "[none]"
    type: "ConvNetV1"
    residual_block_num: 2
    residual_filter_num: 2
    value_head_conv_output_channels_num: 4
    policy_head_conv_output_channels_num: 4
engine:
    mcts:
        sim_num: 10
        explore_factor: 1.41421
        temperature_policy:
            - [9999, 0.0]
        prior_noise_alpha: 0.0
        prior_noise_epsilon: 0.2
        cache_size: 1000
    model:
        batch_size: 1
        inference:
            engine: {inference_engine}
    threads: 1
self_play:
    games_num: 8

    model_compare:
        games_num: 4
        switching_winning_threshold: 0.55
        warning_losing_threshold: 0.55
training:
    latest_data_entries: 1024
    epoch_size: 128
    batch_size: 4
    learning_rate:
        - [       0.001]
    # l2reg: 0.00005
    use_train_data_across_runs: false
    device: null
"""

        logging.info("Running self play and generating new models...")
        cattus_train.train(cattus_train.Config(**yaml.safe_load(config)))


def test_ttt_convnetv1():
    logging.basicConfig(level=logging.DEBUG, format="[TTT ConvNetV1 Test]: %(message)s")
    _test_convnetv1("tictactoe")


def test_hex_convnetv1():
    logging.basicConfig(level=logging.DEBUG, format="[Hex ConvNetV1 Test]: %(message)s")
    for size in [4, 5, 7, 9, 11]:
        _test_convnetv1(f"hex{size}")


def test_chess_convnetv1():
    logging.basicConfig(level=logging.DEBUG, format="[Chess ConvNetV1 Test]: %(message)s")
    _test_convnetv1("chess")


if __name__ == "__main__":
    test_ttt_convnetv1()
    test_hex_convnetv1()
    test_chess_convnetv1()
    print("test passed")
