import tempfile

import yaml

from cattus_train.config import Config
from cattus_train.train_process import TrainProcess, win_rate


def test_win_rate_counts_draws_as_half():
    # a strictly better challenger (never loses) clears the 0.55 gate despite many draws
    assert win_rate(4, 16, 20) == 0.6
    # two equal models that always draw stay at 0.5 and are not promoted
    assert win_rate(0, 20, 20) == 0.5
    # the two players' scores sum to 1
    assert win_rate(3, 5, 12) + win_rate(4, 5, 12) == 1.0


def _ttt_config(tmp_dir: str) -> Config:
    return Config(
        **yaml.safe_load(f"""
game: tictactoe
iterations: 1
debug: false
working_area: {tmp_dir}
model:
    base: null
    type: ConvNetV1
    residual_block_num: 1
    residual_filter_num: 4
    value_head_conv_output_channels_num: 2
    policy_head_conv_output_channels_num: 2
engine:
    mcts:
        sim_num: 2
        explore_factor: 1.41421
        temperature_policy: [[9999, 0.0]]
        prior_noise_alpha: 0.0
        prior_noise_epsilon: 0.0
        cache_size: 0
    model:
        batch_size: 1
        inference:
            engine: onnx-ort
    threads: 1
self_play:
    engine_overrides: {{}}
    games_num: 2
    model_compare:
        games_num: 0
        switching_winning_threshold: 0.55
        warning_losing_threshold: 0.55
training:
    latest_data_entries: 100
    epoch_size: 100
    batch_size: 8
    learning_rate: [[0.001]]
    use_train_data_across_runs: true
    device: cpu
""")
    )


def test_train_with_no_data_is_skipped():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tp = TrainProcess(_ttt_config(tmp_dir), run_id="test")
        base = (tp._load_model(tp._base_model_path), tp._base_model_path)
        out = tp._train([base], -1)  # empty games dir must not raise
        assert out == [base]
