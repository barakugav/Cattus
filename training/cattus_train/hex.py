from pathlib import Path

import numpy as np
from construct import Array, Float32l, Int8sl, Int64ul, Struct
from torch import nn

from cattus_train import net_utils
from cattus_train.trainable_game import DataEntry, DataEntryParseError, Game


class HexNetType:
    SimpleTwoHeaded = "simple_two_headed"
    ConvNetV1 = "ConvNetV1"


class Hex(Game):
    def __init__(self, size):
        self.BOARD_SIZE = size
        self.PLANES_NUM = 3
        self.MOVE_NUM = self.BOARD_SIZE * self.BOARD_SIZE

        self.ENTRY_FORMAT = Struct(
            "planes" / Array(self.PLANES_NUM * 2, Int64ul),
            "probs" / Array(self.MOVE_NUM, Float32l),
            "winner" / Int8sl,
        )

    def load_data_entry(self, path: Path) -> DataEntry:
        with open(path, "rb") as f:
            entry_bytes = f.read()
        if len(entry_bytes) != self.ENTRY_FORMAT.sizeof():
            raise DataEntryParseError(
                f"invalid training data file: {path} ({len(entry_bytes)} != {self.ENTRY_FORMAT.sizeof()})"
            )
        entry = self.ENTRY_FORMAT.parse(entry_bytes)
        # planes of 128bit are saved as two 64bit values
        planes = np.array(entry.planes, dtype=np.uint64).reshape((self.PLANES_NUM, 2))
        probs = np.array(entry.probs, dtype=np.float32)
        winner = float(entry.winner)

        assert len(planes) == self.PLANES_NUM
        assert len(probs) == self.MOVE_NUM
        return DataEntry(planes=planes, probs=probs, winner=winner)

    def _get_input_shape(self):
        return (1, self.PLANES_NUM, self.BOARD_SIZE, self.BOARD_SIZE)

    def _create_model_convnetv1(self, cfg: dict):
        return net_utils.ConvNetV1(
            input_shape=self._get_input_shape(),
            residual_block_num=cfg["residual_block_num"],
            residual_filter_num=cfg["residual_filter_num"],
            value_head_conv_output_channels_num=cfg["value_head_conv_output_channels_num"],
            policy_head_conv_output_channels_num=cfg["policy_head_conv_output_channels_num"],
            moves_num=self.MOVE_NUM,
        )

    def create_model(self, net_type: str, cfg: dict) -> nn.Module:
        if net_type == HexNetType.SimpleTwoHeaded:
            return net_utils.SimpleTwoHeadedModel(self._get_input_shape(), self.MOVE_NUM)
        elif net_type == HexNetType.ConvNetV1:
            return self._create_model_convnetv1(cfg)
        else:
            raise ValueError("Unknown model type: " + net_type)

    def model_input_shape(self, net_type: str) -> tuple:
        if net_type == HexNetType.SimpleTwoHeaded or net_type == HexNetType.ConvNetV1:
            return self._get_input_shape()
        else:
            raise ValueError("Unknown model type: " + net_type)
