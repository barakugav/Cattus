from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import nn


@dataclass(kw_only=True, slots=True)
class DataEntry:
    planes: np.ndarray
    probs: np.ndarray
    winner: float


class Game(ABC):
    PLANES_NUM: int
    BOARD_SIZE: int

    @abstractmethod
    def create_model(self, net_type: str, cfg: dict) -> nn.Module: ...

    @abstractmethod
    def model_input_shape(self, net_type: str) -> tuple: ...

    @abstractmethod
    def load_data_entry(self, path: Path) -> DataEntry: ...


class DataEntryParseError(ValueError):
    def __init__(self, message, errors):
        super().__init__(message)
