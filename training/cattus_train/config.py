import dataclasses
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic.dataclasses import Field, dataclass


@dataclass(config={"extra": "allow"}, kw_only=True)
class ModelConfig:
    type: str
    base: Path | str = "[none]"


@dataclass(config={"extra": "forbid"}, kw_only=True)
class MctsConfig:
    sim_num: int
    explore_factor: float
    temperature_policy: list[tuple[int, float]]
    prior_noise_alpha: float
    prior_noise_epsilon: float
    cache_size: int = 0


@dataclass(config={"extra": "forbid"}, kw_only=True)
class ExecutorchConfig:
    engine: Literal["executorch"] = "executorch"
    backend: Literal["none", "xnnpack", "mps"] = "none"


@dataclass(config={"extra": "forbid"}, kw_only=True)
class TorchPyConfig:
    engine: Literal["torch-py"] = "torch-py"
    device: Literal["cpu", "cuda", "mps"] | None = None


@dataclass(config={"extra": "forbid"}, kw_only=True)
class TorchTchRsConfig:
    engine: Literal["torch-tch-rs"] = "torch-tch-rs"
    device: Literal["cpu", "cuda", "mps"] | None = None


@dataclass(config={"extra": "forbid"}, kw_only=True)
class OnnxTractConfig:
    engine: Literal["onnx-tract"] = "onnx-tract"


@dataclass(config={"extra": "forbid"}, kw_only=True)
class OnnxOrtConfig:
    engine: Literal["onnx-ort"] = "onnx-ort"


InferenceConfig = ExecutorchConfig | TorchPyConfig | TorchTchRsConfig | OnnxTractConfig | OnnxOrtConfig


@dataclass(config={"extra": "forbid"}, kw_only=True)
class EngineModelConfig:
    batch_size: int
    inference: InferenceConfig = Field(discriminator="engine", default=None)


@dataclass(config={"extra": "forbid"}, kw_only=True)
class EngineConfig:
    mcts: MctsConfig
    model: EngineModelConfig
    threads: int

    def copy_with_overrides(self, overrides: dict[str, Any]) -> "EngineConfig":
        data = dataclasses.asdict(self)

        def override(d: dict[str, Any], o: dict[str, Any]):
            for k, v in o.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    override(d[k], v)
                else:
                    d[k] = v

        override(data, overrides)
        return EngineConfig(**data)


@dataclass(config={"extra": "forbid"}, kw_only=True)
class ModelCompareConfig:
    games_num: int
    engine_overrides: dict[str, Any] = Field(default_factory=dict)
    switching_winning_threshold: float = Field(ge=0.5, le=1.0)
    warning_losing_threshold: float = Field(ge=0.5, le=1.0)


@dataclass(config={"extra": "forbid"}, kw_only=True)
class SelfPlayConfig:
    games_num: int
    engine_overrides: dict[str, Any] = Field(default_factory=dict)
    model_compare: ModelCompareConfig


@dataclass(config={"extra": "forbid"}, kw_only=True)
class TrainingConfig:
    batch_size: int
    learning_rate: list[list[float]]
    use_train_data_across_runs: bool = False
    threads: Optional[int] = 1
    latest_data_entries: int
    iteration_data_entries: int
    device: Literal["cpu", "cuda", "mps"] | None = None

    def __post_init__(self):
        if self.latest_data_entries < self.iteration_data_entries:
            raise ValueError("latest_data_entries must be >= iteration_data_entries")


@dataclass(config={"extra": "forbid"}, kw_only=True)
class Config:
    working_area: Path
    games_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    metrics_dir: Optional[Path] = None
    game: str
    model: ModelConfig
    engine: EngineConfig
    self_play: SelfPlayConfig
    model_num: int = 1
    iterations: int
    training: TrainingConfig
    debug: bool

    def self_play_engine_cfg(self) -> EngineConfig:
        return self.engine.copy_with_overrides(self.self_play.engine_overrides)

    def model_compare_engine_cfg(self) -> EngineConfig:
        return self.engine.copy_with_overrides(self.self_play.model_compare.engine_overrides)
