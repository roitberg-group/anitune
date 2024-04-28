import itertools
import hashlib

import typing as tp
from dataclasses import dataclass, asdict
from pathlib import Path

from ani_ftune.utils import (
    _BATCH_PATH,
    _FTUNE_PATH,
    _TRAIN_PATH,
    _DEBUG_FTUNE_PATH,
    _DEBUG_TRAIN_PATH,
)


def load_state_dict(path: Path) -> tp.Dict[str, tp.Any]:
    r"""
    Load a model's state dict either from a torch .pt file, or a lightning
    .ckpt file (it is assumed that the model is located in a .model attribute
    in the LightningModule in the latter case).
    """
    import torch  # noqa

    _state_dict = torch.load(path, map_location="cpu")
    if "state_dict" in _state_dict:
        _state_dict = _state_dict["state_dict"]
        return {
            k.replace("model.", ""): v
            for k, v in _state_dict.items()
            if k.startswith("model.")
        }
    return _state_dict


class ConfigError(RuntimeError):
    pass


@dataclass
class FinetuneConfig:
    r"""
    ftune-specific configurations
    """

    state_dict_path: tp.Optional[Path]
    num_head_layers: int = 1
    backbone_lr: float = 0.0

    @property
    def frozen_backbone(self) -> bool:
        return self.backbone_lr == 0.0

    @property
    def pretrained_state_dict(self) -> tp.Dict[str, tp.Any]:
        if self.state_dict_path is None:
            return {}
        return load_state_dict(self.state_dict_path)


# Dataset parameters
@dataclass
class DatasetConfig:
    r"""
    dataset-specific configurations
    """

    fold_idx: tp.Union[int, str] = "single"
    folds: tp.Optional[int] = None
    train_frac: float = 0.8
    validation_frac: float = 0.2
    properties: tp.Tuple[str, ...] = ()
    data_names: tp.Tuple[str, ...] = ()
    src_paths: tp.Tuple[Path, ...] = ()
    lot: str = "wb97x-631gd"
    batch_size: int = 2560
    shuffle_seed: int = 1234
    label: str = ""

    @property
    def functional(self) -> str:
        return self.lot.split("-")[0]

    @property
    def basis_set(self) -> str:
        return self.lot.split("-")[1]

    @property
    def split_dict(self) -> tp.Dict[str, float]:
        return {
            "training": self.train_frac,
            "validation": self.validation_frac,
        }

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        return "_".join(
            itertools.chain(
                (p.stem.replace("-", "_") for p in sorted(self.src_paths)),
                sorted(self.data_names),
            )
        )

    @property
    def path(self) -> Path:
        dict_ = asdict(self)
        dict_.pop("fold_idx")
        dict_.pop("label")
        state = sorted((k, v) for k, v in dict_.items())
        hasher = hashlib.shake_128()
        hasher.update(str(state).encode())
        _path = _BATCH_PATH / f"{self.name}-{hasher.hexdigest(4)}"
        print(f"Dataset path: {_path}")
        return _path


@dataclass
class ModelConfig:
    r"""
    model-specific configurations
    """

    kwargs: tp.Tuple[tp.Tuple[str, tp.Union[bool, int, float, str, None]], ...] = ()
    builder: str = "FlexibleANI"
    symbols: tp.Optional[tp.Tuple[str, ...]] = None

    @property
    def kwargs_dict(self) -> tp.Dict[str, tp.Union[bool, int, float, str, None]]:
        return {k: v for k, v in self.kwargs}


@dataclass
class LossConfig:
    r"""
    loss-specific configurations
    """

    terms_and_factors: tp.Tuple[tp.Tuple[str, float], ...] = (("Energies", 1.0),)
    uncertainty_weighted: bool = False


@dataclass
class OptimizerConfig:
    r"""
    Optimizer configuration
    """

    lr: float = 0.5e-3
    weight_decay: float = 1e-7


@dataclass
class SchedulerConfig:
    r"""
    lr-Scheduler configuration
    """

    monitor_label: str = "energies"
    factor: float = 0.5
    patience: int = 100
    threshold: float = 0.0


@dataclass
class AccelConfig:
    r"""
    Acceleration specific configuration. Does not affect reproducibility.
    """

    device: str = "gpu"
    use_cuda_ops: bool = True
    max_batches_per_packet: int = 300
    num_workers: int = 2
    prefetch_factor: int = 2
    max_epochs: int = 2000
    limit: tp.Optional[int] = None
    deterministic: bool = False
    detect_anomaly: bool = False

    @property
    def log_interval(self) -> tp.Optional[int]:
        return max(1, self.limit) if self.limit is not None else 50

    @property
    def train_limit(self) -> tp.Optional[int]:
        return self.limit if self.limit is None else max(1, int(self.limit))

    @property
    def validation_limit(self) -> tp.Optional[int]:
        return self.limit if self.limit is None else max(1, int(self.limit / 10))


@dataclass
class TrainConfig:
    r"""
    Configuration for all the training.
    """

    name: str = "run"
    ds: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    optim: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    ftune: tp.Optional[FinetuneConfig] = None
    debug: bool = False
    accel: AccelConfig = AccelConfig()

    @property
    def ftune_mode(self) -> bool:
        return self.ftune is not None

    @property
    def path(self) -> Path:
        dict_ = asdict(self)
        if self.ftune_mode:
            root = _FTUNE_PATH
            if self.debug:
                root = _DEBUG_FTUNE_PATH
        else:
            root = _TRAIN_PATH
            if self.debug:
                root = _DEBUG_TRAIN_PATH

        dict_.pop("debug")
        dict_.pop("accel")
        dict_.pop("name")
        keys = tuple(dict_.keys())
        for k in keys:
            if isinstance(dict_[k], dict):
                dict_[k] = sorted((k, v) for k, v in dict_[k].items())
        state = sorted((k, v) for k, v in dict_.items())
        hasher = hashlib.shake_128()
        hasher.update(str(state).encode())
        return root / f"{self.name}-{self.ds.fold_idx}-{hasher.hexdigest(4)}"
