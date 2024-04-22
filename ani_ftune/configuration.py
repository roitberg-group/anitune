import hashlib

import typing as tp
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ani_ftune.utils import DATA_ELEMENTS


_BATCH_PATH = Path.home().joinpath(".local/torchani/Batched")
_BATCH_PATH.mkdir(exist_ok=True, parents=True)

_TRAIN_PATH = Path.home().joinpath(".local/torchani/Train")
_TRAIN_PATH.mkdir(exist_ok=True, parents=True)

_FTUNE_PATH = Path.home().joinpath(".local/torchani/Finetune")
_FTUNE_PATH.mkdir(exist_ok=True, parents=True)


# Dataset parameters
@dataclass
class DatasetConfig:
    fold_idx: tp.Union[int, str] = "single"
    folds: tp.Optional[int] = None
    train_frac: float = 0.8
    validation_frac: float = 0.2
    name: str = "TestData"
    lot: str = "wb97x-631gd"
    batch_size: int = 2560
    shuffle_seed: int = 1234

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
    def path(self) -> Path:
        dict_ = asdict(self)
        dict_.pop("fold_idx")
        state = sorted((k, v) for k, v in dict_.items())
        hasher = hashlib.shake_128()
        hasher.update(str(state).encode())
        _path = _BATCH_PATH / f"{self.name}-{hasher.hexdigest(4)}"
        print(f"Dataset path: {_path}")
        return _path


@dataclass
class ModelConfig:
    flags: tp.Tuple[tp.Tuple[str, bool], ...] = (("repulsion", True), ("dispersion", False))
    builder: str = "FlexibleANI"
    symbols: tp.Optional[tp.Tuple[str, ...]] = None

    def get_symbols(
        self, ds_name: str = "", basis_set: str = "", functional: str = ""
    ) -> tp.Tuple[str, ...]:
        from torchani import datasets  # noqa
        if self.symbols is not None:
            return self.symbols
        print("Fetching present chemical symbols from dataset...")
        symbols = DATA_ELEMENTS.get(ds_name)
        if symbols is not None:
            return symbols
        ds = getattr(datasets, ds_name)(
            skip_check=True, basis_set=basis_set, functional=functional, verbose=False
        )
        symbols = tuple(ds.present_elements(chem_symbols=True))
        print(f"Found {symbols}")
        return symbols

    @property
    def flag_dict(self) -> tp.Dict[str, bool]:
        return {k: v for k, v in self.flags}


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
    finetune: bool = False
    debug: bool = False
    ds: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    optim: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    accel: AccelConfig = field(default=AccelConfig(), compare=False)

    @property
    def path(self) -> Path:
        dict_ = asdict(self)
        dict_.pop("accel")
        keys = tuple(dict_.keys())
        for k in keys:
            if isinstance(dict_[k], dict):
                dict_[k] = sorted((k, v) for k, v in dict_[k].items())
        state = sorted((k, v) for k, v in dict_.items())
        hasher = hashlib.shake_128()
        hasher.update(str(state).encode())
        if self.finetune:
            return _FTUNE_PATH / f"{self.name}-{hasher.hexdigest(4)}"
        if self.debug:
            from uuid import uuid4
            hasher.update(uuid4().bytes)
            return Path("/tmp") / "{self.name}-{self.ds.fold_idx}-{hasher.hexdigest(4)}"
        return _TRAIN_PATH / f"{self.name}-{self.ds.fold_idx}-{hasher.hexdigest(4)}"
