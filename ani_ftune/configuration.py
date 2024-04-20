import hashlib

import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

from torchani import datasets

_BATCH_PATH = Path.home().joinpath(".local/torchani/Batched")
_BATCH_PATH.mkdir(exist_ok=True, parents=True)

_TRAIN_PATH = Path.home().joinpath(".local/torchani/Train")
_TRAIN_PATH.mkdir(exist_ok=True, parents=True)

_FTUNE_PATH = Path.home().joinpath(".local/torchani/Finetune")
_FTUNE_PATH.mkdir(exist_ok=True, parents=True)


# Dataset parameters
@dataclass(frozen=True)
class DatasetConfig:
    fold_idx: tp.Union[int, str] = field(default="single", compare=False)
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
        hasher = hashlib.shake_128()
        hasher.update(bytes(str(hash(self)).encode()))
        return _BATCH_PATH / f"{self.name}-{hasher.hexdigest(4)}"


@dataclass(frozen=True)
class ModelConfig:
    flags: tp.FrozenSet[tp.Tuple[str, bool]] = frozenset(
        {("repulsion", True), ("dispersion", False)}
    )
    builder: str = "FlexibleANI"
    symbols: tp.Optional[tp.Tuple[str, ...]] = None

    def get_symbols(
        self, ds_name: str = "", basis_set: str = "", functional: str = ""
    ) -> tp.Tuple[str, ...]:
        if self.symbols is not None:
            return self.symbols
        print("Fetching present chemical symbols from dataset...")
        ds = getattr(datasets, ds_name)(
            skip_check=True, basis_set=basis_set, functional=functional, verbose=False
        )
        symbols = tuple(ds.present_elements(chem_symbols=True))
        print(f"Found {symbols}")
        return symbols

    @property
    def flag_dict(self) -> tp.Dict[str, bool]:
        return {k: v for k, v in self.flags}


@dataclass(frozen=True)
class LossConfig:
    r"""
    loss-specific configurations
    """

    terms_and_factors: tp.FrozenSet[tp.Tuple[str, float]] = frozenset(
        {("Energies", 1.0)}
    )
    uncertainty_weighted: bool = False


@dataclass(frozen=True)
class OptimizerConfig:
    r"""
    Optimizer configuration
    """

    lr: float = 0.5e-3
    weight_decay: float = 1e-7


@dataclass(frozen=True)
class SchedulerConfig:
    r"""
    lr-Scheduler configuration
    """

    monitor_label: str = "energies"
    factor: float = 0.5
    patience: int = 100
    threshold: float = 0.0


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class TrainConfig:
    r"""
    Configuration for all the training.
    """

    name: str = "run"
    finetune: bool = False
    ds: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    optim: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    accel: AccelConfig = field(default=AccelConfig(), compare=False)

    @property
    def path(self) -> Path:
        hasher = hashlib.shake_128()
        hasher.update(bytes(str(hash(self)).encode()))
        if self.finetune:
            return _FTUNE_PATH / f"{self.name}-{hasher.hexdigest(4)}"
        return _TRAIN_PATH / f"{self.name}-{self.ds.fold_idx}-{hasher.hexdigest(4)}"
