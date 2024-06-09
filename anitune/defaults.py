r"""
This module holds default values for different architectures, optimizers and lr
schedulers
"""

import sys
import typing as tp
from dataclasses import dataclass, asdict
from anitune.utils import Scalar, ScalarTuple


@dataclass
class Options:
    def as_dict(self) -> tp.Dict[str, Scalar]:
        return asdict(self)


@dataclass
class ArchArgs(Options):
    pass


@dataclass
class FlexANI1(ArchArgs):
    # AEV
    radial_cutoff: float = 5.2
    angular_cutoff: float = 3.5
    radial_shifts: int = 16
    angular_shifts: int = 4
    angle_sections: int = 8
    radial_precision: float = 16.0
    angular_precision: float = 8.0
    angular_zeta: float = 32.0
    # Networks
    atomic_maker: str = "ani1x"
    activation: str = "gelu"
    bias: bool = False
    # Potentials and cutoff
    cutoff_fn: str = "smooth2"
    dispersion: bool = False
    repulsion: bool = True


@dataclass
class FlexANI2(ArchArgs):
    # AEV
    radial_cutoff: float = 5.2
    angular_cutoff: float = 3.5
    radial_shifts: int = 16
    angular_shifts: int = 8
    angle_sections: int = 4
    radial_precision: float = 19.7
    angular_precision: float = 12.5
    angular_zeta: float = 14.1
    # Networks
    atomic_maker: str = "ani2x"
    activation: str = "gelu"
    bias: bool = False
    # Potentials and cutoff
    cutoff_fn: str = "smooth2"
    dispersion: bool = False
    repulsion: bool = True


@dataclass
class ScaleANI1(ArchArgs):
    # AEV
    radial_cutoff: float = 5.2
    angular_cutoff: float = 3.5
    radial_shifts: int = 16
    angular_shifts: int = 4
    angle_sections: int = 8
    radial_precision: float = 16.0
    angular_precision: float = 8.0
    angular_zeta: float = 32.0
    # Networks, 3 items -> 4 layers
    layer_dims: str = "1/4,2/3,1/2"
    activation: str = "gelu"
    bias: bool = False
    # Potentials and cutoff
    cutoff_fn: str = "smooth2"
    dispersion: bool = False
    repulsion: bool = True


@dataclass
class ScaleANI2(ArchArgs):
    # AEV
    radial_cutoff: float = 5.2
    angular_cutoff: float = 3.5
    radial_shifts: int = 16
    angular_shifts: int = 8
    angle_sections: int = 4
    radial_precision: float = 19.7
    angular_precision: float = 12.5
    angular_zeta: float = 14.1
    # Networks, 5 items -> 4 layers
    layer_dims: str = "1/4,2/3,1/2"
    activation: str = "gelu"
    bias: bool = False
    # Potentials and cutoff
    cutoff_fn: str = "smooth2"
    dispersion: bool = False
    repulsion: bool = True


@dataclass
class SchedulerArgs(Options):
    pass


@dataclass
class ReduceLROnPlateau(SchedulerArgs):
    factor: float = 0.5
    patience: int = 100
    threshold: float = 0.0


@dataclass
class StepLR(SchedulerArgs):
    gamma: float = 0.5  # gamma
    step_size: int = 150  # step_size in epochs


@dataclass
class ExponentialLR(SchedulerArgs):
    gamma: float = 0.99


# max_epochs is set to scheduler.max_epochs automatically
@dataclass
class CosineAnnealingLR(SchedulerArgs):
    total_epochs: int = 500
    eta_min: float = 0.0  # eta_min


@dataclass
class OptimizerArgs(Options):
    lr: float = 5e-4
    weight_decay: float = 1e-6


# eps for Adamax, Radam, Adam, AdamW is not shown, and it is always 1e-8
@dataclass
class Adam(OptimizerArgs):
    amsgrad: bool = False


@dataclass
class AdamW(OptimizerArgs):
    amsgrad: bool = False


@dataclass
class RMSprop(OptimizerArgs):
    momentum: float = 0.0
    alpha: float = 0.99
    centered: bool = True


# @dataclass
# class Radam(OptimizerArgs):
# decoupled_weight_decay: bool = True  # decoupled_weight_decay


@dataclass
class Adamax(OptimizerArgs):
    pass


@dataclass
class SGD(OptimizerArgs):
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    # Has no fused impl


def resolve_options_raw(
    _options: tp.Optional[tp.List[str]], cls: str
) -> tp.Dict[str, Scalar]:
    if _options is None:
        _options = []
    try:
        default_options: tp.Dict[str, Scalar] = getattr(
            sys.modules[__name__], cls
        )().as_dict()
    except AttributeError:
        raise RuntimeError(f"Unknown class {cls}")
    options: tp.Dict[str, Scalar] = dict()
    if _options is not None:
        for kv in _options:
            if "=" not in kv:
                raise RuntimeError(
                    f"Incorrect format for option {kv}, should be key=value"
                )
            k = kv.split("=")[0]
            if k not in default_options:
                raise RuntimeError(f"Incorrect key in option {kv}")
            try:
                _v = kv.split("=")[1].replace("true", "True").replace("false", "False")
                v = eval(_v)
            except Exception:
                raise RuntimeError(f"Incorrect value in option {kv}") from None
            options.update({k: v})
    default_options.update(options)
    return default_options


def make_scalar_tuples(options: tp.Dict[str, Scalar]) -> tp.Tuple[ScalarTuple, ...]:
    return tuple(sorted((k, v) for k, v in options.items()))
    raise NotImplementedError()


def resolve_options(
    _options: tp.Optional[tp.List[str]], cls: str
) -> tp.Tuple[ScalarTuple, ...]:
    return make_scalar_tuples(resolve_options_raw(_options, cls))


def parse_scheduler_str(scheduler: str) -> str:
    scheduler = scheduler.capitalize()
    if "LR" not in scheduler:
        scheduler = "".join((scheduler, "LR"))
    if scheduler == "PlateauLR":
        scheduler = "ReduceLROnPlateau"
    elif scheduler in ["CosineLR", "CosLR"]:
        scheduler = "CosineAnnealingLR"
    elif scheduler in ["ExponentialLR", "ExpLR"]:
        scheduler = "ExponentialLR"
    elif scheduler == "StepLR":
        pass
    else:
        raise ValueError(f"Unsupported scheduler {scheduler}")
    return scheduler


def parse_optimizer_str(optimizer: str) -> str:
    optimizer = optimizer.capitalize()
    if optimizer.lower() == "rmsprop":
        optimizer = "RMSprop"
    if optimizer.lower() == "sgd":
        optimizer = "SGD"
    if optimizer.lower() == "adamw":
        optimizer = "AdamW"
    return optimizer
