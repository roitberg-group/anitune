r"""
This module holds default values for different architectures, optimizers and lr schedulers
"""

import typing as tp
from dataclasses import dataclass, asdict
from anitune.utils import Scalar, ScalarTuple


@dataclass
class Options:
    def asdict(self) -> tp.Dict[str, Scalar]:
        return asdict(self)


@dataclass
class FlexANI1(Options):
    neighborlist: str = "full_pairwise"
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
class FlexANI2(FlexANI1):
    # AEV
    angular_shifts: int = 8
    angle_sections: int = 4
    radial_precision: float = 19.7
    angular_precision: float = 12.5
    angular_zeta: float = 14.1
    # Networks
    atomic_maker: str = "ani2x"


@dataclass
class AutoscaleANI2(Options):
    neighborlist: str = "full_pairwise"
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
    # Also accepts explicit numbers, 5 items -> 4 layers
    layer_dims: str = "in,1/4,2/3,1/2,1"
    activation: str = "gelu"
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
    factor: float = 0.5  # gamma
    interval: int = 150  # step_size in epochs


@dataclass
class ExponentialLR(SchedulerArgs):
    factor: float = 0.99  # gamma


# max_epochs is set to scheduler.max_epochs automatically
@dataclass
class CosineAnnealingLR(SchedulerArgs):
    total_epochs: int = 500
    final_lr: float = 0.0  # eta_min


@dataclass
class OptimizerArgs(Options):
    lr: float = 5e-4
    weight_decay: float = 1e-6


# eps for Adamax, Radam, Adam, AdamW is not shown, and it is always 1e-8
# Has no fused impl
@dataclass
class Adamax(OptimizerArgs):
    pass


# Has no fused impl
@dataclass
class Radam(OptimizerArgs):
    decoupled: bool = True  # decoupled_weight_decay


@dataclass
class Adam(OptimizerArgs):
    amsgrad: bool = False
    fused: bool = True  # Fused is not shown


@dataclass
class AdamW(Adam):
    pass


@dataclass
class SGD(OptimizerArgs):
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    fused: bool = True


def resolve_options(
    _options: tp.Optional[tp.List[str]], cls: str
) -> tp.Tuple[ScalarTuple, ...]:
    if _options is None:
        _options = []
    try:
        default_options_cls: Options = getattr(__name__, cls)
    except AttributeError:
        raise RuntimeError(f"Unknown class {cls}")
    _tp = default_options_cls.__annotations__
    default_options = default_options_cls.asdict()
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
            _v = kv.split("=")[1]
            try:
                v = _tp[k](_v)
            except Exception:
                raise RuntimeError(f"Incorrect value in option {kv}") from None
            options.update({k: v})
    default_options.update(options)
    return tuple(sorted((k, v) for k, v in default_options.items()))
