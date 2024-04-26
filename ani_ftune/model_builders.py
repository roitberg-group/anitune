import torch
import functools
import typing as tp
from torchani.models import BuiltinModel
from torchani.assembler import (
    FlexibleANI as _FlexibleANI,
    ANI1x as _ANI1x,
    ANI2x as _ANI2x,
    ANI1ccx as _ANI1ccx,
    ANIdr as _ANIdr,
    ANIala as _ANIala,
)
from ani_ftune.config import TrainConfig, ModelConfig


def fetch_pretrained_config(name_or_idx: str) -> TrainConfig:
    name, idx = name_or_idx.split(":")
    config = TrainConfig()
    config.ds.fold_idx = idx
    config.model = ModelConfig(
        builder=name.replace("ani", "ANI"),
        symbols=(),
        kwargs=(("model_index", int(idx)),),
    )
    return config


def FlexibleANI(**kwargs: tp.Any) -> BuiltinModel:
    return _FlexibleANI(**kwargs)


def ANI1x(lot: str, symbols: tp.Sequence[str], **kwargs: tp.Any) -> BuiltinModel:
    return _ANI1x(**kwargs)


def ANI2x(lot: str, symbols: tp.Sequence[str], **kwargs: tp.Any) -> BuiltinModel:
    return _ANI2x(**kwargs)


def ANI1ccx(lot: str, symbols: tp.Sequence[str], **kwargs: tp.Any) -> BuiltinModel:
    return _ANI1ccx(**kwargs)


def ANIdr(lot: str, symbols: tp.Sequence[str], **kwargs: tp.Any) -> BuiltinModel:
    return _ANIdr(**kwargs)


def ANIala(lot: str, symbols: tp.Sequence[str], **kwargs: tp.Any) -> BuiltinModel:
    return _ANIala(**kwargs)


def ReactionANI(
    lot: str,  # functional-basis
    symbols: tp.Sequence[str],
    ensemble_size: int = 1,
    radial_cutoff: float = 5.2,
    angular_cutoff: float = 3.5,
    radial_shifts: int = 16,
    angular_shifts: int = 4,
    angle_sections: int = 8,
    neighborlist: str = "full_pairwise",
    repulsion: bool = True,
    dispersion: bool = False,
    use_cuda_ops: bool = False,
    periodic_table_index: bool = True,
) -> BuiltinModel:
    from torchani.potentials import RepulsionXTB
    from torchani import atomics
    from torchani.assembler import Assembler, _parse_cuda_ops
    from torchani.aev import AEVComputer, StandardRadial, StandardAngular

    asm = Assembler(
        ensemble_size=ensemble_size, periodic_table_index=periodic_table_index
    )
    asm.set_symbols(symbols)
    asm.set_global_cutoff_fn("smooth2")

    asm.set_featurizer(
        AEVComputer,
        radial_terms=StandardRadial.cover_linearly(
            start=0.9,
            cutoff=radial_cutoff,
            eta=16.0,
            num_shifts=radial_shifts,
        ),
        angular_terms=StandardAngular.cover_linearly(
            start=0.9,
            eta=12.5,
            zeta=32.0,
            num_shifts=angular_shifts,
            num_angle_sections=angle_sections,
            cutoff=angular_cutoff,
        ),
        extra=_parse_cuda_ops(use_cuda_ops),
    )
    asm.set_atomic_maker(
        functools.partial(atomics.like_1x, activation=torch.nn.GELU(), bias=False)
    )
    asm.set_neighborlist(neighborlist)
    asm.set_gsaes_as_self_energies(lot)
    if repulsion:
        asm.add_pairwise_potential(
            RepulsionXTB,
            cutoff=radial_cutoff,
        )
    return asm.assemble()
