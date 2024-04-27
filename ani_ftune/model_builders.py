import typing as tp
from torchani.models import BuiltinModel
from torchani.assembler import (
    FlexANI1 as _FlexANI1,
    FlexANI2 as _FlexANI2,
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


def FlexANI1(**kwargs: tp.Any) -> BuiltinModel:
    return _FlexANI1(**kwargs)


def FlexANI2(**kwargs: tp.Any) -> BuiltinModel:
    return _FlexANI2(**kwargs)


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
