import warnings
import typing as tp

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torchani.models import BuiltinModel
    from torchani.assembler import (
        FlexANI1 as _FlexANI1,
        FlexANI2 as _FlexANI2,
        ANI1x as _ANI1x,
        ANI2x as _ANI2x,
        ANI1ccx as _ANI1ccx,
        ANIdr as _ANIdr,
        ANIala as _ANIala,
        ELEMENTS_1X,
        ELEMENTS_2X,
    )
from anitune.config import TrainConfig, ModelConfig
from anitune.utils import Scalar


def fetch_pretrained_config(name_or_idx: str) -> TrainConfig:
    name, idx = name_or_idx.split(":")
    config = TrainConfig()
    config.ds.fold_idx = idx
    if (("1x" in name) or ("1ccx" in name)):
        symbols = ELEMENTS_1X
    else:
        symbols = ELEMENTS_2X
    config.model = ModelConfig(
        arch_fn=name.replace("ani", "ANI"),
        arch_options=(("model_index", int(idx)),),
        symbols=symbols,
    )
    return config


class ArchAssembler:
    def __call__(
        self,
        lot: str,
        symbols: tp.Sequence[str],
        **kwargs: Scalar,
    ) -> BuiltinModel:
        raise NotImplementedError


class FlexANI1(ArchAssembler):
    def __call__(
        self,
        lot: str,
        symbols: tp.Sequence[str],
        **kwargs: Scalar,
    ) -> BuiltinModel:
        return _FlexANI1(
            lot=lot,
            symbols=symbols,
            **kwargs,
        )


class FlexANI2(ArchAssembler):
    def __call__(
        self,
        lot: str,
        symbols: tp.Sequence[str],
        **kwargs: Scalar,
    ) -> BuiltinModel:
        return _FlexANI2(
            lot=lot,
            symbols=symbols,
            **kwargs,
        )


class ANI1x(ArchAssembler):
    def __call__(
        self,
        lot: str,
        symbols: tp.Sequence[str],
        **kwargs: Scalar,
    ) -> BuiltinModel:
        return _ANI1x(**kwargs)


class ANI2x(ArchAssembler):
    def __call__(
        self,
        lot: str,
        symbols: tp.Sequence[str],
        **kwargs: Scalar,
    ) -> BuiltinModel:
        return _ANI2x(**kwargs)


class ANI1ccx(ArchAssembler):
    def __call__(
        self,
        lot: str,
        symbols: tp.Sequence[str],
        **kwargs: Scalar,
    ) -> BuiltinModel:
        return _ANI1ccx(**kwargs)


class ANIdr(ArchAssembler):
    def __call__(
        self,
        lot: str,
        symbols: tp.Sequence[str],
        **kwargs: Scalar,
    ) -> BuiltinModel:
        return _ANIdr(**kwargs)


class ANIala(ArchAssembler):
    def __call__(
        self,
        lot: str,
        symbols: tp.Sequence[str],
        **kwargs: Scalar,
    ) -> BuiltinModel:
        return _ANIala(**kwargs)
