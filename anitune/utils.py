import typing as tp
from enum import Enum
from pathlib import Path

Scalar = tp.Union[bool, int, float, str, None]
ScalarTuple = tp.Tuple[str, tp.Union[bool, int, float, str, None]]


class DiskData(Enum):
    TRAIN = "train"
    FTUNE = "ftune"
    BATCH = "batch"
    ENSEMBLE = "ensemble"


_ENSEMBLE_PATH = Path.home().joinpath(".local/torchani/Ensemble")
_ENSEMBLE_PATH.mkdir(exist_ok=True, parents=True)

_BATCH_PATH = Path.home().joinpath(".local/torchani/Batched")
_BATCH_PATH.mkdir(exist_ok=True, parents=True)

_TRAIN_PATH = Path.home().joinpath(".local/torchani/Train")
_TRAIN_PATH.mkdir(exist_ok=True, parents=True)

_FTUNE_PATH = Path.home().joinpath(".local/torchani/Ftune")
_FTUNE_PATH.mkdir(exist_ok=True, parents=True)


class DisambiguationError(RuntimeError):
    pass


def select_paths(
    names_or_idxs: tp.Iterable[str],
    kind: DiskData = DiskData.TRAIN,
) -> tp.List[Path]:
    root: Path
    if kind is DiskData.TRAIN:
        root = _TRAIN_PATH
    elif kind is DiskData.FTUNE:
        root = _FTUNE_PATH
    elif kind is DiskData.ENSEMBLE:
        root = _ENSEMBLE_PATH
    elif kind is DiskData.BATCH:
        root = _BATCH_PATH
    else:
        raise ValueError("Incorrect disk data")
    sorted_paths = sorted(root.iterdir())
    selected_paths = []
    for name_or_idx in names_or_idxs:
        try:
            idx = int(name_or_idx)
            if idx > len(sorted_paths) or idx < 0:
                raise RuntimeError(f"Index {idx} invalid")
            selected_paths.append(sorted_paths[idx])
        except ValueError:
            paths = [p for p in sorted_paths if p.name.startswith(name_or_idx)]
            if not paths:
                raise RuntimeError(
                    f"No paths starting with name {name_or_idx} found"
                ) from None
            elif len(paths) > 1:
                raise DisambiguationError(
                    f"More than one path starts with {name_or_idx}"
                ) from None
            else:
                selected_paths.append(paths[0])
    return selected_paths
