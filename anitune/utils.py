import typing as tp
from enum import Enum
from pathlib import Path


class DiskData(Enum):
    TRAIN = "train"
    FTUNE = "ftune"
    DEBUG_TRAIN = "debug-train"
    DEBUG_FTUNE = "debug-ftune"
    BATCH = "batch"


_BATCH_PATH = Path.home().joinpath(".local/torchani/Batched")
_BATCH_PATH.mkdir(exist_ok=True, parents=True)

_TRAIN_PATH = Path.home().joinpath(".local/torchani/Train")
_TRAIN_PATH.mkdir(exist_ok=True, parents=True)

_DEBUG_TRAIN_PATH = Path.home().joinpath(".local/torchani/DebugTrain")
_DEBUG_TRAIN_PATH.mkdir(exist_ok=True, parents=True)

_DEBUG_FTUNE_PATH = Path.home().joinpath(".local/torchani/DebugFtune")
_DEBUG_FTUNE_PATH.mkdir(exist_ok=True, parents=True)

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
    elif kind is DiskData.DEBUG_FTUNE:
        root = _DEBUG_FTUNE_PATH
    elif kind is DiskData.DEBUG_TRAIN:
        root = _DEBUG_TRAIN_PATH
    else:
        root = _BATCH_PATH
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


def simplify_metric(arg: str) -> str:
    return (
        arg
        .replace("valid_", "")
        .replace("train_", "")
        .replace("mae_energies", "Eₘₐₑ")
        .replace("rmse_energies", "E")
        .replace("mae_forces", "Fₘₐₑ")
        .replace("rmse_forces", "F")
        .replace("|ang", "/Å")
        .replace("_kcal|mol", "㎉㏖")
    )
