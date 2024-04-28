from enum import Enum
import typing as tp
from pathlib import Path


class DisambiguationError(RuntimeError):
    pass


class DiskData(Enum):
    TRAIN = "train"
    FTUNE = "ftune"
    DEBUG_TRAIN = "debug-train"
    DEBUG_FTUNE = "debug-ftune"
    BATCH = "batch"


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
