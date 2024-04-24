import typing as tp
from pathlib import Path
DATA_ELEMENTS = {
    "ANI1x": ("H", "C", "N", "O"),
    "ANIExCorr": ("H", "C", "N", "O", "S"),
    "ANI1ccx": ("H", "C", "N", "O"),
    "TestData": ("H", "C", "N", "O"),
    "ANI2x": ("H", "C", "N", "O", "F", "S", "Cl"),
}


def load_state_dict(path: Path) -> tp.Dict[str, tp.Any]:
    r"""
    Load a model's state dict either from a torch .pt file, or a lightning
    .ckpt file (it is assumed that the model is located in a .model attribute
    in the LightningModule in the latter case).
    """
    import torch  # noqa

    _state_dict = torch.load(path)
    if "state_dict" in _state_dict:
        _state_dict = _state_dict["state_dict"]
        return {
            k.replace("model.", ""): v
            for k, v in _state_dict.items()
            if k.startswith("model.")
        }
    return _state_dict
