# type: ignore
import shutil
from pathlib import Path
from typing import Sequence, Union, Dict
from collections import OrderedDict

import torch
from torch import Tensor

import torchani
from torchani.aev.cutoffs import CutoffSmooth

import builders


def combine_state_dicts(
    state_dicts: Union[str, Path, Sequence[Union[str, Path]]],
    tag: str = "neural_networks",
) -> "OrderedDict[str, Tensor]":
    r"""Takes state_dicts of individual ani
    models and assembles one state dict for an ensemble from them

    It can accept either an iterable of paths to state dicts, or a single
    path to a directory that it assumes contains only state_dicts

    This works for individual models that have one or more (not nested)
    submodules with "neural_networks" in their name (or the string passed
    to tag). It relables the keys:

    (...).(...)tag(...).(...)

    into:

    (...).(...)tag(...).0.(...), (...).(...)tag(...).1.(...), etc

    for the different state dicts passed, all the state dicts are then
    merged into one unique state dict which is returned.

    Arguments:
        state_dicts: string, Path or iterable of string / paths of the
            state_dicts for the individual members of the ensemble
        tag: A part of the name that all the modules that need
            ensembling have, by default it is "neural_networks"
    Returns:
        OrderedDict: A well formed state_dict for an ensemble of models
    """
    if isinstance(state_dicts, (str, Path)):
        state_dicts = Path(state_dicts).resolve()
        assert state_dicts.is_dir()
        state_dicts_seq = [
            file_ for file_ in state_dicts.iterdir() if file_.suffix in (".pt", ".pth")
        ]
    else:
        # Ensure that these are paths
        state_dicts_seq = [Path(s).resolve() for s in state_dicts]

    def _load(path: Path) -> Dict[str, Tensor]:
        try:
            checkpoint = torch.load(path)["model"]
        except KeyError:
            checkpoint = torch.load(path)
        return checkpoint

    checkpoints = [_load(p) for p in state_dicts_seq]
    # All keys besides ones that have the tag in them should be equal
    common_keys = {k for od in checkpoints for k in od.keys() if tag not in k}
    ensembling_keys = {k for od in checkpoints for k in od.keys() if tag in k}
    ensemble_state_dict = {k: checkpoints[0][k] for k in common_keys}
    ensemble_state_dict = OrderedDict({k: checkpoints[0][k] for k in common_keys})

    # Check that all common keys are equal
    for j, od in enumerate(checkpoints[1:]):
        for k in common_keys:
            if (checkpoints[0][k] != od[k]).any():
                raise ValueError(
                    f"The value for {k} is different in {state_dicts_seq[0].as_posix()} and {state_dicts_seq[j].as_posix()}"
                )

    # add index for the ensemble after name of tagged modules
    for j, od in enumerate(checkpoints):
        for k in ensembling_keys:
            tokens = k.split(".")
            ensembling_names = [t for t in tokens if tag in t]
            if len(ensembling_names) != 1:
                raise ValueError(
                    "Modules to be ensembled can't be nested, error in {k}"
                )
            ensembling_name = ensembling_names[0]
            tokens.insert(tokens.index(ensembling_name) + 1, str(j))
            ensemble_state_dict[".".join(tokens)] = od[k].clone()
        del od

    return ensemble_state_dict


# Build a model like the one built during training
elements = ("H", "C", "N", "O", "S", "F", "Cl")
model = torchani.models.ANI2x(
    pretrained=False,
    cutoff_fn="smooth",
    periodic_table_index=True,
    atomic_maker=builders.dispersion_atomics,
    ensemble_size=7,
    dispersion=True,
    dispersion_kwargs={
        "elements": elements,
        "cutoff": 8.0,
        "cutoff_fn": CutoffSmooth(order=2),
        "functional": "B97-3c",
    },
    repulsion=True,
    repulsion_kwargs={
        "elements": elements,
        "cutoff": 5.1,
        "cutoff_fn": CutoffSmooth(order=2),
    },
)
model_state_dict = model.state_dict()
model_keys = set(model_state_dict.keys())

path = Path(
    "/home/ignacio/RemoteMachines/hpg-blue/torchani-runs/trials-processed-with-bb-ensemble"
)
best_checkpoints = []

for p in sorted(path.iterdir()):
    if not p.is_dir():
        continue
    for f in sorted(p.iterdir()):
        if "model_best" == f.stem:
            best_checkpoints.append(f)

for f in best_checkpoints:
    dst_path = Path("./ensemble").resolve()
    dst_path.mkdir(exist_ok=True, parents=True)
    shutil.copy(f, dst_path / f"{f.stem}_{f.parent.name[-1]}.pt")

ensemble_state_dict = combine_state_dicts(
    sorted(p for p in dst_path.iterdir() if not p.stem.startswith("ensemble"))
)
ensemble_keys = set(ensemble_state_dict.keys())

for k, v in model_state_dict.items():
    if k not in ensemble_keys:
        ensemble_state_dict.update({k: v})

torch.save(ensemble_state_dict, dst_path / "ensemble_anid_reacting_state_dict.pt")
