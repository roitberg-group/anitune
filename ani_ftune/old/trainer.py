# type: ignore
import socket
from pathlib import Path
from itertools import chain
from typing import Tuple, Any, Dict
import argparse

import torch
import torch.utils.tensorboard
from torch.optim import lr_scheduler
from ruamel.yaml import YAML

from torchani.utils import sorted_gsaes

from . import runners
from . import builders
from . import core
from .locations import locations

hostname = socket.gethostname()
if "." in hostname:
    hostname = hostname.split(".")[-1]
locations = locations[hostname]


def init_params(m):
    pass


def extract_class_parameters(
    config,
    module,
    key_name: str,
    extra_args: Tuple[Any, ...] = tuple(),
    extra_kwargs: Dict[str, Any] = None,
):
    extra_kwargs = dict() if extra_kwargs is None else extra_kwargs
    obj_args = config[key_name].copy()
    obj_class = obj_args.pop("class")
    extra_kwargs.update(obj_args)
    return getattr(module, obj_class)(*extra_args, **extra_kwargs)


def main():
    parser = argparse.ArgumentParser(prog="ANI Train")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-num", default=-1, type=int)
    parser.add_argument(
        "--config", required=True, help="Configuration file for the run"
    )
    args = parser.parse_args()
    print("Running on device", torch.cuda.get_device_name())
    yaml = YAML(typ="rt")
    config = yaml.load(Path(args.config).resolve())

    model_num = args.model_num
    # General
    DEVICE = args.device
    DEBUG = args.debug  # In debug runs nothing is logged or saved

    # Paths and names
    DATASET_ROOT_DIR = Path(locations["datasets"])
    OUTPUT_ROOT_DIR = locations["runs"]
    DATASET_NAME = config["data"]["dataset_name"]
    RUN_NAME = config["general"]["run_name"]
    SET_NAME = f"trials-{DATASET_NAME}"

    if model_num != -1:
        RUN_NAME = RUN_NAME + f"-{model_num}"

    # Data config
    FOLDS = None if model_num == -1 else 8
    if model_num == -1:
        data_splits = config["data"].get("splits", None)
        SPLITS = (
            data_splits
            if data_splits is not None
            else {"training": 0.8, "validation": 0.2}
        )
    else:
        SPLITS = None
    VALIDATION_SPLIT = f"validation"
    TRAINING_SPLIT = f"training"
    if model_num != -1:
        VALIDATION_SPLIT = VALIDATION_SPLIT + str(model_num)
        TRAINING_SPLIT = TRAINING_SPLIT + str(model_num)
    training_set, validation_set = core.prepare_learning_sets(
        config["data"],
        SPLITS,
        FOLDS,
        DATASET_ROOT_DIR,
        TRAINING_SPLIT,
        VALIDATION_SPLIT,
    )
    # Model and atomic networks
    try:
        atomic_maker = getattr(builders, config["atomic_maker"]["class"])
        extra_kwargs = {"atomic_maker": atomic_maker}
    except KeyError:
        extra_kwargs = dict()
    use_gsaes = config["model"].pop("use_gsaes", True)
    model = extract_class_parameters(
        config, builders, "model", extra_kwargs=extra_kwargs
    )
    model.apply(init_params)
    print(model)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    # GSAEs
    if use_gsaes:
        gsaes = sorted_gsaes(
            model.get_chemical_symbols(),
            functional=config["data"]["functional"],
            basis_set=config["data"]["basis_set"],
        )
        model.energy_shifter.self_energies = torch.tensor(gsaes, dtype=torch.float)
    # probably not needed but torch.optim documentation asks for this
    model = model.to(DEVICE)

    # Runner / Scheduler / Optimizer
    opt_params = [model.parameters()]
    if "uw_loss" in config.keys():
        uw_loss = builders.UncertaintyWeightedLoss(config["uw_loss"])
        opt_params.append(uw_loss.parameters())
    opt_params = chain.from_iterable(opt_params)
    optimizer = extract_class_parameters(
        config, torch.optim, "optimizer", extra_args=(opt_params,)
    )
    scheduler = extract_class_parameters(
        config, lr_scheduler, "scheduler", extra_args=(optimizer,)
    )
    runner_kwargs = {"model": model, "optimizer": optimizer, "device": DEVICE}
    if "uw_loss" in config.keys():
        runner_kwargs.update({"uw_loss": uw_loss})
    runner = extract_class_parameters(
        config, runners, "runner", extra_kwargs=runner_kwargs
    )

    # Checkpoint paths
    run_output_path = Path(f"{OUTPUT_ROOT_DIR}/{SET_NAME}/{RUN_NAME}/").resolve()
    persistent_objects = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "model": model,
        "runner": runner,
    }
    if "uw_loss" in config.keys():
        persistent_objects.update({"uw_loss": uw_loss})
    core.execute_training(
        persistent_objects,
        scheduler,
        runner,
        optimizer,
        training_set,
        validation_set,
        run_output_path,
        config,
        debug=DEBUG,
    )


if __name__ == "__main__":
    main()
