# type: ignore
import time
import csv
import warnings
from pathlib import Path
from typing import Optional, Union, Dict, Any
from collections import OrderedDict

import torch
from torch.optim import Optimizer
import torch.utils.tensorboard
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from ruamel.yaml import YAML

import torchani

from .runners import Runner
from .utils import _calculate_hash

try:
    import wandb

    WANDB_AVAIL = True
except ImportError:
    WANDB_AVAIL = False

# Mypy
PathLike = Union[str, Path]
ScalarMetrics = Dict[str, float]
LRScheduler = Union[ReduceLROnPlateau, _LRScheduler]
Stateful = Union[torch.nn.Module, torchani.models.BuiltinModel, LRScheduler, Optimizer]


class Logger:
    def __init__(
        self,
        path: Optional[PathLike] = None,
        log_tensorboard: bool = True,
        log_csv: bool = True,
        log_wandb: bool = True,
        debug: bool = False,
    ):
        path = (
            Path(path).resolve()
            if path is not None
            else Path("./runs/default_set/default_run").resolve()
        )
        self._debug = debug
        if not debug:
            assert path.is_dir()
        self._log_tb = log_tensorboard
        self._log_csv = log_csv
        self._log_wandb = log_wandb
        if log_tensorboard:
            self._tb_writer = torch.utils.tensorboard.SummaryWriter(path)
        if log_csv:
            self._csv_train_path = path / "metrics_train.csv"
            self._csv_validate_path = path / "metrics_validate.csv"
        if log_wandb:
            assert WANDB_AVAIL
            if not debug:
                wandb.init(
                    project="qrnn-nacho", name=path.name, entity="roitberg-group"
                )

    def wandb_watch(self, model, *args, **kwargs):
        wandb.watch(model, *args, **kwargs)

    def log_scalars(
        self,
        step: int,
        train_metrics: Optional[ScalarMetrics] = None,
        validate_metrics: Optional[ScalarMetrics] = None,
        other: Optional[ScalarMetrics] = None,
    ) -> None:
        if self._debug:
            return
        train_metrics = {} if train_metrics is None else train_metrics
        validate_metrics = {} if validate_metrics is None else validate_metrics
        other = {} if other is None else other
        train_total_factor = sum(v for k, v in train_metrics.items() if "factor" in k)
        for k in list(train_metrics.keys()):
            if "factor" in k:
                train_metrics[k + "_relative"] = train_metrics[k] / train_total_factor

        if self._log_wandb:
            for k, v in train_metrics.items():
                wandb.log({f"{k}/train": v}, step=step)
            for k, v in validate_metrics.items():
                wandb.log({f"{k}/validate": v}, step=step)
            for k, v in other.items():
                wandb.log({k: v}, step=step)

        if self._log_tb:
            for k, v in train_metrics.items():
                self._tb_writer.add_scalar(f"{k}/train", v, step)
            for k, v in validate_metrics.items():
                self._tb_writer.add_scalar(f"{k}/validate", v, step)
            for k, v in other.items():
                self._tb_writer.add_scalar(k, v, step)

        if self._log_csv:
            if train_metrics:
                train_metrics.update(other)
                self._dump_csv_metrics(
                    step, self._csv_train_path, train_metrics, first_step=1
                )
            if validate_metrics:
                validate_metrics.update(other)
                self._dump_csv_metrics(
                    step, self._csv_validate_path, validate_metrics, first_step=0
                )

    def _dump_csv_metrics(
        self, step: int, path: PathLike, metrics: ScalarMetrics, first_step: int
    ):
        metrics = OrderedDict(sorted(metrics.items()))
        row = [step] + list(metrics.values())
        with open(path, "a") as f:
            writer = csv.writer(f, delimiter=",")
            if step == first_step:
                writer.writerow(["epoch"] + sorted(metrics.keys()))
                for j in range(0, first_step):
                    writer.writerow([j] + [""] * len(metrics.keys()))
            writer.writerow(row)


# Checkpointing
def _ensure_state_dicts(objects: Dict[str, Stateful]):
    for v in objects.values():
        assert hasattr(v, "state_dict")
        assert hasattr(v, "load_state_dict")


def _save_checkpoint(
    path: PathLike,
    objects: Dict[str, Stateful],
    kind: str = "default",
    debug: bool = False,
) -> None:
    _ensure_state_dicts(objects)
    if debug:
        return
    for k, v in objects.items():
        torch.save(v.state_dict(), Path(path).resolve() / f"{k}_{kind}.pt")


def _load_checkpoint(
    path: PathLike, objects: Dict[str, Stateful], kind: str = "default"
) -> None:
    _ensure_state_dicts(objects)
    for k in objects.keys():
        objects[k].load_state_dict(torch.load(path / f"{k}_{kind}.pt"))


def prepare_learning_sets(
    data_config: Dict[str, Any],
    splits: Optional[int],
    folds: Optional[int],
    root_dataset_path: Path,
    training_split: str,
    validation_split: str,
):
    DatasetClass = getattr(torchani.datasets, data_config["class"])
    batch_all_properties = data_config["batch_all_properties"]
    functional = data_config.get("functional", None)
    basis_set = data_config.get("basis_set", None)
    dummy_properties = data_config.get("dummy_properties", None)
    selected_properties = data_config.get("selected_properties", None)
    assert (splits is None or folds is None) and (splits is not folds)
    ds_path = root_dataset_path.joinpath(data_config["dataset_name"])
    include_properties = None if batch_all_properties else selected_properties
    shuffle_seed = 123456789
    # If the batched dataset path is given, then it is assumed that the
    # necessary dataset exists, to make sure the dataset that would be built is the same
    # as the current dataset we append a hash to the dataset path that encodes the build
    # information. The hash must store:
    # shuffle_seed, splits, folds, batch_size, include_properties, hashes of
    # component datasets
    if DatasetClass.__name__ == "ANIDataset":
        kwargs = dict()
        if dummy_properties is not None:
            kwargs.update({"dummy_properties": dummy_properties})
        ds = DatasetClass.from_dir(ds_path, **kwargs)
    else:
        kwargs = {"download": True}
        if functional is not None:
            kwargs.update({"functional": functional})
        if basis_set is not None:
            kwargs.update({"basis_set": basis_set})
        if dummy_properties is not None:
            kwargs.update({"dummy_properties": dummy_properties})
        ds = DatasetClass(ds_path, **kwargs)
    digest = _calculate_hash(
        folds, splits, sorted(ds.store_md5s), shuffle_seed, include_properties
    )

    if splits is not None:
        batched_dataset_path = Path(ds_path.as_posix() + f"-batched-{digest}").resolve()
    else:
        batched_dataset_path = Path(
            ds_path.as_posix() + f"-ensemble-batched-{digest}"
        ).resolve()

    if not batched_dataset_path.is_dir():
        torchani.datasets.create_batched_dataset(
            ds,
            include_properties=include_properties,
            dest_path=batched_dataset_path,
            batch_size=data_config["batch_size"],
            max_batches_per_packet=1500,
            shuffle_seed=shuffle_seed,
            splits=splits,
            folds=folds,
        )

    training_set = torch.utils.data.DataLoader(
        torchani.datasets.ANIBatchedDataset(
            batched_dataset_path, split=training_split, properties=selected_properties
        ),
        shuffle=True,
        num_workers=data_config["num_workers"],
        prefetch_factor=data_config["prefetch_factor"],
        pin_memory=True,
        batch_size=None,
    )

    validation_set = torch.utils.data.DataLoader(
        torchani.datasets.ANIBatchedDataset(
            batched_dataset_path, split=validation_split, properties=selected_properties
        ),
        shuffle=False,
        num_workers=data_config["num_workers"],
        prefetch_factor=data_config["prefetch_factor"],
        pin_memory=True,
        batch_size=None,
    )
    return training_set, validation_set


def execute_training(
    persistent_objects: Dict[str, Stateful],
    scheduler: LRScheduler,
    runner: Runner,
    optimizer: Optimizer,
    training_set,
    validation_set,
    run_output_path: Path,
    config: Dict[str, Any],
    debug=False,
):
    # Get general training parameters
    max_epochs = config["general"]["max_epochs"]
    early_stopping_lr = config["general"]["early_stopping_lr"]
    track_metric = config["general"]["track_metric"]

    # Load latest checkpoint if it exists
    # First we check the output path and see if it has any checkpoint files,
    # if it doesn't then this means the previous run failed even before writing any
    # checkpoint, so we erase everything we usually write to these files and move on
    if run_output_path.is_dir():
        files_in_path = list(run_output_path.iterdir())
        if not any(f.name.endswith("latest.pt") for f in files_in_path):
            for p in files_in_path:
                if p.name.startswith("events") or p.name in [
                    "train_metrics.csv",
                    "validate_metrics.csv",
                    "config.yaml",
                ]:
                    p.unlink()

    # If the run is not debug then we may have to load a checkpoint file, if
    # one exists, in this case we make sure that our configuration file matches
    # the configuration file from the checkpoint, otherwise we create the
    # output path and dump our configuration file there. DEBUG runs don't write
    # anything so these steps are not important for those.
    is_restart = False
    if debug:
        warnings.warn(
            "Running in DEBUG mode, checkpoints and tensorboard files will not be saved!!!"
        )
    else:
        yaml = YAML(typ="rt")
        if run_output_path.is_dir() and any(run_output_path.iterdir()):
            is_restart = True
            _load_checkpoint(run_output_path, persistent_objects, kind="latest")
            restart_config = yaml.load(run_output_path / "config.yaml")
            if not {k: v for k, v in restart_config.items()} == {
                k: v for k, v in config.items()
            }:
                raise RuntimeError("Tried to run restart with a modified config")

        else:
            run_output_path.mkdir(parents=True, exist_ok=True)
            yaml.dump(config, run_output_path / "config.yaml")
    logger = Logger(run_output_path, debug=debug, **config["logger"])
    wandb_watch_args = config.get("wandb", None)
    if wandb_watch_args is not None:
        logger.wandb_watch(runner._model, **wandb_watch_args)

    # Main Training loop
    initial_epoch = scheduler.last_epoch  # type: ignore
    if not is_restart and initial_epoch == 0:  # Zeroth epoch is just validating
        validate_metrics = runner.eval(
            validation_set, initial_epoch, track_metric=track_metric
        )
        initial_lr = optimizer.param_groups[0]["lr"]
        logger.log_scalars(
            initial_epoch,
            validate_metrics=validate_metrics,
            other={"learning_rate": initial_lr, "epoch_time_seconds": 0.0},
        )
        _save_checkpoint(
            run_output_path, persistent_objects, kind="latest", debug=debug
        )

    for epoch in range(initial_epoch + 1, max_epochs):
        start = time.time()
        # Run training and validation
        train_metrics = runner.train(training_set, epoch)
        validate_metrics = runner.eval(validation_set, epoch, track_metric=track_metric)
        # LR Scheduler update
        metric = (
            (validate_metrics[track_metric],)
            if isinstance(scheduler, ReduceLROnPlateau)
            else tuple()
        )
        scheduler.step(*metric)
        # Checkpoint
        if runner.best_metric_improved_last_run:
            runner.best_metric_improved_last_run = False
            _save_checkpoint(
                run_output_path, persistent_objects, kind="best", debug=debug
            )
        _save_checkpoint(
            run_output_path, persistent_objects, kind="latest", debug=debug
        )
        # Logging
        learning_rate = optimizer.param_groups[0]["lr"]
        logger.log_scalars(
            scheduler.last_epoch,  # type: ignore
            train_metrics=train_metrics,
            validate_metrics=validate_metrics,
            other={
                "learning_rate": learning_rate,
                "epoch_time_seconds": time.time() - start,
            },
        )
        # Early stopping
        if learning_rate < early_stopping_lr:
            break
