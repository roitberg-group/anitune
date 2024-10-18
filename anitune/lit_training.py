import logging
import json
import warnings
from copy import deepcopy
import pickle
import sys
import typing as tp

from rich.prompt import Confirm

from anitune.console import console
from anitune.config import TrainConfig


def train_lit_model(
    config: TrainConfig,
    restart: bool = False,
    verbose: bool = False,
) -> None:
    r"""
    Train an ANI-style neural network potential using PyTorch Lightning
    """
    import torch
    import lightning

    if not verbose:
        from lightning_utilities.core.rank_zero import log

        log.setLevel(logging.ERROR)
    from lightning.pytorch.callbacks import (
        LearningRateMonitor,
        EarlyStopping,
        BackboneFinetuning,
    )
    from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

    from torchani import assembly, models, datasets
    from anitune import losses
    from anitune.lit_model import LitModel
    from anitune.lit_callbacks import (
        SaveConfig,
        ModelCheckpointWithMetrics,
    )

    if not config.ds.path.exists():
        raise RuntimeError("Dataset does not exist")

    with open(config.ds.path / "creation_log.json", mode="rt", encoding="utf-8") as f:
        ds_symbols = json.load(f)["symbols"]
    if not config.model.builtin:
        model = getattr(assembly, config.model.arch_fn)(
            lot=config.ds.lot,
            symbols=config.model.symbols or ds_symbols,
            **config.model.options,
        )
    else:
        model = getattr(models, config.model.arch_fn)(**config.model.options)
    if config.ftune is not None:
        if config.ftune.pretrained_state_dict:
            model.load_state_dict(config.ftune.pretrained_state_dict)

    ckpt_path = (config.path / "latest-model") / "latest.ckpt"
    if not restart and config.path.is_dir():
        if not Confirm.ask("Run already exists, do you want to restart it?"):
            console.print("Exiting without training")
            sys.exit(0)
        else:
            # Reload config from the path
            path = config.path / "config.pkl"
            if not path.is_file():
                raise ValueError(f"{path} is not a config file")
            accel = deepcopy(config.accel)
            with open(path, mode="rb") as f:
                config = pickle.load(f)
            config.accel = accel
            restart = True

    if ckpt_path.is_file():
        # Not sure what the problem with mypy is here
        lit_model = LitModel.load_from_checkpoint(  # type: ignore
            ckpt_path,
            model=model,
        )
    else:
        lit_model = LitModel(
            model,
            loss_terms=tuple(
                getattr(losses, name)(factor=factor)
                for name, factor in config.loss.terms_and_factors
            ),
            monitor_label=config.monitor_label,
            # Loss
            uncertainty_weighted=config.loss.uncertainty_weighted,
            # Optim
            optimizer_cls=config.optim.cls,
            optimizer_options=config.optim.options,
            # Scheduler
            scheduler_cls=config.scheduler.cls,
            scheduler_options=config.scheduler.options,
            # Ftune
            num_head_layers=0 if config.ftune is None else config.ftune.num_head_layers,
        )

    if restart:
        console.print(f"Restarting run {config.path}")
    else:
        init_model_path = config.path / "init-model"
        init_model_path.mkdir(exist_ok=False, parents=True)
        torch.save(
            {"state_dict": lit_model.state_dict()}, init_model_path / "init.ckpt"
        )

    kwargs: tp.Dict[str, tp.Any] = {
        "num_workers": config.accel.num_workers,
        "prefetch_factor": config.accel.prefetch_factor,
        "pin_memory": True,
        "batch_size": None,
    }
    training_label = (
        f"training{config.ds.fold_idx if config.ds.fold_idx != 'train' else ''}"
    )
    validation_label = (
        f"validation{config.ds.fold_idx if config.ds.fold_idx != 'train' else ''}"
    )
    training = torch.utils.data.DataLoader(
        datasets.ANIBatchedDataset(config.ds.path, split=training_label),
        shuffle=True,
        **kwargs,
    )
    validation = torch.utils.data.DataLoader(
        datasets.ANIBatchedDataset(config.ds.path, split=validation_label),
        shuffle=False,
        **kwargs,
    )

    lr_monitor = LearningRateMonitor()
    early_stopping = EarlyStopping(
        monitor=lit_model.monitor_label,
        strict=True,
        mode="min",
        patience=config.accel.early_stop_patience,  # type: ignore
    )
    best_model_ckpt = ModelCheckpointWithMetrics(
        dirpath=config.path / "best-model",
        filename="best",
        save_top_k=1,
        enable_version_counter=False,
        # Specific configuration for saving the best model
        monitor=lit_model.monitor_label,
        mode="min",
        save_weights_only=True,
    )
    latest_model_ckpt = ModelCheckpointWithMetrics(
        dirpath=config.path / "latest-model",
        filename="latest",
        save_top_k=1,
        enable_version_counter=False,
    )
    save_model_config = SaveConfig(config)
    callbacks = [
        lr_monitor,
        early_stopping,
        best_model_ckpt,
        latest_model_ckpt,
        save_model_config,
    ]
    (config.path / "tb-logs").mkdir(exist_ok=True, parents=True)
    tb_logger = TensorBoardLogger(
        save_dir=config.path,
        name="tb-logs",
        default_hp_metric=False,
    )
    (config.path / "csv-logs").mkdir(exist_ok=True, parents=True)
    csv_logger = CSVLogger(
        save_dir=config.path,
        name="csv-logs",
    )
    loggers = [tb_logger, csv_logger]

    # Finetuning configuration
    if config.ftune is not None:
        if config.ftune.frozen_backbone:
            unfreeze_epoch = config.accel.max_epochs + 1
        else:
            unfreeze_epoch = 0
        ftune_callback = BackboneFinetuning(
            lambda_func=lambda epoch: 1.0,
            backbone_initial_lr=config.ftune.backbone_lr,
            unfreeze_backbone_at_epoch=unfreeze_epoch,
            should_align=False,
            train_bn=False,
            verbose=False,
        )
        callbacks.append(ftune_callback)
    trainer = lightning.Trainer(
        default_root_dir=config.path,
        devices=1,
        accelerator=config.accel.device,
        max_epochs=config.accel.max_epochs,
        logger=loggers,
        callbacks=callbacks,
        limit_train_batches=config.accel.train_limit,
        limit_val_batches=config.accel.validation_limit,
        log_every_n_steps=config.accel.log_interval,
        deterministic=config.accel.deterministic,
        detect_anomaly=config.accel.detect_anomaly,
        profiler=config.accel.profiler,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message="Checkpoint directory.*",
            category=UserWarning,
        )
        trainer.fit(
            lit_model,
            train_dataloaders=training,
            val_dataloaders=validation,
            ckpt_path=ckpt_path if ckpt_path.is_file() else None,
        )
