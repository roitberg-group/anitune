import typing as tp

from ani_ftune.exceptions import ConfigError
from ani_ftune.configuration import TrainConfig


def train_from_scratch(config: TrainConfig) -> None:
    import torch  # noqa
    import lightning  # noqa
    from lightning.pytorch.callbacks import (  # noqa
        LearningRateMonitor,
        EarlyStopping,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

    from torchani import datasets  # noqa
    from ani_ftune.lit_models import LitModel  # noqa
    from ani_ftune import model_builders  # noqa
    from ani_ftune import losses  # noqa
    from ani_ftune.callbacks import MergeTensorBoardLogs

    model = getattr(model_builders, config.model.builder)(
        lot=config.ds.lot,
        symbols=config.model.get_symbols(
            config.ds.name,
            basis_set=config.ds.basis_set,
            functional=config.ds.functional,
        ),
        use_cuda_ops=config.accel.use_cuda_ops,
        **config.model.flag_dict,
    )

    ckpt_path = (config.path / "latest-model") / "latest.ckpt"
    if ckpt_path.is_file():
        lit_model = LitModel.load_from_checkpoint(ckpt_path, model=model)
    else:
        lit_model = LitModel(
            model,
            loss_terms=tuple(
                getattr(losses, name)(factor=factor)
                for name, factor in config.loss.terms_and_factors
            ),
            # Loss
            uncertainty_weighted=config.loss.uncertainty_weighted,
            # Optim
            weight_decay=config.optim.weight_decay,
            lr=config.optim.lr,
            # Scheduler
            monitor_label=config.scheduler.monitor_label,
            plateau_factor=config.scheduler.factor,
            plateau_patience=config.scheduler.patience,
            plateau_threshold=config.scheduler.threshold,
        )

    if not config.ds.path.exists():
        split_kwargs: tp.Dict[str, tp.Union[int, tp.Dict[str, float]]]
        if config.ds.folds is not None:
            if config.ds.train_frac != 0.8 or config.ds.validation_frac != 0.2:
                raise ConfigError("Train and val frac can't be set if training to folds")
            if not isinstance(config.ds.fold_idx, int):
                raise ConfigError("A fold idx must be present when training to folds")
            split_kwargs = {"folds": config.ds.folds}
        else:
            split_kwargs = {"splits": config.ds.split_dict}
        datasets.create_batched_dataset(
            locations=getattr(datasets, config.ds.name)(
                skip_check=True,
                functional=config.ds.functional,
                basis_set=config.ds.basis_set,
            ),
            max_batches_per_packet=config.accel.max_batches_per_packet,
            dest_path=config.ds.path,
            batch_size=config.ds.batch_size,
            shuffle_seed=config.ds.shuffle_seed,
            **split_kwargs,
        )

    kwargs: tp.Dict[str, tp.Any] = {
        "num_workers": config.accel.num_workers,
        "prefetch_factor": config.accel.prefetch_factor,
        "pin_memory": True,
        "batch_size": None,
    }
    training_label = f"training{config.ds.fold_idx if config.ds.fold_idx != 'single' else ''}"
    validation_label = f"validation{config.ds.fold_idx if config.ds.fold_idx != 'single' else ''}"
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
        patience=lit_model.hparams.plateau_patience * 2,  # type: ignore
    )
    best_model_ckpt = ModelCheckpoint(
        dirpath=config.path / "best-model",
        filename="best",
        save_top_k=1,
        enable_version_counter=False,
        # Specific configuration for saving the best model
        monitor=lit_model.monitor_label,
        mode="min",
        save_weights_only=True,
    )
    latest_model_ckpt = ModelCheckpoint(
        dirpath=config.path / "latest-model",
        filename="latest",
        save_top_k=1,
        enable_version_counter=False,
    )
    tb_logger = TensorBoardLogger(save_dir=config.path, version=None, name="tb-versioned-logs")
    csv_logger = CSVLogger(save_dir=config.path, version=None, name="csv-versioned-logs")
    merge_tb_logs = MergeTensorBoardLogs(src="tb-versioned-logs", dest="tb-logs")
    trainer = lightning.Trainer(
        default_root_dir=config.path,
        devices=1,
        accelerator=config.accel.device,
        max_epochs=config.accel.max_epochs,
        callbacks=[lr_monitor, early_stopping, best_model_ckpt, latest_model_ckpt, merge_tb_logs],
        logger=[tb_logger, csv_logger],
        limit_train_batches=config.accel.train_limit,
        limit_val_batches=config.accel.validation_limit,
        log_every_n_steps=config.accel.log_interval,
    )
    trainer.fit(
        lit_model,
        train_dataloaders=training,
        val_dataloaders=validation,
        ckpt_path=ckpt_path if ckpt_path.is_file() else None,
    )
