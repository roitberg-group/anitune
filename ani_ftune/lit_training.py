import typing as tp

import torch
import lightning
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
)

from torchani import datasets

from ani_ftune.exceptions import ConfigError
from ani_ftune import model_builders
from ani_ftune.lit_models import LitModel
from ani_ftune import losses
from ani_ftune.configuration import (
    TrainConfig,
    DatasetConfig,
    AccelConfig,
    ModelConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
)


config = TrainConfig(
    ds=DatasetConfig(
        name="TestData",
        batch_size=1500,
        folds=5,
        fold_idx=0,
    ),
    accel=AccelConfig(max_batches_per_packet=100),
    model=ModelConfig(),
    loss=LossConfig(
        terms_and_factors=frozenset({("Energies", 1.0)}),
    ),
    optim=OptimizerConfig(),
    scheduler=SchedulerConfig(),
)


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
training_label = f"training{config.ds.fold_idx}"
validation_label = f"validation{config.ds.fold_idx}"
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
trainer = lightning.Trainer(
    devices=1,
    accelerator=config.accel.device,
    max_epochs=config.accel.max_epochs,
    callbacks=[lr_monitor, early_stopping],
)
trainer.fit(
    lit_model,
    train_dataloaders=training,
    val_dataloaders=validation,
)
