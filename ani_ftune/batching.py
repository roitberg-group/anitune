import typing as tp
import pickle
from ani_ftune.config import DatasetConfig, ConfigError


def batch(config: DatasetConfig, max_batches_per_packet: int = 300) -> None:
    from torchani import datasets

    split_kwargs: tp.Dict[str, tp.Union[int, tp.Dict[str, float]]]
    if config.folds is not None:
        if config.train_frac != 0.8 or config.validation_frac != 0.2:
            raise ConfigError("Train and val frac can't be set if training to folds")
        if not isinstance(config.fold_idx, int):
            raise ConfigError("A fold idx must be present when training to folds")
        split_kwargs = {"folds": config.folds}
    else:
        split_kwargs = {"splits": config.split_dict}

    if config.src_paths:
        ds = datasets.ANIDataset(locations=config.src_paths)
        if config.name:
            raise ValueError(
                "Dataset name should not be set if custom source paths are specified"
            )

    else:
        ds = getattr(datasets, config.name)(
            skip_check=True,
            functional=config.functional,
            basis_set=config.basis_set,
        )
    datasets.create_batched_dataset(
        locations=ds,
        max_batches_per_packet=max_batches_per_packet,
        dest_path=config.path,
        batch_size=config.batch_size,
        shuffle_seed=config.shuffle_seed,
        **split_kwargs,
    )
    config.fold_idx = -1
    with open(config.path / "ds_config.pkl", mode="wb") as f:
        pickle.dump(config, f)
