import typing as tp
import pickle
from anitune.config import DatasetConfig, ConfigError


def batch(config: DatasetConfig, max_batches_per_packet: int = 300) -> None:
    r"""
    Create a prebatched dataset from a given configuration
    """
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

    src_paths = list(config.src_paths)
    for name in config.data_names:
        ds = getattr(datasets, name)(
            skip_check=True,
            functional=config.functional,
            basis_set=config.basis_set,
        )
        src_paths.extend(ds.store_locations)
    src_paths = sorted(set(src_paths))

    datasets.create_batched_dataset(
        locations=datasets.ANIDataset(locations=src_paths),
        max_batches_per_packet=max_batches_per_packet,
        dest_path=config.path,
        batch_size=config.batch_size,
        shuffle_seed=config.shuffle_seed,
        properties=config.properties,
        **split_kwargs,
    )
    config.fold_idx = -1
    with open(config.path / "ds_config.pkl", mode="wb") as f:
        pickle.dump(config, f)
