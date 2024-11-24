import typing as tp

from anitune.config import DatasetConfig, ConfigError


def batch_data(config: DatasetConfig, max_batches_per_packet: int = 300) -> None:
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

    raw_src_paths = list(config.raw_src_paths)
    for builtin in config.data_names:
        name, lot = builtin.split(":")
        ds = getattr(datasets, name)(
            skip_check=True,
            lot=lot,
        )
        raw_src_paths.extend(ds.store_locations)
    raw_src_paths = sorted(set(raw_src_paths))
    ds = datasets.ANIDataset(locations=raw_src_paths)
    _props = config.properties
    if _props:
        if "species" not in _props and "species" in ds.tensor_properties:
            config.properties.append("species")
        if "coordinates" not in _props and "coordinates" in ds.tensor_properties:
            config.properties.append("coordinates")

    datasets.create_batched_dataset(
        src=ds,
        max_batches_per_packet=max_batches_per_packet,
        dest_path=config.path,
        batch_size=config.batch_size,
        divs_seed=config.divs_seed,
        batch_seed=config.batch_seed,
        properties=config.properties,
        **split_kwargs,
    )
    config.fold_idx = -1
    config.to_json_file(config.path / "ds_config.json")
