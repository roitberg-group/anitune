from dataclasses import dataclass
import typing as tp
from pathlib import Path

import torch
from torchani.datasets import ANIDataset, ANIBatchedDataset, create_batched_dataset
from torchani.datasets._annotations import Conformers


@dataclass
class LearnSets:
    training: tp.Iterable[Conformers]
    validation: tp.Iterable[Conformers]
    testing: tp.Iterable[Conformers] = ()


def batch(ds: ANIDataset, batched_ds_path: Path, batch_size: int = 64) -> LearnSets:
    if not batched_ds_path.is_dir():
        create_batched_dataset(
            ds,
            include_properties=("species", "coordinates", "energies", "forces"),
            dest_path=batched_ds_path,
            batch_size=batch_size,
            max_batches_per_packet=13000,
            splits={
                "training": 0.8,
                "validation": 0.2,
            },
        )

    training = torch.utils.data.DataLoader(
        ANIBatchedDataset(batched_ds_path, split="training"),
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        batch_size=None,
    )

    validation = torch.utils.data.DataLoader(
        ANIBatchedDataset(batched_ds_path, split="validation"),
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        batch_size=None,
    )
    return LearnSets(
        training=training,
        validation=validation,
    )

# TODO: Use lightning to training
