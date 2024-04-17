import typing as tp

import torch
from torch import Tensor
import lightning
import torch.utils.tensorboard

import torchani
from torchani.models import BuiltinModel
from torchani.datasets import ANIDataset, ANIBatchedDataset
from torchani.units import hartree2kcalpermol
from torchani.assembler import FlexibleANI

from anitune.losses import LossKind, LossFactory


class LitModel(lightning.LightningModule):
    def __init__(
        self,
        model: BuiltinModel,
        opt_lr: float = 0.5e-3,
        opt_weight_decay: float = 1e-7,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 100,
        scheduler_threshold: float = 0.0,
        min_lr: float = 1e-8,
        max_epochs: int = 1000,
        loss_sqrt_atoms: bool = True,
        loss: LossKind = LossKind.MSE,
        loss_factors: tp.Optional[tp.Dict[str, float]] = None,
        train_energies: bool = True,
        train_forces: bool = False,
        train_dipoles: bool = False,
    ) -> None:
        super().__init__()
        _loss_factors = {
            "energy_factor": 1.0,
            "dipole_factor": 0.001,
            "force_factor": 0.01,
        }
        if loss_factors is not None:
            if not set(loss_factors.keys()).issubset(_loss_factors.keys()):
                raise ValueError(f"Allowed loss factors are {set(_loss_factors.keys())}")
            _loss_factors.update(loss_factors)

        self.loss = LossFactory(loss, **_loss_factors)
        self.model = model
        self.opt_weight_decay = opt_weight_decay
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.opt_lr = opt_lr
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_threshold = scheduler_threshold
        self.loss_sqrt_atoms
        self.save_hyperparameters(ignore="model")

        self.train_forces = train_forces
        self.train_dipoles = train_dipoles

    def training_step(
        self,
        batch: tp.Any,
        batch_idx: int,
    ) -> Tensor:
        return self.batch_eval(
            tp.cast(tp.Dict[str, Tensor], batch),
            batch_idx,
        )

    def validation_step(
        self,
        batch: tp.Any,
        batch_idx: int,
    ) -> None:
        torch.set_grad_enabled(True)
        self.batch_eval(
            tp.cast(tp.Dict[str, Tensor], batch),
            batch_idx,
        )

    def batch_eval(
        self,
        batch: tp.Dict[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        if self.train_forces:
            batch["coordinates"].requires_grad_(True)
        output = self.model((batch["species"], batch["coordinates"]))
        energies = output.energies

        forces: tp.Optional[Tensor]
        if self.train_forces:
            forces = -torch.autograd.grad(
                energies.sum(),
                batch["coordinates"],
            )[0]
        else:
            forces = None

        dipoles: tp.Optional[Tensor]
        if hasattr(output, "dipoles") and self.train_dipoles:
            dipoles = output.dipoles

        if self.train_forces:
            batch["coordinates"].requires_grad_(True)

        loss = self.loss(
            energies=energies,
            forces=forces,
            dipoles=dipoles,
            target=batch,
        )
        return loss

    def configure_optimizers(self) -> tp.Any:
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.opt_lr,
            weight_decay=self.opt_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            threshold=self.scheduler_threshold,
        )
        return opt, scheduler
