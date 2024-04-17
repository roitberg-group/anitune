import typing as tp
from enum import Enum

import torch
from torch import Tensor


class LossKind(Enum):
    MSE = "mse"
    MSE_UNCERTAINTY_WEIGHTED = "mse-uncertainty-weighted"


class Loss(torch.nn.Module):
    def __init__(
        self,
        energy_factor: float,
        force_factor: float,
        dipole_factor: float,
        energy_sqrt_num_atoms: bool = True,
        force_sqrt_num_atoms: bool = False,
    ) -> None:
        super().__init__()
        self.energy_factor = energy_factor
        self.force_factor = force_factor
        self.dipole_factor = dipole_factor
        self.energy_sqrt_num_atoms = energy_sqrt_num_atoms
        self.force_sqrt_num_atoms = force_sqrt_num_atoms

    def forward(
        self,
        energies: tp.Optional[Tensor],
        forces: tp.Optional[Tensor],
        dipoles: tp.Optional[Tensor],
        batch: tp.Dict[str, Tensor],
    ) -> tp.Dict[str, Tensor]:
        loss = torch.tensor(0.0, dtype=torch.float)
        dipole_loss = torch.tensor(0.0, dtype=torch.float)
        force_loss = torch.tensor(0.0, dtype=torch.float)
        energy_loss = torch.tensor(0.0, dtype=torch.float)

        num_atoms = (batch["species"] >= 0).sum(dim=1, dtype=torch.float)
        if energies is not None:
            energy_loss = (energies - batch["energies"]).pow(2).sum()
            if self.energy_sqrt_num_atoms:
                energy_loss / num_atoms.sqrt()
            else:
                energy_loss / num_atoms
            loss += self.energy_factor * energy_loss

        if forces is not None:
            force_loss = (forces - batch["forces"]).pow(2).sum() / 3
            if self.force_sqrt_num_atoms:
                force_loss / num_atoms.sqrt()
            else:
                force_loss / num_atoms
            loss += self.force_factor * force_loss

        if dipoles is not None:
            dipole_loss = (dipoles - batch["forces"]).pow(2).sum() / 3
            loss += self.dipole_factor * dipole_loss
        return {
            "total": loss,
            "dipole": dipole_loss,
            "force": force_loss,
            "energy": energy_loss,
        }


def LossFactory(
    loss: LossKind = LossKind.MSE,
    *args,
    **kwargs,
) -> Loss:
    raise NotImplementedError()
