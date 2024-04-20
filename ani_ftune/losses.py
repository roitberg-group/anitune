from dataclasses import dataclass
import typing as tp
from enum import Enum

import torch
from torch import Tensor


class Penalty(Enum):
    SQUARE = 0
    ABS = 1


@dataclass
class LossTerm:
    label: str
    is_extensive: bool
    scale_by_sqrt_atoms: bool = False
    is_vec3: bool = False
    factor: float = 1.0
    grad_label: tp.Optional[str] = None
    penalty: Penalty = Penalty.SQUARE


def Forces(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="forces",
        grad_label="energies",
        is_vec3=True,
        is_extensive=True,
        factor=factor,
    )


def Energies(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies",
        is_extensive=True,
        factor=factor,
    )


def EnergiesSqrtAtoms(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies",
        is_extensive=True,
        factor=factor,
        scale_by_sqrt_atoms=True,
    )


def TotalCharge(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="total_charge",
        is_extensive=True,
        factor=factor,
    )


def EnergiesXC(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies-xc",
        is_extensive=True,
        factor=factor,
    )


def Dipoles(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="dipoles",
        is_extensive=False,
        is_vec3=True,
        factor=factor,
    )


def AtomicCharges(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="dipoles",
        is_extensive=False,
        is_vec3=True,
        factor=factor,
    )


class MultiTaskLoss(torch.nn.Module):
    def is_enabled(self, value: str) -> bool:
        return any(term.label == value for term in self.terms)

    def __init__(
        self,
        terms: tp.Sequence[LossTerm],
        uncertainty_weighted: bool = False,
    ) -> None:
        super().__init__()
        self.terms = tuple(terms)
        if uncertainty_weighted:
            raise NotImplementedError("Uncertainty Weighted loss not implemented yet")

    def forward(
        self,
        pred: tp.Dict[str, Tensor],
        targ: tp.Dict[str, Tensor],
    ) -> tp.Dict[str, Tensor]:
        losses: tp.Dict[str, Tensor] = {}
        num_atoms = (targ["species"] >= 0).sum(dim=1, dtype=torch.float)

        losses["loss"] = torch.tensor(0.0, dtype=torch.float, device=targ["species"].device)
        for term in self.terms:
            k = term.label

            diff = pred[k] - targ[k]
            num_atoms = num_atoms.view((-1,) + (1,) * (diff.ndim - 1))
            if term.penalty is Penalty.SQUARE:
                error = diff.pow(2)
            elif term.penalty is Penalty.ABS:
                error = torch.abs(diff)
            if term.scale_by_sqrt_atoms:
                error = error * num_atoms.sqrt()
            if term.is_extensive:
                error = error / num_atoms
            if term.is_vec3:
                error = error / 3

            merror = error.mean()
            losses[k] = merror
            losses["loss"] += merror * term.factor
        return losses
