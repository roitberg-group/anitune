r"""
Holds base class for loss terms and some simple loss terms used to train
ANI-style models
"""

from dataclasses import dataclass
import typing as tp
from enum import Enum

import torch
from torch import Tensor


class Penalty(Enum):
    SQUARE = "square"
    ABS = "abs"


@dataclass
class LossTerm:
    label: str
    is_extensive: bool
    scale_by_sqrt_atoms: bool = False
    is_vec3: bool = False
    factor: float = 1.0
    grad_of: tp.Optional[str] = None
    penalty: Penalty = Penalty.SQUARE


def Forces(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="forces",
        grad_of="energies",
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
        label="energies_xc",
        is_extensive=True,
        factor=factor,
    )


def EnergiesXCSqrtAtoms(factor: float = 1.0) -> LossTerm:
    return LossTerm(
        label="energies_xc",
        is_extensive=True,
        factor=factor,
        scale_by_sqrt_atoms=True,
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
        label="atomic_charges",
        is_extensive=True,
        factor=factor,
    )


class MultiTaskLoss(torch.nn.Module):
    r"""
    Represents a loss with multiple objectives (potentially scalar or vector
    valued)
    """

    def is_enabled(self, label: str) -> bool:
        r"""
        True if a specific label is being used in the loss
        """
        return any(term.label == label for term in self.terms)

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
        r"""
        Calculate a dictionary of losses for some given predicted and target properties

        As an example, if the enabled loss terms are 'energies' and 'forces',
        then the output loss dictionary will be
        {'loss': <full-loss>, 'energies': <energy-loss>, 'forces': <force-loss>}.

        'species' must be one of the target properties
        """
        if "species" not in targ:
            raise ValueError("'species' must be one of the target properties")
        losses: tp.Dict[str, Tensor] = {}

        losses["loss"] = torch.tensor(
            0.0, dtype=torch.float, device=targ["species"].device
        )
        for term in self.terms:
            k = term.label

            diff = pred[k] - targ[k]

            if term.penalty is Penalty.SQUARE:
                error = diff.pow(2)
            elif term.penalty is Penalty.ABS:
                error = torch.abs(diff)

            if term.is_vec3:
                error = error / 3

            if term.scale_by_sqrt_atoms or term.is_extensive:
                num_atoms = (targ["species"] >= 0).sum(dim=1, dtype=torch.float)
                num_atoms = num_atoms.view((-1,) + (1,) * (diff.ndim - 1))
                if term.scale_by_sqrt_atoms:
                    error = error * num_atoms.sqrt()
                if term.is_extensive:
                    error = error / num_atoms

            mean_error = error.mean()
            losses[k] = mean_error
            losses["loss"] += mean_error * term.factor
        return losses
