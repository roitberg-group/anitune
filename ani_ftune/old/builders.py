from typing import Iterable

import torch
from torch.nn import Parameter, ParameterDict

from torchani.models import ANI1x, ANI2x, ANID, ANI1ccx
from torchani.atomics import like_1x, like_2x, like_1ccx, standard
from .lean_dipole_model import DipoleEnergyLean
from .dipole_model import SimpleLocalDipoleMP, DipolesAndEnergies1x

__all__ = ["ANI1x", "ANI2x", "ANID", "ANI1ccx", "dispersion_atomics",
           "light_ions_atomics", "like_1x", "like_2x",
           "like_1ccx", "standard", "SimpleLocalDipoleMP",
           "DipolesAndEnergies1x", "DipoleEnergyLean"]


def light_ions_atomics(atom: str = 'H'):
    dims_for_atoms = {'H': (384, 144, 128, 96),
                      'C': (384, 144, 128, 96),
                      'N': (384, 144, 128, 96),
                      'O': (384, 144, 128, 96)}
    return standard(dims_for_atoms[atom],
                    activation=torch.nn.GELU(),
                    bias=False)


def embed_light_ions_atomics(atom: str = 'H'):
    dims_for_atoms = {'H': (404, 144, 128, 96),
                      'C': (404, 144, 128, 96),
                      'N': (404, 144, 128, 96),
                      'O': (404, 144, 128, 96)}
    return standard(dims_for_atoms[atom],
                    activation=torch.nn.GELU(),
                    bias=False)


def light_ions_lean(atom: str = 'H'):
    dims_for_atoms = {'H': (426, 144, 128, 96),
                      'C': (426, 144, 128, 96),
                      'N': (426, 144, 128, 96),
                      'O': (426, 144, 128, 96)}
    return standard(dims_for_atoms[atom],
                    activation=torch.nn.GELU(),
                    bias=False)


def light_ions_lean_2x(atom: str = 'H'):
    dims_for_atoms = {'H': (1028, 200, 128, 96),
                      'C': (1028, 200, 128, 96),
                      'N': (1028, 200, 128, 96),
                      'O': (1028, 200, 128, 96),
                      'S': (1028, 200, 128, 96),
                      'F': (1028, 200, 128, 96),
                      'Cl': (1028, 200, 128, 96)}
    return standard(dims_for_atoms[atom],
                    activation=torch.nn.GELU(),
                    bias=False)


def full_couple_light_ions_atomics(atom: str = 'H'):
    dims_for_atoms = {'H': (436, 144, 128, 96),
                      'C': (436, 144, 128, 96),
                      'N': (436, 144, 128, 96),
                      'O': (436, 144, 128, 96)}
    return standard(dims_for_atoms[atom],
                    activation=torch.nn.GELU(),
                    bias=False)


def dispersion_atomics(atom: str = 'H'):
    dims_for_atoms = {'H': (1008, 256, 192, 160),
                      'C': (1008, 256, 192, 160),
                      'N': (1008, 192, 160, 128),
                      'O': (1008, 192, 160, 128),
                      'S': (1008, 160, 128, 96),
                      'F': (1008, 160, 128, 96),
                      'Cl': (1008, 160, 128, 96)}
    return standard(dims_for_atoms[atom],
                    activation=torch.nn.GELU(),
                    bias=False)


def long_range_atomics(atom: str = 'H'):
    dims_for_atoms = {'H': (1043, 256, 192, 160),
                      'C': (1043, 256, 192, 160),
                      'N': (1043, 192, 160, 128),
                      'O': (1043, 192, 160, 128),
                      'S': (1043, 160, 128, 96),
                      'F': (1043, 160, 128, 96),
                      'Cl': (1043, 160, 128, 96)}
    return standard(dims_for_atoms[atom],
                    activation=torch.nn.GELU(),
                    bias=False)


class UncertaintyWeightedLoss(torch.nn.Module):
    def __init__(self, keys: Iterable[str]):
        super().__init__()
        self.params = ParameterDict({k: Parameter(torch.tensor(0.0, dtype=torch.float))
                                     for k in keys})
