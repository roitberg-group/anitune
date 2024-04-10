import math
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Callable

from torch import Tensor
from torch.optim import Optimizer
import torch.utils.tensorboard
from tqdm import tqdm

from torchani import geometry
from torchani import datasets, units, transforms, models

DatasetType = Union[torch.utils.data.DataLoader, datasets.ANIBatchedDataset]
DeviceType = Union[int, str, torch.device]
Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]
PathLike = Union[str, Path]
ScalarMetrics = Dict[str, float]


# Validation / Training logic, including loss
class Runner:
    def __init__(self, model: models.BuiltinModel,
                       optimizer: Optimizer,
                       transform: Optional[Transform] = None,
                       device: Optional[DeviceType] = None,
                       best_metric: float = math.inf):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self._transform = transform.to(device) if transform is not None else transforms.Identity().to(device)
        self._model = model.to(device)
        self._optimizer = optimizer
        self._squared_error = torch.nn.MSELoss(reduction='none')
        # metric to track to check if it improves
        self.best_metric = best_metric
        self.best_metric_improved_last_run = False

    def inner_loop(self, batch, metrics: Dict[str, Tensor], train: bool = False) -> Tuple[Tensor, int]:
        # This method is the one that should be overriden, must return the batch loss
        # (not averaged) and the number of conformations in the batch (shape of species)
        species = batch['species'].long()
        coordinates = batch['coordinates'].float()
        target_energies = batch['energies'].float()

        predicted_energies = self._model((species, coordinates)).energies

        batch_loss = self._squared_error(predicted_energies, target_energies)
        return batch_loss, species.shape[0]

    def set_extra_metrics(self):
        # Must return metric_name : initial value dict, metrics that end with
        # "hartree" or "hartree_per_angstrom" are treated specially (other
        # metrics with different units are added to them)
        return {}

    def set_train_only_metrics(self):
        return {}

    def set_eval_only_metrics(self):
        return {}

    def _run(self, dataset: DatasetType,
                   epoch: Optional[int] = None,
                   train: bool = False,
                   use_tqdm: bool = True,
                   verbose: bool = True) -> ScalarMetrics:
        split = self._get_split(dataset)
        msg = f"epoch {epoch}, {split}" if epoch is not None else split
        metrics = {'loss': 0.0, 'count': 0}
        metrics.update(self.set_extra_metrics())
        if train:
            metrics.update(self.set_train_only_metrics())
        else:
            metrics.update(self.set_eval_only_metrics())
        for batch in tqdm(dataset, total=len(dataset), desc=msg, disable=not use_tqdm):
            batch = self._transform({k: v.to(self._device, non_blocking=True)
                                     for k, v in batch.items()})
            batch_loss, count = self.inner_loop(batch, metrics, train=train)
            if train:
                self._run_backwards(batch_loss.mean())
            metrics['loss'] += batch_loss.detach().sum().item()
            metrics['count'] += count
        metrics = self._average_metrics(metrics)
        metrics = self._add_kcalpermol_metrics(metrics)
        if verbose:
            self._print_metrics(split, metrics, epoch)
        return metrics

    def _add_kcalpermol_metrics(self, metrics: ScalarMetrics):
        for k in metrics.copy().keys():
            if k.endswith('_hartree'):
                metrics[k.replace('_hartree', '_kcalpermol')] = units.hartree2kcalmol(metrics[k])
            elif k.endswith('_hartree_per_angstrom'):
                metrics[k.replace('_hartree_per_angstrom', '_kcalpermol_per_angstrom')] = units.hartree2kcalmol(metrics[k])
            elif k.endswith('_eA'):
                metrics[k.replace('_eA', '_debye')] = units.ea2debye(metrics[k])
        return metrics

    def _average_metrics(self, metrics: ScalarMetrics):
        count = metrics.pop('count')
        for k in metrics.copy().keys():
            metrics[k] = (metrics[k] / count)
            if 'rmse' in k:
                metrics[k] = math.sqrt(metrics[k])
        return metrics

    def _run_backwards(self, batch_loss):
        self._optimizer.zero_grad()
        batch_loss.backward()
        self._optimizer.step()

    @staticmethod
    def _get_split(dataset):
        if isinstance(dataset, datasets.ANIBatchedDataset):
            split = dataset.split
        else:
            assert isinstance(dataset.dataset, datasets.ANIBatchedDataset)
            split = dataset.dataset.split
        return split

    def _print_metrics(self, split: str, metrics: ScalarMetrics, epoch: Optional[int] = None):
        print(f'epoch {epoch}, {split} metrics:' if epoch is not None else f'{split} metrics:')
        for k, v in metrics.items():
            print(f'    {k} = {v}')
        print()

    def train(self, dataset: DatasetType, epoch: int, **kwargs: bool) -> ScalarMetrics:
        self._model.train()
        metrics = self._run(dataset, epoch, train=True, **kwargs)
        return metrics

    def eval(self, dataset: DatasetType, epoch: int, track_metric: Optional[str] = None, **kwargs: bool) -> ScalarMetrics:
        self._model.eval()
        with torch.no_grad():
            metrics = self._run(dataset, epoch, train=False, **kwargs)
        if track_metric is not None:
            if metrics[track_metric] < self.best_metric:
                self.best_metric = metrics[track_metric]
                self.best_metric_improved_last_run = True
            metrics.update({f'best_{track_metric}': self.best_metric})
        return metrics

    def load_state_dict(self, state_dict: ScalarMetrics) -> None:
        self.__dict__.update(state_dict)

    def state_dict(self) -> ScalarMetrics:
        return {'best_metric': self.best_metric}


class MultiRunner(Runner):
    _DEFAULT_LOSSES = {"force": 0.1, "energy": 1.0, "dipole": 0.1, "diss": 1e-4, "charge": 1.0, "force_smoothness": 1e-4, "charge_smoothness": 1e-4}

    def __init__(self, losses: Dict[str, Optional[float]],
                       scale_energy_sqrt: bool = True,
                       processed_energies: bool = False,
                       processed_forces: bool = False,
                       processed_dipoles: bool = False,
                       uw_loss: torch.nn.Module = None,
                       **kwargs):
        # fill losses with default parameters if they are "None"
        for k, v in losses.copy().items():
            if v is None:
                losses[k] = self._DEFAULT_LOSSES[k]

        assert set(losses.keys()).issubset(self._DEFAULT_LOSSES.keys())
        self._losses = losses

        self._scale_energy_sqrt = scale_energy_sqrt
        self._relu = torch.nn.ReLU()
        self._energies_key = 'energies' if not processed_energies else "processed_energies"
        self._forces_key = 'forces' if not processed_forces else "processed_forces"
        self._dipoles_key = 'dipoles' if not processed_dipoles else "processed_dipoles"
        self._charges_key = "charges"
        self._uw_loss = uw_loss
        super().__init__(**kwargs)

    def inner_loop(self, batch, metrics: Dict[str, Tensor], train: bool = False) -> Tuple[Tensor, int]:
        species = batch['species'].long()
        if "energy" in self._losses.keys():
            target_energies = batch[self._energies_key].float()
        if "force" in self._losses.keys():
            target_forces = batch[self._forces_key].float()
        if "dipole" in self._losses.keys():
            target_dipoles = batch[self._dipoles_key].float()
            total_charges = batch[self._charges_key].float()
        if "charge" in self._losses.keys() and "dipole" not in self._losses.keys():
            total_charges = batch[self._charges_key].float()
        num_atoms = (species >= 0).sum(dim=1).float()

        batch_loss = dict()
        with torch.enable_grad():
            coordinates = batch['coordinates'].float()
            if "force" in self._losses.keys():
                coordinates.requires_grad_(True)

            if "dipole" in self._losses.keys() or "charge" in self._losses.keys():
                # The predicted energies output by the model must already include the
                # qi * qj / rij factor
                if "charge" in self._losses.keys():
                    _, predicted_energies, predicted_charges, predicted_precharges = self._model((species, coordinates), total_charges=total_charges)
                else:
                    _, predicted_energies, predicted_charges = self._model((species, coordinates), total_charges=total_charges)
                if "charge_smoothness" in self._losses.keys():
                    charge_derivatives = torch.autograd.grad(predicted_charges.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                    batch_loss["charge_smoothness"] = charge_derivatives.pow(2).sum(dim=(1, 2))
            else:
                predicted_energies = self._model((species, coordinates)).energies

            if "energy" in self._losses.keys():
                squared_energy_error = self._squared_error(predicted_energies, target_energies)
                if self._scale_energy_sqrt:
                    batch_loss["energy"] = squared_energy_error / num_atoms.sqrt()
                else:
                    batch_loss["energy"] = squared_energy_error

            if "force" in self._losses.keys():
                predicted_forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
                squared_force_error = self._squared_error(predicted_forces, target_forces).sum(dim=(1, 2))
                batch_loss["force"] = squared_force_error / (3 * num_atoms)
                if "force_smoothness" in self._losses.keys():
                    batch_loss["force_smoothness"] = predicted_forces.pow(2).sum(dim=(1, 2))

        if "diss" in self._losses.keys():
            shifter = self._model.energy_shifter
            converter = self._model.species_converter
            batch_loss["diss"] = self._relu(shifter.sae(converter((species, coordinates))[0]) - predicted_energies) ** 2

        if "charge" in self._losses.keys():
            squared_charge_error = self._squared_error(total_charges, predicted_precharges.sum(-1))
            batch_loss["charge"] = squared_charge_error

        if "dipole" in self._losses.keys():
            _, displaced_coordinates = geometry.displace_to_com_frame((species, coordinates))
            predicted_dipoles = (displaced_coordinates * predicted_charges.unsqueeze(-1)).sum(-2)
            squared_dipole_error = self._squared_error(target_dipoles, predicted_dipoles).sum(-1)
            batch_loss["dipole"] = squared_dipole_error / 3

        if "energy" in self._losses.keys():
            metrics['energy_rmse_hartree'] += squared_energy_error.detach().sum().item()
            metrics['energy_mae_hartree'] += torch.abs(predicted_energies.detach() - target_energies.detach()).sum().item()
        if "force" in self._losses.keys():
            metrics['force_rmse_hartree_per_angstrom'] += batch_loss["force"].detach().sum().item()
            metrics['force_mae_hartree_per_angstrom'] += (torch.sqrt(squared_force_error.detach()) / (3 * num_atoms)).sum().item()
        if "dipole" in self._losses.keys():
            metrics['dipole_rmse_eA'] += batch_loss["dipole"].detach().sum().item()
            metrics['dipole_mae_eA'] += (torch.sqrt(squared_dipole_error.detach()) / 3).sum().item()
        if "charge" in self._losses.keys():
            metrics['charge_rmse_e'] += batch_loss["charge"].detach().sum().item()
            metrics['charge_mae_e'] += (torch.sqrt(squared_charge_error.detach())).sum().item()

        # scale the losses after registering metrics
        for k, v in self._losses.items():
            batch_loss[k] *= v

        if self._uw_loss is not None:
            for k, s in self._uw_loss.params.items():
                if not train:
                    # very possibly not needed
                    s.detach()
                batch_loss[k] = 0.5 * torch.exp(-s) * batch_loss[k] + 0.5 * s
                if train:
                    metrics[k + "_factor"] += 0.5 * math.exp(-s.item()) * len(species)
        batch_loss = sum(batch_loss.values())
        return batch_loss, species.shape[0]

    def set_extra_metrics(self):
        metrics = dict()
        if "energy" in self._losses.keys():
            metrics.update({'energy_rmse_hartree': 0.0, 'energy_mae_hartree': 0.0})
        if "force" in self._losses.keys():
            metrics.update({'force_rmse_hartree_per_angstrom': 0.0, 'force_mae_hartree_per_angstrom': 0.0})
        if "dipole" in self._losses.keys():
            metrics.update({'dipole_rmse_eA': 0.0, 'dipole_mae_eA': 0.0})
        if "charge" in self._losses.keys():
            metrics.update({'charge_rmse_e': 0.0, 'charge_mae_e': 0.0})
        return metrics

    def set_train_only_metrics(self):
        metrics = dict()
        if self._uw_loss is not None:
            metrics.update({k + "_factor": 0.0 for k in self._uw_loss.params.keys()})
        return metrics
