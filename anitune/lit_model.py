import itertools
import typing as tp

import torch
from torch import Tensor
import lightning
import torchmetrics
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from torchani.assembly import ANI
from torchani.units import hartree2kcalpermol

from anitune import losses
from anitune.annotations import Scalar


class LitModel(lightning.LightningModule):
    r"""
    ANI-style model, wrapped to enable training with PyTorch Lightning
    """

    def __init__(
        self,
        model: ANI,
        loss_terms_and_factors: tp.Dict[str, float],
        optimizer_options: tp.Dict[str, Scalar],
        scheduler_options: tp.Dict[str, Scalar],
        monitor_label: str = "valid_rmse_default",
        optimizer_cls: str = "AdamW",
        scheduler_cls: str = "ReduceLROnPlateau",
        uncertainty_weighted: bool = False,
        num_head_layers: int = 0,
    ) -> None:
        super().__init__()
        self.optimizer_options = optimizer_options
        self.scheduler_options = scheduler_options
        self.optimizer_cls = optimizer_cls
        self.scheduler_cls = scheduler_cls
        self.train_metrics = torch.nn.ModuleDict()
        self.train_losses = torch.nn.ModuleDict()
        self.valid_metrics = torch.nn.ModuleDict()

        loss_terms = tuple(
            getattr(losses, name)(factor=factor)
            for name, factor in loss_terms_and_factors.items()
        )

        for term in loss_terms:
            k = term.label
            # torchmetrics.MeanSquaredError(squared=False) is directly the RMSE,
            # no extra calculations are needed
            self.train_metrics[k] = torchmetrics.MetricCollection(
                {
                    "rmse": torchmetrics.MeanSquaredError(squared=False),
                    "mae": torchmetrics.MeanAbsoluteError(),
                },
                prefix="train_",
            )
            self.valid_metrics[k] = self.train_metrics[k].clone(prefix="valid_")

        if len(loss_terms) == 1 and monitor_label == "valid_rmse_default":
            monitor_label = f"valid_rmse_{loss_terms[0].label}"
        elif not any(monitor_label.endswith(term.label) for term in loss_terms):
            raise ValueError(
                "The monitored label must be one of the enabled loss terms"
            )
        self.monitor_label = monitor_label

        self.loss = losses.MultiTaskLoss(loss_terms, uncertainty_weighted)
        self.model = model

        # Hyperparameters
        self.save_hyperparameters(ignore="model")

        # Backbone for finetuning
        module_list = torch.nn.ModuleList()
        if num_head_layers > 0:
            for k in model.get_chemical_symbols():
                layers = model.neural_networks.atomics[k].layers
                last_layer = model.neural_networks.atomics[k].last_layer
                rev_layers = itertools.chain([last_layer], reversed(layers))
                module_list.extend(list(rev_layers)[:num_head_layers])
        self.backbone = module_list

    def training_step(
        self,
        batch: tp.Dict[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        pred = self.batch_eval(batch)
        with torch.no_grad():
            for k, v in self.train_metrics.items():
                v.update(pred[k], batch[self.loss.term(k).targ_label])
        loss_dict = self.loss(pred, batch)
        return loss_dict["loss"]

    def validation_step(
        self,
        batch: tp.Dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        with torch.enable_grad():
            pred = self.batch_eval(batch)

        for k, v in self.valid_metrics.items():
            v.update(pred[k], batch[self.loss.term(k).targ_label])

    def on_train_epoch_end(self) -> None:
        self._log_metrics(self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics(self.valid_metrics)

    def _log_metrics(self, computers: torch.nn.ModuleDict) -> None:
        metrics = {}
        for k, c in computers.items():
            m = c.compute()
            c.reset()
            for name, v in m.items():
                metrics[f"{name}_{k}"] = v
                #  names of metrics
                #  (energies|...)_(train|valid)_(rmse|mae)[kcal|mol[|ang]]
                if "energies" in k:
                    metrics[f"{name}_{k}_kcal|mol"] = hartree2kcalpermol(v)
                elif "forces" in k:
                    metrics[f"{name}_{k}_kcal|mol|ang"] = hartree2kcalpermol(v)
        self.log_dict(metrics)

    def batch_eval(
        self,
        batch: tp.Dict[str, Tensor],
    ) -> tp.Dict[str, Tensor]:
        for term in self.loss.grad_terms:
            batch[term.grad_wrt_to_targ_label].requires_grad_(True)

        pred = self.model.sp((batch["species"], batch["coordinates"]))

        for term in self.loss.grad_terms:
            pred[term.label] = -torch.autograd.grad(
                pred[term.grad_of_label].sum(),
                batch[term.grad_wrt_to_targ_label],
                retain_graph=True,
            )[0]

        for term in self.loss.grad_terms:
            batch[term.grad_wrt_to_targ_label].requires_grad_(False)
        return pred

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Optimizer setup
        opt_type = getattr(torch.optim, self.optimizer_cls)
        optimizer = opt_type(self.model.parameters(), **self.optimizer_options)

        scheduler_type = getattr(torch.optim.lr_scheduler, self.scheduler_cls)
        # Schedulers setup
        scheduler = scheduler_type(
            optimizer=optimizer,
            **self.scheduler_options,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "strict": True,
            "monitor": self.monitor_label,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
