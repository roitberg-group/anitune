import math
import itertools
import typing as tp

import torch
from torch import Tensor
import lightning
import torchmetrics
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning.pytorch.loggers import TensorBoardLogger

from torchani.assembly import ANI
from torchani.units import hartree2kcalpermol

from anitune import losses
from anitune.lit_callbacks import NoLogLRMonitor
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
        monitor_label: str = "valid/rmse_default",
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

        loss_terms = tuple(
            getattr(losses, name)(factor=factor)
            for name, factor in loss_terms_and_factors.items()
        )
        metrics: tp.Dict[str, torchmetrics.Metric] = {}
        for term in loss_terms:
            for div in ("valid", "train"):
                # torchmetrics.MeanSquaredError(squared=False) is directly the RMSE
                metrics[f"{div}/rmse_{term.label}"] = torchmetrics.MeanSquaredError(
                    squared=False
                )
                metrics[f"{div}/mae_{term.label}"] = torchmetrics.MeanAbsoluteError()
        self.metrics = torchmetrics.MetricCollection(metrics)

        if len(loss_terms) == 1 and monitor_label == "valid/rmse_default":
            monitor_label = f"valid/rmse_{loss_terms[0].label}"
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

    def on_train_start(self) -> None:
        # Log hyperparameters to tensorboard events file (only a single time)
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                # hparams seems to be of the correct type, but lightning marks it
                # differently
                logger.log_hyperparams(
                    self.hparams, {self.monitor_label: math.inf}  # type: ignore
                )
                print("Logging hparams, maybe?")

    def training_step(
        self,
        batch: tp.Dict[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        pred = self.batch_eval(batch)
        with torch.no_grad():
            self._update_metrics("train", pred, batch)
        loss_dict = self.loss(pred, batch)
        return loss_dict["loss"]

    def validation_step(
        self,
        batch: tp.Dict[str, Tensor],
        batch_idx: int,
    ) -> None:
        with torch.enable_grad():
            pred = self.batch_eval(batch)
        self._update_metrics("valid", pred, batch)

    def _update_metrics(
        self, div: str, pred: tp.Dict[str, Tensor], batch: tp.Dict[str, Tensor]
    ) -> None:
        for k, v in self.metrics.items():
            if not k.startswith(f"{div}/"):
                continue
            label = k.split("_")[-1]
            v.update(pred[label], batch[self.loss.term(label).targ_label])

    # Metrics are logged at the end of each validation epoch only
    # This is only correct if check_val_every_n_epochs=1
    def on_validation_epoch_end(self) -> None:
        results = {}
        for k, c in self.metrics.items():
            if not c.update_called:
                continue
            results[k] = c.compute()
            c.reset()
            if "energies" in k:
                results[f"{k}_kcal|mol"] = hartree2kcalpermol(results[k])
            elif "forces" in k:
                results[f"{k}_kcal|mol|ang"] = hartree2kcalpermol(results[k])

        # I believe callbacks is technically pvt API
        for c in self.trainer.callbacks:  # type: ignore
            if isinstance(c, NoLogLRMonitor):
                results.update(c.extract_stats(self.trainer))
                break

        self.log_dict(results)

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
