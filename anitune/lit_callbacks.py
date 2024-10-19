r"""
Custom Lightning compatible callbacks
"""

import json
import typing as tp
from pathlib import Path

from torch import Tensor
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning import Trainer, LightningModule

from anitune.config import TrainConfig


class ModelCheckpointWithMetrics(ModelCheckpoint):
    r"""
    Checkpoint a model and also save the callback metrics from the trainer
    """

    def check_monitor_top_k(
        self, trainer: Trainer, current: tp.Optional[Tensor] = None
    ) -> bool:
        should_update_and_save = super().check_monitor_top_k(trainer, current)
        if should_update_and_save:
            self._dump_metrics(trainer)
        return should_update_and_save

    def _save_topk_checkpoint(
        self, trainer: Trainer, monitor_candidates: tp.Dict[str, Tensor]
    ) -> None:
        super()._save_topk_checkpoint(trainer, monitor_candidates)
        # In this case check_monitor_top_k is not called, and the
        # metrics should be dump every step
        if self.monitor is not None:
            return
        self._dump_metrics(trainer)

    def _dump_metrics(self, trainer: Trainer) -> None:
        #  names of metrics are (energies|...)_(train|valid)_(rmse|mae)[kcal|mol[|ang]]
        candidates = trainer.callback_metrics
        if self.dirpath is not None:
            dirpath = Path(self.dirpath).resolve()
        else:
            dirpath = Path(trainer.default_root_dir).resolve()
        dirpath.mkdir(exist_ok=True)

        metrics: tp.Dict[str, tp.Union[int, float]] = {"epoch": trainer.current_epoch}
        for k, v in candidates.items():
            if k.startswith("valid/") or k.startswith("train/"):
                metrics[k] = v.item()
        with open(dirpath / "metrics.json", mode="wt", encoding="utf-8") as ft:
            json.dump(metrics, ft, indent=4)


class SaveConfig(Callback):
    r"""
    Save the configuration of a training run at the start of the run
    """

    def __init__(
        self,
        config: TrainConfig,
    ) -> None:
        super().__init__()
        self._config = config

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        root = Path(trainer.default_root_dir).resolve()
        self._config.to_json_file(root / "config.json")
