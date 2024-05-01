import pickle
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
    def _save_topk_checkpoint(self, trainer: Trainer, monitor_candidates: tp.Dict[str, Tensor]) -> None:
        super()._save_topk_checkpoint(trainer, monitor_candidates)
        #  names of metrics are (energies|...)_(train|valid)_(rmse|mae)[kcal|mol[|ang]]
        if self.dirpath is not None:
            dirpath = Path(self.dirpath).resolve()
        else:
            dirpath = Path(trainer.default_root_dir).resolve()

        metrics: tp.Dict[str, tp.Union[int, float]] = {"epoch": monitor_candidates["epoch"].item()}
        for k, v in monitor_candidates.items():
            if "_rmse" in k or "_mae" in k:
                metrics[k] = v.item()
        with open(dirpath / "metrics.pkl", mode="wb") as fb:
            pickle.dump(metrics, fb)


class SaveConfig(Callback):
    r"""
    Save the configuration of a training run at the start of the run
    """

    def __init__(
        self,
        config: TrainConfig,
        dests: tp.Iterable[str] = ("latest-model", "best-model"),
    ) -> None:
        super().__init__()
        self._dests = (dests,) if isinstance(dests, str) else dests
        self._config = config

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        root = Path(trainer.default_root_dir).resolve()

        with open(root / "config.pkl", mode="wb") as f:
            pickle.dump(self._config, f)

        for dest in self._dests:
            if dest:
                dir_ = root / dest
                dir_.mkdir(exist_ok=True, parents=True)
                with open(dir_ / "model_config.pkl", mode="wb") as f:
                    pickle.dump(self._config.model, f)


class MergeTensorBoardLogs(Callback):
    r"""
    Combine all of the tensorboard logs for different versions at the end of a run
    """

    def __init__(self, src: str, dest: str = "tb-logs") -> None:
        super().__init__()
        self._src = src
        self._dest = dest

    def _merge_logs(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        root = Path(trainer.default_root_dir).resolve()
        tb_versioned_logs = root / self._src
        if (not tb_versioned_logs.is_dir()) or (not any(tb_versioned_logs.iterdir())):
            return

        tb_logs = root / self._dest
        tb_logs.mkdir(exist_ok=True, parents=True)
        versions = sorted(tb_versioned_logs.iterdir())
        for d in versions:
            for f in d.iterdir():
                if "tfevents" in f.name:
                    symlink = tb_logs / f.name
                    if not symlink.exists():
                        symlink.symlink_to(f)

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._merge_logs(trainer, pl_module)

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        self._merge_logs(trainer, pl_module)
