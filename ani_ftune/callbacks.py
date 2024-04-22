from pathlib import Path
from lightning.pytorch.callbacks import Callback
from lightning import Trainer, LightningModule


class MergeTensorBoardLogs(Callback):
    def __init__(self, src: str, dest: str) -> None:
        super().__init__()
        self._src = src
        self._dest = dest

    def _merge_logs(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        trainer_root = Path(trainer.default_root_dir).resolve()
        tb_versioned_logs = trainer_root / self._src
        if (not tb_versioned_logs.is_dir()) or (not any(tb_versioned_logs.iterdir())):
            return

        tb_logs = trainer_root / self._dest
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
