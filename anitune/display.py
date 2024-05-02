import json
import pickle
import typing_extensions as tpx

from typer import Option
from rich.table import Table

from anitune.console import console
from anitune.utils import (
    _TRAIN_PATH,
    _FTUNE_PATH,
    _BATCH_PATH,
    _ENSEMBLE_PATH,
)


def simplify_metric(arg: str) -> str:
    return (
        arg.replace("valid_", "")
        .replace("train_", "")
        .replace("mae_energies", "Emae")
        .replace("rmse_energies", "E")
        .replace("mae_forces", "Fmae")
        .replace("rmse_forces", "F")
        .replace("|ang", "/ang")
        .replace("|mol", "/mol")
    )


def ls(
    sizes: tpx.Annotated[
        bool,
        Option(
            "-s/-S",
            "--sizes/--no-sizes",
            help="Show file sizes",
        ),
    ] = False,
    best: tpx.Annotated[
        bool,
        Option(
            "-k/-K",
            "--best/--no-best",
            help="Show best metrics",
        ),
    ] = True,
    latest: tpx.Annotated[
        bool,
        Option(
            "-c/-C",
            "--current/--no-current",
            help="Show current metrics",
        ),
    ] = False,
    mae: tpx.Annotated[
        bool,
        Option(
            "-m/-M",
            "--mae/--no-mae",
            help="Show MAE metrics",
        ),
    ] = False,
    hartree: tpx.Annotated[
        bool,
        Option(
            "--hartree/--no-hartree",
            help="Show metrics in hartrees",
        ),
    ] = False,
) -> None:
    batch = sorted(_BATCH_PATH.iterdir())
    train = sorted(_TRAIN_PATH.iterdir())
    ftune = sorted(_FTUNE_PATH.iterdir())
    ensemble = sorted(_ENSEMBLE_PATH.iterdir())
    if batch:
        table = Table(title="Batched datasets", box=None)
        table.add_column("", style="magenta")
        table.add_column("data-name", style="magenta")
        table.add_column("divs", style="magenta")
        table.add_column("builtin-src")
        table.add_column("other-src")
        table.add_column("lot")
        table.add_column("conformers")
        table.add_column("symbols")
        table.add_column("properties")
        table.add_column("batch-size")
        table.add_column("batch-seed")
        if sizes:
            table.add_column("size (GB)")
        for j, p in enumerate(batch):
            try:
                with open(p / "ds_config.pkl", mode="rb") as fb:
                    ds_config = pickle.load(fb)
                with open(p / "creation_log.json", mode="rt") as ft:
                    ds_log = json.load(ft)

                row_args = [
                    f"[bold]{j}[/bold]",
                    p.name,
                    (
                        f"{ds_config.folds}-folds"
                        if ds_config.folds is not None
                        else f"train:{ds_config.train_frac} valid:{ds_config.validation_frac}"
                    ),
                    " ".join(ds_config.data_names) or "--",
                    " ".join((p.stem for p in ds_config.src_paths)) or "--",
                    ds_config.lot,
                    str(ds_log["num_conformers"]),
                    " ".join(ds_log["symbols"]),
                    " ".join(ds_log["properties"]),
                    str(ds_config.batch_size),
                    str(ds_config.shuffle_seed),
                ]
                if sizes:
                    size = sum(f.stat().st_size for f in p.glob("**/*") if f.is_file())
                    row_args.append(format(size / 1024**3, ".1f"))
            except Exception:
                row_args = [
                    f"[bold]{j}[/bold]",
                    p.name,
                ]
                row_args.extend(["?"] * 9)
                if sizes:
                    row_args.append("?")
            table.add_row(*row_args)
        console.print(table)
    else:
        console.print("(No batched datasets found)")

    if train:
        table = Table(title="Training runs", box=None)
        table.add_column("", style="green")
        table.add_column("run-name", style="green")
        table.add_column("data|div", style="magenta")
        table.add_column("builder")
        table.add_column("wd")
        table.add_column("lr")
        table.add_column("epoch(best)")
        if best:
            table.add_column("best-valid")
            table.add_column("best-train")
        if latest:
            table.add_column("curr-valid")
            table.add_column("curr-train")
        for j, p in enumerate(train):
            try:
                with open(p / "config.pkl", mode="rb") as fb:
                    config = pickle.load(fb)
                with open((p / "best-model") / "metrics.pkl", mode="rb") as fb:
                    metrics = pickle.load(fb)
                    best_epoch = metrics.pop("epoch")
                with open((p / "latest-model") / "metrics.pkl", mode="rb") as fb:
                    latest_metrics = pickle.load(fb)
                    epoch = latest_metrics.pop("epoch")
                row_args = [
                    f"[bold]{j}[/bold]",
                    p.name,
                    f"{config.ds.path.name}|{config.ds.fold_idx}",
                    config.model.builder,
                    f"{config.optim.weight_decay:.0e}",
                    f"{config.optim.lr:.0e}",
                    f"{epoch}({best_epoch})",
                ]
                if best:
                    if not mae:
                        metrics = {k: v for k, v in metrics.items() if "mae" not in k}
                    if not hartree:
                        metrics = {
                            k: v
                            for k, v in metrics.items()
                            if not (
                                ("energies" in k and "kcal" not in k)
                                or ("forces" in k and "kcal" not in k)
                            )
                        }
                    else:
                        metrics = {
                            k: v
                            for k, v in metrics.items()
                            if not (
                                ("energies" in k and "kcal" in k)
                                or ("forces" in k and "kcal" in k)
                            )
                        }
                    row_args.extend(
                        [
                            " ".join(
                                f"{simplify_metric(k)}={v:.2f}"
                                for k, v in metrics.items()
                                if "valid" in k
                            ),
                            " ".join(
                                f"{simplify_metric(k)}={v:.2f}"
                                for k, v in metrics.items()
                                if "train" in k
                            ),
                        ]
                    )
                if latest:
                    if not mae:
                        latest_metrics = {
                            k: v for k, v in latest_metrics.items() if "mae" not in k
                        }
                    if not hartree:
                        latest_metrics = {
                            k: v
                            for k, v in latest_metrics.items()
                            if not (
                                ("energies" in k and "kcal" not in k)
                                or ("forces" in k and "kcal" not in k)
                            )
                        }
                    else:
                        latest_metrics = {
                            k: v
                            for k, v in latest_metrics.items()
                            if not (
                                ("energies" in k and "kcal" in k)
                                or ("forces" in k and "kcal" in k)
                            )
                        }
                    row_args.extend(
                        [
                            " ".join(
                                f"{simplify_metric(k)}={v:.2f}"
                                for k, v in latest_metrics.items()
                                if "valid" in k
                            ),
                            " ".join(
                                f"{simplify_metric(k)}={v:.2f}"
                                for k, v in latest_metrics.items()
                                if "train" in k
                            ),
                        ]
                    )
            except Exception:
                row_args = [
                    f"[bold]{j}[/bold]",
                    p.name,
                ]
                row_args.extend(["?"] * 5)
            table.add_row(*row_args)
        console.print(table)
    else:
        console.print("(No training runs found)")
    console.print()
    if ftune:
        table = Table(title="Finetuning runs", box=None)
        table.add_column("", style="blue")
        table.add_column("run-name", style="blue")
        table.add_column("data|div", style="magenta")
        table.add_column("from", style="green")
        table.add_column("head")
        table.add_column("wd")
        table.add_column("head|bbone-lr")
        table.add_column("epoch(best)")
        if best:
            table.add_column("best-valid")
            table.add_column("best-train")
        if latest:
            table.add_column("curr-valid")
            table.add_column("curr-train")
        for j, p in enumerate(ftune):
            try:
                with open(p / "config.pkl", mode="rb") as fb:
                    config = pickle.load(fb)
                with open((p / "best-model") / "metrics.pkl", mode="rb") as fb:
                    metrics = pickle.load(fb)
                    best_epoch = metrics.pop("epoch")
                with open((p / "latest-model") / "metrics.pkl", mode="rb") as fb:
                    latest_metrics = pickle.load(fb)
                    epoch = latest_metrics.pop("epoch")
                row_args = [
                    f"[bold]{j}[/bold]",
                    p.name,
                    f"{config.ds.path.name}|{config.ds.fold_idx}",
                    config.ftune.pretrained_name,
                    str(config.ftune.num_head_layers),
                    f"{config.optim.weight_decay:.0e}",
                    f"{config.optim.lr:.0e}|{config.ftune.backbone_lr:.0e}",
                    f"{epoch}({best_epoch})",
                ]
                if best:
                    if not mae:
                        metrics = {k: v for k, v in metrics.items() if "mae" not in k}
                    if not hartree:
                        metrics = {
                            k: v
                            for k, v in metrics.items()
                            if not (
                                ("energies" in k and "kcal" not in k)
                                or ("forces" in k and "kcal" not in k)
                            )
                        }
                    else:
                        metrics = {
                            k: v
                            for k, v in metrics.items()
                            if not (
                                ("energies" in k and "kcal" in k)
                                or ("forces" in k and "kcal" in k)
                            )
                        }
                    row_args.extend(
                        [
                            " ".join(
                                f"{simplify_metric(k)}={v:.2f}"
                                for k, v in metrics.items()
                                if "valid" in k
                            ),
                            " ".join(
                                f"{simplify_metric(k)}={v:.2f}"
                                for k, v in metrics.items()
                                if "train" in k
                            ),
                        ]
                    )
                if latest:
                    if not mae:
                        latest_metrics = {
                            k: v for k, v in latest_metrics.items() if "mae" not in k
                        }
                    if not hartree:
                        latest_metrics = {
                            k: v
                            for k, v in latest_metrics.items()
                            if not (
                                ("energies" in k and "kcal" not in k)
                                or ("forces" in k and "kcal" not in k)
                            )
                        }
                    else:
                        latest_metrics = {
                            k: v
                            for k, v in latest_metrics.items()
                            if not (
                                ("energies" in k and "kcal" in k)
                                or ("forces" in k and "kcal" in k)
                            )
                        }
                    row_args.extend(
                        [
                            " ".join(
                                f"{simplify_metric(k)}={v:.2f}"
                                for k, v in latest_metrics.items()
                                if "valid" in k
                            ),
                            " ".join(
                                f"{simplify_metric(k)}={v:.2f}"
                                for k, v in latest_metrics.items()
                                if "train" in k
                            ),
                        ]
                    )
            except Exception:
                row_args = [
                    f"[bold]{j}[/bold]",
                    p.name,
                ]
                row_args.extend(["?"] * 7)
            table.add_row(*row_args)
        console.print(table)
    else:
        console.print("(No finetuning runs found)")
    console.print()
    if ensemble:
        table = Table(title="Ensembles", box=None)
        table.add_column("", style="cyan")
        table.add_column("name", style="cyan")
        table.add_column("ftune-src", style="blue")
        table.add_column("train-src", style="green")
        table.add_column("num-networks")
        for j, p in enumerate(ensemble):
            with open(p / "src_config.pkl", mode="rb") as fb:
                config = pickle.load(fb)
            row_args = [
                f"[bold]{j}[/bold]",
                p.name,
                " ".join(config["ftune-src"]) or "--",
                " ".join(config["train-src"]) or "--",
                str(config["num"]),
            ]
            table.add_row(*row_args)
        console.print(table)
    else:
        console.print("(No ensembles found)")
    console.print()
