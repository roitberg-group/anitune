r"""Command line interface entrypoints"""

import json
import warnings
import pickle
import shutil
import typing as tp
import typing_extensions as tpx
from pathlib import Path

from typer import Option, Typer
from rich.table import Table

from ani_ftune.console import console
from ani_ftune.utils import DiskData, select_paths
from ani_ftune.lit_training import train_from_scratch
from ani_ftune.config import (
    load_state_dict,
    FinetuneConfig,
    TrainConfig,
    DatasetConfig,
    AccelConfig,
    ModelConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    _DEBUG_TRAIN_PATH,
    _DEBUG_FTUNE_PATH,
    _TRAIN_PATH,
    _FTUNE_PATH,
    _BATCH_PATH,
)

app = Typer(
    rich_markup_mode="markdown",
    help=r"""## ANI

    Utility for generating a fine-tuned models from pre trained ANI style models,
    given a set of reference structures.
    """,
)


@app.command(help="Prebatch a dataset")
def prebatch(
    name: tpx.Annotated[
        str,
        Option(
            "-n",
            "--name",
            help="Name for the prebatched dataset",
        ),
    ] = "",
    lot: tpx.Annotated[
        str,
        Option(
            "--lot",
            help="Level of theory",
        ),
    ] = "wb97x-631gd",
    _src_paths: tpx.Annotated[
        tp.Optional[tp.List[Path]],
        Option(
            "-s",
            "--data-path",
            help="Paths to data to fine-tune the model with",
        ),
    ] = None,
    _data_names: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "-d",
            "--data-name",
            help="Builtin dataset name",
        ),
    ] = None,
    _properties: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "-p",
            "--property",
            help="Properties to prebatch, all by default",
        ),
    ] = None,
    batch_size: tpx.Annotated[
        int,
        Option(
            "--batch-size",
            help="Batch size",
        ),
    ] = 2560,
    folds: tpx.Annotated[
        tp.Optional[int],
        Option(
            "--folds",
            help="Number of folds to train an ensemble to",
        ),
    ] = None,
    train_frac: tpx.Annotated[
        float,
        Option(
            "--tf",
            "--train-frac",
            help="Training set fraction",
        ),
    ] = 0.8,
    validation_frac: tpx.Annotated[
        float,
        Option(
            "--vf",
            "--validation-frac",
            help="Validation set fraction",
        ),
    ] = 0.2,
    data_seed: tpx.Annotated[
        int,
        Option(
            "--data-seed",
            help="Seed for dataset prebatching",
        ),
    ] = 1234,
) -> None:
    from ani_ftune.batching import batch

    properties = () if _properties is None else tuple(sorted(_properties))
    src_paths = () if _src_paths is None else tuple(sorted(_src_paths))
    data_names = () if _data_names is None else tuple(sorted(_data_names))
    ds = DatasetConfig(
        label=name,
        lot=lot,
        properties=properties,
        data_names=data_names,
        src_paths=src_paths,
        batch_size=batch_size,
        fold_idx=-1,
        folds=folds,
        validation_frac=validation_frac,
        train_frac=train_frac,
        shuffle_seed=data_seed,
    )
    batch(ds, max_batches_per_packet=100)


@app.command(help="Clean debug runs")
def clean() -> None:
    if any(_DEBUG_TRAIN_PATH.iterdir()):
        shutil.rmtree(_DEBUG_TRAIN_PATH)
        _DEBUG_TRAIN_PATH.mkdir(exist_ok=True, parents=True)
        console.print("Cleaned all training debug runs")
    else:
        console.print("No debug training runs to clean")
    console.print()
    if any(_DEBUG_FTUNE_PATH.iterdir()):
        shutil.rmtree(_DEBUG_FTUNE_PATH)
        _DEBUG_FTUNE_PATH.mkdir(exist_ok=True, parents=True)
        console.print("Cleaned all finetuning debug runs")
    else:
        console.print("No debug finetuning runs to clean")


@app.command(help="Continue a previously started training")
def restart(
    ftune_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-f",
            "--ftune-run",
            help="Name or idx of the run",
        ),
    ] = "",
    ptrain_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-t",
            "--train-run",
            help="Name or idx of the run",
        ),
    ] = "",
    debug: tpx.Annotated[
        bool,
        Option(
            "-g/-G",
            "--debug/--no-debug",
            help="Restart a debug run",
        ),
    ] = False,
) -> None:
    if (
        ftune_name_or_idx
        and ptrain_name_or_idx
        or not (ftune_name_or_idx or ptrain_name_or_idx)
    ):
        raise ValueError("One and only one of -f and -t should be specified")
    name_or_idx = ftune_name_or_idx or ptrain_name_or_idx
    if debug:
        kind = DiskData.DEBUG_FTUNE if ftune_name_or_idx else DiskData.DEBUG_TRAIN
    else:
        kind = DiskData.FTUNE if ftune_name_or_idx else DiskData.TRAIN

    path = select_paths((name_or_idx,), kind=kind)[0] / "config.pkl"
    if not path.is_file():
        raise ValueError(f"{path} is not a file dir")

    with open(path, mode="rb") as f:
        config = pickle.load(f)
    train_from_scratch(config, restart=True)


@app.command(help="Display training and finetuning runs")
def ls(
    sizes: tpx.Annotated[
        bool,
        Option(
            "-l",
            "--extra/--no-extra",
            help="Show file sizes",
        ),
    ] = False,
) -> None:
    batch = sorted(_BATCH_PATH.iterdir())
    train = sorted(_TRAIN_PATH.iterdir())
    ftune = sorted(_FTUNE_PATH.iterdir())
    debug_train = sorted(_DEBUG_TRAIN_PATH.iterdir())
    debug_ftune = sorted(_DEBUG_FTUNE_PATH.iterdir())

    if train or debug_train:
        console.print("Training runs:")
        for j, p in enumerate(train):
            console.print(f"{j}. {p.name}", style="green", highlight=False)
        for j, p in enumerate(debug_train):
            console.print(f"(debug) {j}. {p.name}", style="yellow", highlight=False)
    else:
        console.print("(No training runs found)")

    console.print()
    if ftune or debug_ftune:
        console.print("Finetuning runs:")
        for j, p in enumerate(ftune):
            console.print(f"[bold]{j}[/bold]. {p.name}", style="blue", highlight=False)
        for j, p in enumerate(debug_ftune):
            console.print(
                f"(debug) [bold]{j}[/bold]. {p.name}", style="yellow", highlight=False
            )
    else:
        console.print("(No finetuning runs found)")

    console.print()
    if batch:
        table = Table(title="Batched datasets", box=None)
        table.add_column("index")
        table.add_column("name", style="magenta")
        table.add_column("builtin-src", style="magenta")
        table.add_column("other-src", style="magenta")
        table.add_column("lot")
        table.add_column("conformers")
        table.add_column("symbols")
        table.add_column("properties")
        table.add_column("batch-size")
        table.add_column("batching-seed")
        table.add_column("divisions")
        if sizes:
            table.add_column("size (GB)")
        for j, p in enumerate(batch):
            try:
                with open(p / "ds_config.pkl", mode="rb") as fb:
                    ds_config = pickle.load(fb)
                with open(p / "creation_log.json", mode="rt") as ft:
                    ds_log = json.load(ft)

                row_args = [
                    f"[bold][magenta]{j}[/magenta][/bold]",
                    p.name,
                    "|".join(ds_config.data_names) or "--",
                    "|".join((p.stem for p in ds_config.src_paths)) or "--",
                    ds_config.lot,
                    str(ds_log["num_conformers"]),
                    ",".join(ds_log["symbols"]),
                    "|".join(ds_log["properties"]),
                    str(ds_config.batch_size),
                    str(ds_config.shuffle_seed),
                    (
                        f"{ds_config.folds}-folds"
                        if ds_config.folds is not None
                        else f"train:{ds_config.train_frac}, valid:{ds_config.validation_frac}"
                    ),
                ]
                if sizes:
                    size = sum(f.stat().st_size for f in p.glob("**/*") if f.is_file())
                    row_args.append(format(size / 1024**3, ".1f"))

            except Exception:
                row_args = [
                    f"[bold][magenta]{j}[/magenta][/bold]",
                    p.name,
                ]
                row_args.extend(["?"] * 9)
                if sizes:
                    row_args.append("?")
            table.add_row(*row_args)
        console.print(table)
    else:
        console.print("(No batched datasets found)")


@app.command(help="Delete specific training or finetuning run")
def rm(
    debug_ftune_name_or_idx: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "--df",
            "--debug-ftune-run",
            help="Name or idx of the run debug finetune",
        ),
    ] = None,
    debug_ptrain_name_or_idx: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "--dt",
            "--debug-train-run",
            help="Name or idx of the run debug pretrain",
        ),
    ] = None,
    ftune_name_or_idx: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "-f",
            "--ftune-run",
            help="Name or idx of the finetune run",
        ),
    ] = None,
    ptrain_name_or_idx: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "-t",
            "--train-run",
            help="Name or idx of the pretrain run",
        ),
    ] = None,
    batch_name_or_idx: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "-b",
            help="Name or idx of the batched dataset",
        ),
    ] = None,
) -> None:
    for selectors, dkind in zip(
        (
            debug_ftune_name_or_idx,
            debug_ptrain_name_or_idx,
            ftune_name_or_idx,
            ptrain_name_or_idx,
            batch_name_or_idx,
        ),
        (
            DiskData.DEBUG_FTUNE,
            DiskData.DEBUG_TRAIN,
            DiskData.FTUNE,
            DiskData.TRAIN,
            DiskData.BATCH,
        ),
    ):
        if selectors is not None:
            paths = select_paths(selectors, kind=dkind)
            for p in paths:
                shutil.rmtree(p)
                console.print(f"Removed {p.name}")
            console.print()


@app.command(help="Compare the params of a ftuned model and the original model")
def compare(
    pretrained_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-t",
            help="Name or idx of the pretrained run",
        ),
    ],
    ftuned_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-f",
            help="Name or idx of the finetuned run",
        ),
    ],
    debug: tpx.Annotated[
        bool,
        Option(
            "-g/-G",
            "--debug/--no-debug",
            help="Run debug",
        ),
    ] = False,
) -> None:
    if pretrained_name_or_idx.split(":")[0] in (
        "ani1x",
        "ani2x",
        "ani1ccx",
        "anidr",
        "aniala",
    ):
        model_name, idx = pretrained_name_or_idx.split(":")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from torchani import assembler

        model = getattr(assembler, model_name.replace("ani", "ANI"))()[int(idx)]
        pretrained_state_dict = model.state_dict()
    else:
        pretrained_path = (
            select_paths(
                (pretrained_name_or_idx,),
                kind=DiskData.TRAIN if not debug else DiskData.DEBUG_TRAIN,
            )[0]
            / "best-model"
        )
        pretrained_state_dict = load_state_dict(pretrained_path / "best.ckpt")

    ftuned_path = (
        select_paths(
            (ftuned_name_or_idx,),
            kind=DiskData.FTUNE if not debug else DiskData.DEBUG_FTUNE,
        )[0]
        / "best-model"
    )
    ftuned_state_dict = load_state_dict(ftuned_path / "best.ckpt")
    for k in pretrained_state_dict:
        if "weight" in k or "bias" in k:
            pretrained_param = pretrained_state_dict[k]
            ftuned_param = ftuned_state_dict[k]
            diff = pretrained_param - ftuned_param
            if (diff == 0.0).all():
                console.print(f"No difference found for param {k}")
            else:
                diff = diff.abs()
                console.print(f"Difference found for param {k}")
                console.print(f"Min abs diff: {diff.min()}")
                console.print(f"Mean abs diff: {diff.mean()}")
                console.print(f"Max abs diff: {diff.max()}")
            console.print()


@app.command(
    help="Benchmark a (usually ftuned) ANI model, compare results with the original"
)
def bench(
    pretrained_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-t",
            help="Name or idx of the pretrained run",
        ),
    ],
    ftuned_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-f",
            help="Name or idx of the finetuned run",
        ),
    ],
    test_paths: tpx.Annotated[
        tp.List[Path],
        Option(
            "-t",
            "--test-path",
            help="Paths to test data",
        ),
    ],
) -> None:
    raise NotImplementedError("Not implemented yet")


@app.command(help="Train an ANI-style model from scratch")
def train(
    batch_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-b",
            help="Name or idx of the batched dataset",
        ),
    ],
    name: tpx.Annotated[
        str,
        Option(
            "-n",
            "--name",
            help="Name of the run",
        ),
    ] = "train",
    lot: tpx.Annotated[
        str,
        Option(
            "--lot",
            help="Level of theory",
        ),
    ] = "wb97x-631gd",
    builder: tpx.Annotated[
        str,
        Option(
            "-r",
            "--builder",
            help="Builder function",
        ),
    ] = "FlexANI2",
    repulsion_xtb: tpx.Annotated[
        bool,
        Option(
            "--repulsion-xtb/--no-repulsion-xtb",
            help="Add a repulsion XTB term to the model",
        ),
    ] = False,
    dispersion_2body_d3: tpx.Annotated[
        bool,
        Option(
            "--dispersion-2body-d3/--no-dispersion-2body-d3",
            help="Add 2-body dispersion D3 term to the model",
        ),
    ] = False,
    _fold_idx: tpx.Annotated[
        str,
        Option(
            "--fold-idx",
            help="Fold idx",
        ),
    ] = "single",
    lr: tpx.Annotated[
        float,
        Option(
            "--lr",
            help="Learning rate for optimizer",
        ),
    ] = 0.5e-3,
    weight_decay: tpx.Annotated[
        float,
        Option(
            "--wd",
            help="Weight decay for optimizer",
        ),
    ] = 1e-7,
    xc: tpx.Annotated[
        bool,
        Option(
            "--xc/--no-xc",
            help="Train to exchange-correlation energies",
        ),
    ] = False,
    sqrt_atoms: tpx.Annotated[
        bool,
        Option(
            "--sqrt-atoms/--no-sqrt-atoms",
            help="Use sqrt atoms in energies",
        ),
    ] = False,
    energies: tpx.Annotated[
        float,
        Option(
            "-e",
            "--energies-factor",
            help="Train with energies",
        ),
    ] = 1.0,
    forces: tpx.Annotated[
        float,
        Option(
            "-f",
            "--forces-factor",
            help="Train with forces",
        ),
    ] = 0.0,
    dipoles: tpx.Annotated[
        float,
        Option(
            "-m",
            "--dipoles-factor",
            help="Train with dipoles",
        ),
    ] = 0.0,
    atomic_charges: tpx.Annotated[
        float,
        Option(
            "-a",
            "--atomic-charges-factor",
            help="Train with atomic charges",
        ),
    ] = 0.0,
    total_charge: tpx.Annotated[
        float,
        Option(
            "-q",
            "--total-charge-factor",
            help="Train with total charge",
        ),
    ] = 0.0,
    debug: tpx.Annotated[
        bool,
        Option(
            "-g/-G",
            "--debug/--no-debug",
            help="Run debug",
        ),
    ] = False,
    limit: tpx.Annotated[
        tp.Optional[int],
        Option(
            "-l",
            "--limit",
            help="Limit number of batches or percent",
        ),
    ] = None,
    deterministic: tpx.Annotated[
        bool,
        Option(
            "--deterministic/--no-deterministic",
            help="Run deterministic training (has a performance penalty)",
        ),
    ] = False,
    detect_anomaly: tpx.Annotated[
        bool,
        Option(
            "--detect-anomaly/--no-detect-anomaly",
            help="Detect anomalies during training (has a performance penalty)",
        ),
    ] = False,
    use_cuda_ops: tpx.Annotated[
        bool,
        Option(
            "--cuda-ops/--no-cuda-ops",
            help="Use cuda acceleration",
        ),
    ] = False,
    data_seed: tpx.Annotated[
        int,
        Option(
            "--data-seed",
            help="Seed for dataset prebatching",
        ),
    ] = 1234,
) -> None:
    batched_dataset_path = select_paths((batch_name_or_idx,), kind=DiskData.BATCH)[0]
    ds_config_path = batched_dataset_path / "ds_config.pkl"
    with open(ds_config_path, mode="rb") as f:
        ds_config = pickle.load(f)

    fold_idx: tp.Union[str, int]
    try:
        fold_idx = int(_fold_idx)
    except ValueError:
        fold_idx = _fold_idx

    ds_config.fold_idx = fold_idx
    if debug:
        if limit is None:
            console.print("Setting train limit to 10 batches for debugging")
            limit = 10
        console.print("Setting deterministic training for debugging purposes")
        deterministic = True
        console.print("Setting anomaly detection for debugging purposes")
        detect_anomaly = True

    terms_and_factors: tp.List[tp.Tuple[str, float]] = []
    if energies > 0.0:
        label = "EnergiesXC" if xc else "Energies"
        terms_and_factors.append(
            (label if not sqrt_atoms else f"{label}SqrtAtoms", energies)
        )
    if forces > 0.0:
        terms_and_factors.append(("Forces", forces))
    if dipoles > 0.0:
        terms_and_factors.append(("Dipoles", dipoles))
    if atomic_charges > 0.0:
        terms_and_factors.append(("AtomicCharges", atomic_charges))
    if total_charge > 0.0:
        terms_and_factors.append(("TotalCharge", total_charge))

    config = TrainConfig(
        name=name,
        debug=debug,
        ds=ds_config,
        accel=AccelConfig(
            max_batches_per_packet=100,
            limit=limit,
            deterministic=deterministic,
            detect_anomaly=detect_anomaly,
            use_cuda_ops=use_cuda_ops,
        ),
        model=ModelConfig(
            builder=builder,
            kwargs=(
                ("dispersion_2body_d3", dispersion_2body_d3),
                ("repulsion_xtb", repulsion_xtb),
            ),
        ),
        loss=LossConfig(
            terms_and_factors=tuple(terms_and_factors),
        ),
        optim=OptimizerConfig(lr=lr, weight_decay=weight_decay),
        scheduler=SchedulerConfig(),
    )
    train_from_scratch(config)


@app.command(help="Fine tune a pretrained ANI model")
def ftune(
    batch_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-b",
            help="Name or idx of the batched dataset",
        ),
    ],
    name_or_idx: tpx.Annotated[
        str,
        Option(
            "-t",
            "--train-run",
            help="Name or idx of the pretrained run, alternatively, ani1x:idx, ani2x:idx, etc. is also supported",
        ),
    ],
    lot: tpx.Annotated[
        str,
        Option(
            "--lot",
            help="Level of theory",
        ),
    ] = "wb97x-631gd",
    name: tpx.Annotated[
        str,
        Option(
            "-n",
            "--name",
            help="Name of the run",
        ),
    ] = "ftune",
    num_head_layers: tpx.Annotated[
        int,
        Option(
            "--num-head-layers",
            help="Number of layers to use as model head",
        ),
    ] = 1,
    # Loss and optimizer specification
    head_lr: tpx.Annotated[
        float,
        Option(
            "--lr",
            help="Learning rate for head of model",
        ),
    ] = 0.5e-4,
    backbone_lr: tpx.Annotated[
        float,
        Option(
            "--backbone-lr",
            help="Learning rate for backbone of model (may be zero)",
        ),
    ] = 0.0,
    weight_decay: tpx.Annotated[
        float,
        Option(
            "--wd",
            help="Weight decay",
        ),
    ] = 1e-7,
    xc: tpx.Annotated[
        bool,
        Option(
            "--xc/--no-xc",
            help="Train to exchange-correlation energies",
        ),
    ] = False,
    sqrt_atoms: tpx.Annotated[
        bool,
        Option(
            "--sqrt-atoms/--no-sqrt-atoms",
            help="Use sqrt atoms in energies",
        ),
    ] = False,
    energies: tpx.Annotated[
        float,
        Option(
            "-e",
            "--energies-factor",
            help="Train with energies",
        ),
    ] = 1.0,
    forces: tpx.Annotated[
        float,
        Option(
            "-f",
            "--forces-factor",
            help="Train with forces",
        ),
    ] = 0.0,
    dipoles: tpx.Annotated[
        float,
        Option(
            "-m",
            "--dipoles-factor",
            help="Train with dipoles",
        ),
    ] = 0.0,
    atomic_charges: tpx.Annotated[
        float,
        Option(
            "-a",
            "--atomic-charges-factor",
            help="Train with atomic charges",
        ),
    ] = 0.0,
    total_charge: tpx.Annotated[
        float,
        Option(
            "-q",
            "--total-charge-factor",
            help="Train with total charge",
        ),
    ] = 0.0,
    deterministic: tpx.Annotated[
        bool,
        Option(
            "--deterministic/--no-deterministic",
            help="Run deterministic training (has a performance penalty)",
        ),
    ] = False,
    detect_anomaly: tpx.Annotated[
        bool,
        Option(
            "--detect-anomaly/--no-detect-anomaly",
            help="Detect anomalies during training (has a performance penalty)",
        ),
    ] = False,
    limit: tpx.Annotated[
        tp.Optional[int],
        Option(
            "-l",
            "--limit",
            help="Limit number of batches or percent",
        ),
    ] = None,
    debug: tpx.Annotated[
        bool,
        Option(
            "-g/-G",
            "--debug/--no-debug",
            help="Debug finetune run",
        ),
    ] = False,
    use_cuda_ops: tpx.Annotated[
        bool,
        Option(
            "--cuda-ops/--no-cuda-ops",
            help="Use cuda acceleration",
        ),
    ] = False,
) -> None:
    batched_dataset_path = select_paths((batch_name_or_idx,), kind=DiskData.BATCH)[0]
    ds_config_path = batched_dataset_path / "ds_config.pkl"
    with open(ds_config_path, mode="rb") as f:
        ds_config = pickle.load(f)
        ds_config.fold_idx = "single"

    if head_lr <= 0.0:
        raise ValueError(
            "Learning rate for the head of the model must be strictly positive"
        )
    if backbone_lr < 0.0:
        raise ValueError(
            "Learning rate for the body of the model must be positive or zero"
        )
    if backbone_lr > head_lr:
        raise ValueError("Backbone lr must be greater or equal to head lr")
    if num_head_layers < 1:
        raise ValueError("There must be at least one head layer")

    pretrain_builtin: bool
    if name_or_idx.split(":")[0] in ("ani1x", "ani2x", "ani1ccx", "anidr", "aniala"):
        from ani_ftune.model_builders import fetch_pretrained_config

        pretrained_config = fetch_pretrained_config(name_or_idx)
        pretrained_state_dict_path = None
        pretrain_builtin = True
    else:
        pretrained_path = select_paths(
            (name_or_idx,),
            kind=DiskData.TRAIN if not debug else DiskData.DEBUG_TRAIN,
        )[0]
        pretrained_config_path = pretrained_path / "config.pkl"
        if not pretrained_config_path.is_file():
            raise ValueError(f"{pretrained_config_path} is not a valid config file")

        with open(pretrained_config_path, mode="rb") as f:
            pretrained_config = pickle.load(f)
        pretrain_builtin = False

        pretrained_state_dict_path = (pretrained_path / "best-model") / "best.ckpt"

        if not pretrained_state_dict_path.is_file():
            raise ValueError(f"{pretrained_state_dict_path} is not a valid checkpoint")

    if pretrain_builtin:
        run_name = f"{name}-from_{name_or_idx}"
    else:
        run_name = f"{name}-from_{pretrained_config.ds.fold_idx}"

    terms_and_factors: tp.List[tp.Tuple[str, float]] = []
    if energies > 0.0:
        label = "EnergiesXC" if xc else "Energies"
        terms_and_factors.append(
            (label if not sqrt_atoms else f"{label}SqrtAtoms", energies)
        )
    if forces > 0.0:
        terms_and_factors.append(("Forces", forces))
    if dipoles > 0.0:
        terms_and_factors.append(("Dipoles", dipoles))
    if atomic_charges > 0.0:
        terms_and_factors.append(("AtomicCharges", atomic_charges))
    if total_charge > 0.0:
        terms_and_factors.append(("TotalCharge", total_charge))

    config = TrainConfig(
        name=run_name,
        ds=ds_config,
        accel=AccelConfig(
            max_batches_per_packet=100,
            limit=limit,
            deterministic=deterministic,
            detect_anomaly=detect_anomaly,
            use_cuda_ops=use_cuda_ops,
        ),
        model=pretrained_config.model,
        loss=LossConfig(
            terms_and_factors=tuple(terms_and_factors),
        ),
        optim=OptimizerConfig(
            lr=head_lr,
            weight_decay=weight_decay,
        ),
        scheduler=SchedulerConfig(),
        ftune=FinetuneConfig(
            state_dict_path=pretrained_state_dict_path,
            num_head_layers=num_head_layers,
            backbone_lr=backbone_lr,
        ),
    )
    train_from_scratch(config)
