r"""Command line interface entrypoints"""

import tempfile
import uuid
import subprocess
from copy import deepcopy
import hashlib
import shutil
import typing as tp
from typing import Optional
from typing_extensions import Annotated
from pathlib import Path

from typer import Argument, Option, Typer

from anitune.console import console
from anitune.paths import ENSEMBLE_PATH, DataKind, select_subdirs
from anitune.lit_training import train_lit_model
from anitune.config import (
    load_state_dict,
    FinetuneConfig,
    TrainConfig,
    DatasetConfig,
    AccelConfig,
    ModelConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    SrcConfig,
)
from anitune.display import ls
from anitune.defaults import (
    resolve_options,
    parse_scheduler_str,
    parse_optimizer_str,
)

app = Typer(
    rich_markup_mode="markdown",
    help=r"""
    ## ANI

    Utility for generating a fine-tuned models from pre trained ANI style
    models, given a set of reference structures
    """,
)


# TODO the ds part of this is broken?
def _fetch_builtin_config(name_or_idx: str) -> TrainConfig:
    name, idx = name_or_idx.split(":")
    config = TrainConfig()
    config.ds.fold_idx = idx
    symbols = ["H", "C", "N", "O"]
    if not (("1x" in name) or ("1ccx" in name)):
        symbols.extend(["S", "F", "Cl"])
    config.model = ModelConfig(
        builtin=True,
        arch_fn=name,
        options={"model_index": int(idx)},
        symbols=symbols,
    )
    return config


@app.command(help="Generate an ensemble from a set of models")
def ensemble(
    name: Annotated[
        str,
        Option("-n", "--ens-name", help="Name of ensemble"),
    ] = "ensemble",
    ftune_names_or_idxs: Annotated[
        Optional[tp.List[str]],
        Option("-f", "--ftune-run", help="Name|idx of train run"),
    ] = None,
    ptrain_names_or_idxs: Annotated[
        Optional[tp.List[str]],
        Option("-t", "--train-run", help="Name|idx of ftune run"),
    ] = None,
) -> None:
    if ptrain_names_or_idxs is None:
        ptrain_names_or_idxs = []
    if ftune_names_or_idxs is None:
        ftune_names_or_idxs = []
    ptrain_paths = select_subdirs(
        ptrain_names_or_idxs,
        kind=DataKind.TRAIN,
    )
    ftune_paths = select_subdirs(
        ftune_names_or_idxs,
        kind=DataKind.FTUNE,
    )
    paths = deepcopy(ptrain_paths)
    paths.extend(ftune_paths)
    hasher = hashlib.shake_128()
    for p in paths:
        hasher.update(p.name.encode("utf-8"))
    paths = [(p / "best-model") / "best.ckpt" for p in paths]

    import torch
    from torchani.utils import merge_state_dicts

    state_dict = merge_state_dicts(paths)

    _hash = hasher.hexdigest(4)
    path = ENSEMBLE_PATH / f"{name}-{_hash}"
    path.mkdir(exist_ok=True, parents=True)
    src_config = SrcConfig(
        train_src=list(p.name for p in ptrain_paths),
        ftune_src=list(p.name for p in ftune_paths),
    )
    src_config.to_json_file(path / "src_config.json")
    torch.save(state_dict, path / "model.pt")


@app.command(help="Generate a pre-batched dataset from one or more ANI datasets")
def batch(
    name: Annotated[
        str,
        Option("-n", "--output-batched-ds-name", help="Name of output batched dataset"),
    ] = "",
    lot: Annotated[
        str,
        Option("-l", "--builtin-src-ds-lot", help="LoT of built-in ds to src from"),
    ] = "wb97x-631gd",
    data_names: Annotated[
        Optional[tp.List[str]],
        Option("-b", "--builtin-src-ds-name", help="Built-in ANI ds to src from"),
    ] = None,
    src_paths: Annotated[
        Optional[tp.List[Path]],
        Option("-s", "--src-ds-path", help="Paths to non-builtin ds to src from"),
    ] = None,
    properties: Annotated[
        Optional[tp.List[str]],
        Option("-p", "--property", help="Properties to batch. All by default"),
    ] = None,
    batch_size: Annotated[
        int,
        Option("--batch-size", help="Batch size"),
    ] = 2560,
    folds: Annotated[
        Optional[int],
        Option("--folds", help="Num. of folds. Useful for training ensembles"),
    ] = None,
    train_frac: Annotated[
        float, Option("--tf", "--train-frac", help="Training set fraction")
    ] = 0.8,
    divs_seed: Annotated[
        int, Option("--divs-seed", help="Seed for divisions (train, validation, etc)")
    ] = 1234,
    batch_seed: Annotated[
        int,
        Option("--shuffle-seed", help="Seed for shuffling divisions before batching"),
    ] = 1234,
) -> None:
    from anitune.batching import batch_data

    ds = DatasetConfig(
        label=name,
        lot=lot,
        properties=[] if properties is None else sorted(properties),
        data_names=[] if data_names is None else sorted(data_names),
        raw_src_paths=[] if src_paths is None else sorted(map(str, src_paths)),
        batch_size=batch_size,
        fold_idx=-1,
        folds=folds,
        validation_frac=1.0 - train_frac,
        train_frac=train_frac,
        batch_seed=batch_seed,
        divs_seed=divs_seed,
    )
    batch_data(ds, max_batches_per_packet=100)


@app.command(help="Continue a previously checkpointed run")
def restart(
    ftune_name_or_idx: Annotated[
        str, Option("-f", "--ftune-run", help="Name or idx of ftune run")
    ] = "",
    ptrain_name_or_idx: Annotated[
        str, Option("-t", "--train-run", help="Name or idx of train run")
    ] = "",
    max_epochs: Annotated[
        Optional[int],
        Option("--max-epochs", help="Max epochs to train"),
    ] = None,
    verbose: Annotated[bool, Option("-v/ ", "--verbose/ ")] = False,
) -> None:
    if (
        ftune_name_or_idx
        and ptrain_name_or_idx
        or not (ftune_name_or_idx or ptrain_name_or_idx)
    ):
        raise ValueError("One and only one of -f and -t should be specified")
    name_or_idx = ftune_name_or_idx or ptrain_name_or_idx
    kind = DataKind.FTUNE if ftune_name_or_idx else DataKind.TRAIN

    path = select_subdirs((name_or_idx,), kind=kind)[0] / "config.json"
    if not path.is_file():
        raise ValueError(f"{path} is not a file dir")

    config = TrainConfig.from_json_file(path)
    if max_epochs is not None:
        config.accel.max_epochs = max_epochs
    train_lit_model(config, restart=True, verbose=verbose)


ls = app.command(help="Display training and finetuning runs")(ls)


@app.command(help="Delete specific training or finetuning run")
def rm(
    ftune_id: Annotated[
        Optional[tp.List[str]],
        Option("-f", "--ftune-run", help="Name|idx of ftune run"),
    ] = None,
    train_id: Annotated[
        Optional[tp.List[str]],
        Option("-t", "--train-run", help="Name|idx of train run"),
    ] = None,
    batch_id: Annotated[
        Optional[tp.List[str]], Option("-b", help="Name|idx of batched dataset")
    ] = None,
    ensemble_id: Annotated[
        Optional[tp.List[str]], Option("-e", help="Name|idx of ensemble")
    ] = None,
) -> None:
    for selectors, dkind in zip(
        (ftune_id, train_id, batch_id, ensemble_id),
        (DataKind.FTUNE, DataKind.TRAIN, DataKind.BATCH, DataKind.ENSEMBLE),
    ):
        if selectors is not None:
            paths = select_subdirs(selectors, kind=dkind)
            for p in paths:
                shutil.rmtree(p)
                console.print(f"Removed {p.name}")
            console.print()


@app.command(help="Compare params of two models")
def compare(
    ftune_id: Annotated[
        str, Option("-f", "--ftune-run", help="Name|idx of ftune run")
    ] = "",
    train_id: Annotated[
        str, Option("-t", "--train-run", help="Name|idx of train run")
    ] = "",
) -> None:
    if (not (ftune_id or train_id)) or (ftune_id and train_id):
        raise ValueError("One and only one of -t or -f has to be specified")
    kind = DataKind.FTUNE if ftune_id else DataKind.TRAIN
    root = select_subdirs(
        (train_id or ftune_id,),
        kind=kind,
    )[0]
    trained_path = root / "best-model"
    trained_state_dict = load_state_dict(trained_path / "best.ckpt")
    init_path = root / "init-model"
    init_state_dict = load_state_dict(init_path / "init.ckpt")

    for k in init_state_dict:
        if "weight" in k or "bias" in k:
            pretrained_param = init_state_dict[k]
            ftuned_param = trained_state_dict[k]
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


@app.command(help="Train from scratch or finetune an ANI-style model")
def train(
    batch_name_or_idx: Annotated[str, Argument(help="Name|idx of the batched dataset")],
    fold_idx: Annotated[
        Optional[int],
        Option(
            "-i",
            "--fold-idx",
            help="Idx to use if training from folds",
            show_default=False,
        ),
    ] = None,
    name: Annotated[str, Option("-n", "--run-name", help="Name of run")] = "",
    auto_restart: Annotated[
        bool,
        Option("--auto-restart/ ", help="Auto restart runs that match a prev run"),
    ] = False,
    max_epochs: Annotated[
        int, Option("--max-epochs", help="Max epochs to train")
    ] = 1000,
    early_stop_patience: Annotated[
        int,
        Option(
            "--early-stop-patience",
            help="Max epochs without improving monitor metric before early stopping",
        ),
    ] = 50,
    # From-scratch specific config
    arch_fn: Annotated[
        str,
        Option(
            "-a",
            "--arch",
            help="Callable that creates the model",
            rich_help_panel="Arch",
        ),
    ] = "simple_ani",
    arch_options: Annotated[
        Optional[tp.List[str]],
        Option(
            "--ao",
            "--arch-opt",
            help="Options for arch fn, key=val fmt",
            rich_help_panel="Arch",
            show_default=False,
        ),
    ] = None,
    # LrSched config
    lrsched: Annotated[
        str,
        Option(
            "-s", "--sched", help="Type of lr-scheduler", rich_help_panel="LR scheduler"
        ),
    ] = "Plateau",
    lrsched_opts: Annotated[
        Optional[tp.List[str]],
        Option(
            "--so",
            "--sched-opt",
            help="Options for lr-scheduler, key=val fmt",
            rich_help_panel="LR scheduler",
            show_default=False,
        ),
    ] = None,
    # Optimizer config
    optim: Annotated[
        str,
        Option("-o", "--optim", help="Type of optimizer", rich_help_panel="Optimizer"),
    ] = "AdamW",
    optim_opts: Annotated[
        Optional[tp.List[str]],
        Option(
            "--oo",
            "--optim-opt",
            rich_help_panel="Optimizer",
            help="Options for optim, key=val fmt (lr, wd are separate)",
            show_default=False,
        ),
    ] = None,
    wd: Annotated[
        float,
        Option("--wd", help="Weight decay for optim", rich_help_panel="Optimizer"),
    ] = 1e-7,
    lr: Annotated[
        float,
        Option(
            "--lr",
            help="Initial lr. If ftune, used for the 'head'",
            rich_help_panel="Optimizer",
        ),
    ] = 1e-4,
    # Loss config
    xc: Annotated[
        bool, Option("--xc/ ", help="Train to XC energies", rich_help_panel="Loss")
    ] = False,
    no_sqrt_atoms: Annotated[
        bool,
        Option(
            "--no-sqrt-atoms/ ",
            help="Divide energy loss by atoms instead of sqrt(atoms)",
            rich_help_panel="Loss",
        ),
    ] = False,
    energies: Annotated[
        float, Option("-e", "--energies", help="Energy factor", rich_help_panel="Loss")
    ] = 1.0,
    forces: Annotated[
        float, Option("-f", "--forces", help="Force factor", rich_help_panel="Loss")
    ] = 0.0,
    dipoles: Annotated[
        float, Option("-m", "--dipoles", help="Dipole factor", rich_help_panel="Loss")
    ] = 0.0,
    atomic_charges: Annotated[
        float,
        Option(
            "-q",
            "--atomic-charges",
            help="Atomic charges factor",
            rich_help_panel="Loss",
        ),
    ] = 0.0,
    total_charge: Annotated[
        float,
        Option(
            "-Q", "--total-charge", help="Total charge factor", rich_help_panel="Loss"
        ),
    ] = 0.0,
    monitor: Annotated[
        str,
        Option(
            "--monitor",
            help="Loss label to monitor during training."
            " Format is 'valid/rmse_energies', 'train/rmse_forces', etc."
            " If a single loss term is present, it is the valid/rmse_'loss-term'."
            " Otherwise, if 'forces' is a loss term, it is valid/rmse_forces."
            " Otherwise it must be explicitly specified.",
            rich_help_panel="Loss",
            show_default=False,
        ),
    ] = "valid/rmse_default",
    # Finetuning specific config
    ftune_from: Annotated[
        str,
        Option(
            "--ftune-from",
            help="Name|idx of pretrain run. ani1x:idx, ... also supported",
            rich_help_panel="Finetuning",
            show_default=False,
        ),
    ] = "",
    num_head_layers: Annotated[
        Optional[int],
        Option(
            "--num-head",
            help="If fine-tuning, num. of head layers. Defaults to 1",
            rich_help_panel="Finetuning",
            show_default=False,
        ),
    ] = None,
    backbone_lr: Annotated[
        Optional[float],
        Option(
            "--backbone-lr",
            help="If fine-tuning, lr for backbone. Defaults to 0",
            rich_help_panel="Finetuning",
            show_default=False,
        ),
    ] = None,
    # Debug and profiling specific config
    debug: Annotated[
        bool,
        Option(
            "-g/ ", "--debug/ ", help="Run in debug config", rich_help_panel="Debug"
        ),
    ] = False,
    profiler: Annotated[
        Optional[str],
        Option(
            "--prof",
            help="Profiler, 'simple', 'advanced', or 'pytorch'",
            rich_help_panel="Debug",
            show_default=False,
        ),
    ] = None,
    limit: Annotated[
        Optional[int],
        Option(
            "--lim",
            help="Limit num batches or percent",
            rich_help_panel="Debug",
            show_default=False,
        ),
    ] = None,
    deterministic: Annotated[
        bool,
        Option(
            "--deterministic/ ",
            help="Deterministic training",
            rich_help_panel="Debug",
        ),
    ] = False,
    detect_anomaly: Annotated[
        bool,
        Option(
            "--detect-anomaly/ ",
            help="Detect anomalies during training",
            rich_help_panel="Debug",
        ),
    ] = False,
    verbose: Annotated[
        bool, Option("-v/ ", "--verbose/ ", rich_help_panel="Debug")
    ] = False,
) -> None:
    batched_dataset_path = select_subdirs((batch_name_or_idx,), kind=DataKind.BATCH)[0]
    ds_config_path = batched_dataset_path / "ds_config.json"
    ds_config = DatasetConfig.from_json_file(ds_config_path)
    ds_config.fold_idx = "train" if fold_idx is None else fold_idx
    if fold_idx is not None:
        if not name:
            name = "train" if not ftune_from else "ftune"
        name = f"{str(fold_idx).zfill(2)}-{name}"

    if debug:
        console.print("Debugging enabled:")
        if name == "train":
            _uuid = uuid.uuid4().hex[:8]
            console.print(f"    - Name set to 'debug-{_uuid}'")
            name = f"debug-{_uuid}"
        if max_epochs == 1000:
            max_epochs = 3
            console.print(f"    - Max epochs set to {max_epochs}")
        if limit is None:
            limit = 3
            console.print(f"    - Batch limit set to {limit}")
        console.print("    - Deterministic mode set")
        deterministic = True
        console.print("    - Anomaly detection mode set")
        detect_anomaly = True

    terms_and_factors: tp.Dict[str, float] = {}
    if energies > 0.0:
        label = "EnergiesXC" if xc else "Energies"
        terms_and_factors[label if no_sqrt_atoms else f"{label}SqrtAtoms"] = energies
    if forces > 0.0:
        terms_and_factors["Forces"] = forces
    if dipoles > 0.0:
        terms_and_factors["Dipoles"] = dipoles
    if atomic_charges > 0.0:
        terms_and_factors["AtomicCharges"] = atomic_charges
    if total_charge > 0.0:
        terms_and_factors["TotalCharge"] = total_charge

    lrsched = parse_scheduler_str(lrsched)
    optim = parse_optimizer_str(optim)
    lrsched_opts = lrsched_opts or []
    optim_opts = (optim_opts or []) + [f"lr={lr}", f"weight_decay={wd}"]

    if lr <= 0.0:
        raise ValueError("lr must be strictly positive")

    # Finetune config
    if ftune_from:
        if arch_fn != "simple_ani" or arch_options:
            raise ValueError("Don't specify arch and arch options when finetuning")
        backbone_lr = backbone_lr or 0.0
        num_head_layers = num_head_layers or 1
        # Validation
        if backbone_lr < 0.0:
            raise ValueError("backbone lr must be positive or zero")
        if backbone_lr > lr:
            raise ValueError("Backbone lr must be greater or equal to head lr")
        if num_head_layers < 1:
            raise ValueError("There must be at least one head layer")
        if lr is not None:
            raise ValueError(
                "Instead of '--lr', specify '--head-lr' and --backbone-lr for finetuning"
            )
        if ftune_from.split(":")[0] in ("ani1x", "ani2x", "ani1ccx", "anidr", "aniala"):
            ptrain_name = ftune_from
            ptrain_config = _fetch_builtin_config(ftune_from)
            raw_ptrain_state_dict_path = ""
        else:
            _path = select_subdirs((ftune_from,), kind=DataKind.TRAIN)[0]

            ptrain_name = _path.name
            ptrain_config = TrainConfig.from_json_file(_path / "config.json")
            raw_ptrain_state_dict_path = str(Path(_path, "best-model", "best.ckpt"))

            if not Path(raw_ptrain_state_dict_path).is_file():
                raise ValueError(f"{raw_ptrain_state_dict_path} is not a valid ckpt")
        ftune_config = FinetuneConfig(
            pretrained_name=ptrain_name,
            raw_state_dict_path=raw_ptrain_state_dict_path,
            num_head_layers=num_head_layers,
            backbone_lr=backbone_lr,
        )
        model_config = ptrain_config.model
    else:
        ftune_config = None
        model_config = ModelConfig(
            arch_fn=arch_fn, options=resolve_options(arch_options or (), arch_fn)
        )

    config = TrainConfig(
        name=name,
        debug=debug,
        ds=ds_config,
        monitor_label=monitor,
        ftune=ftune_config,
        model=model_config,
        loss=LossConfig(terms_and_factors=terms_and_factors),
        optim=OptimizerConfig(resolve_options(optim_opts, optim), optim),
        scheduler=SchedulerConfig(resolve_options(lrsched_opts, lrsched), lrsched),
        accel=AccelConfig(
            max_batches_per_packet=100,
            limit=limit,
            deterministic=deterministic,
            detect_anomaly=detect_anomaly,
            max_epochs=max_epochs,
            early_stop_patience=early_stop_patience,
            profiler=profiler,
        ),
    )
    train_lit_model(config, allow_restart=auto_restart, verbose=verbose)


@app.command(help="Visualize train|ftune process with tensorboard")
def tb(
    ftune_name_or_idx: Annotated[
        Optional[tp.List[str]],
        Option("-f", "--ftune-run", help="Name|idx of ftune run"),
    ] = None,
    ptrain_name_or_idx: Annotated[
        Optional[tp.List[str]],
        Option("-t", "--train-run", help="Name|idx of train run"),
    ] = None,
) -> None:
    paths = []
    for selectors, dkind in zip(
        (
            ftune_name_or_idx,
            ptrain_name_or_idx,
        ),
        (
            DataKind.FTUNE,
            DataKind.TRAIN,
        ),
    ):
        if selectors is not None:
            paths.extend(select_subdirs(selectors, kind=dkind))
    with tempfile.TemporaryDirectory() as d:
        for path in paths:
            run_subdir = Path(d, path.name)
            run_subdir.mkdir()
            tb_path = path / "tb-logs"
            for version_dir in sorted(tb_path.glob("version_*")):
                for events_file in sorted(version_dir.iterdir()):
                    if not events_file.name.startswith("events.out.tfevents"):
                        continue
                    tmp_events_symlink = run_subdir / events_file.name
                    tmp_events_symlink.symlink_to(events_file)
        subprocess.run(["tensorboard", "--logdir", d])
