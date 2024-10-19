r"""Command line interface entrypoints"""

import tempfile
import uuid
import subprocess
from copy import deepcopy
import hashlib
import shutil
import typing as tp
import typing_extensions as tpx
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


# TODO the ds part of this is broken
def fetch_builtin_config(name_or_idx: str) -> TrainConfig:
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
    name: tpx.Annotated[
        str,
        Option(
            "--name",
            help="Name for ensemble",
        ),
    ] = "ensemble",
    ftune_names_or_idxs: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "-f",
            "--ftune-run",
            help="Name or idx of the run",
        ),
    ] = None,
    ptrain_names_or_idxs: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "-t",
            "--train-run",
            help="Name or idx of the run",
        ),
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


@app.command(help="Prebatch a dataset")
def batch(
    name: tpx.Annotated[
        str,
        Option(
            "-n",
            "--name",
            help="Name for the batched dataset",
        ),
    ] = "",
    lot: tpx.Annotated[
        str,
        Option(
            "-l",
            "--lot",
            help="Level of theory",
        ),
    ] = "wb97x-631gd",
    data_names: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "-d",
            "--dataset-name",
            help="Builtin dataset name",
        ),
    ] = None,
    src_paths: tpx.Annotated[
        tp.Optional[tp.List[Path]],
        Option(
            "-s",
            "--src-paths",
            help="Paths to custom datasets",
        ),
    ] = None,
    properties: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "-p",
            "--property",
            help="Properties to batch, all by default",
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
    batch_seed: tpx.Annotated[
        int,
        Option(
            help="Seed for dataset batching",
        ),
    ] = 1234,
    divs_seed: tpx.Annotated[
        int,
        Option(
            help="Seed for dataset batching",
        ),
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
        validation_frac=validation_frac,
        train_frac=train_frac,
        batch_seed=batch_seed,
        divs_seed=divs_seed,
    )
    batch_data(ds, max_batches_per_packet=100)


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
    max_epochs: tpx.Annotated[
        tp.Optional[int],
        Option(
            "--max-epochs",
            help="Maximum number of epochs to train",
        ),
    ] = None,
    verbose: tpx.Annotated[
        bool,
        Option(
            "-v/ ",
            "--verbose/--no-verbose",
            help="Help string",
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
    ensemble_name_or_idx: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "-e",
            help="Name or idx of the ensemble",
        ),
    ] = None,
) -> None:
    for selectors, dkind in zip(
        (
            ftune_name_or_idx,
            ptrain_name_or_idx,
            batch_name_or_idx,
            ensemble_name_or_idx,
        ),
        (
            DataKind.FTUNE,
            DataKind.TRAIN,
            DataKind.BATCH,
            DataKind.ENSEMBLE,
        ),
    ):
        if selectors is not None:
            paths = select_subdirs(selectors, kind=dkind)
            for p in paths:
                shutil.rmtree(p)
                console.print(f"Removed {p.name}")
            console.print()


@app.command(help="Compare the params of a trained model and the original model")
def compare(
    ptrain_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-t",
            help="Name or idx of the pretrained run",
        ),
    ] = "",
    ftune_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-f",
            help="Name or idx of the finetuned run",
        ),
    ] = "",
) -> None:
    if (not (ftune_name_or_idx or ptrain_name_or_idx)) or (
        ftune_name_or_idx and ptrain_name_or_idx
    ):
        raise ValueError("One and only one of -t or -f has to be specified")
    kind = DataKind.FTUNE if ftune_name_or_idx else DataKind.TRAIN
    root = select_subdirs(
        (ptrain_name_or_idx or ftune_name_or_idx,),
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
        Argument(help="Name or idx of the batched dataset"),
    ],
    name: tpx.Annotated[
        str,
        Option(
            "-n",
            "--name",
            help="Name of the run",
        ),
    ] = "train",
    monitor: tpx.Annotated[
        str,
        Option(
            "--monitor",
            help="Label to monitor during training."
            " Format is 'rmse_energies', or 'rmse_forces', etc."
            " If only one loss term is present, it is the validation RMSE of the corresponding loss label."
            " Otherwise it must be explicitly specified",
        ),
    ] = "valid_rmse_default",
    profiler: tpx.Annotated[
        tp.Optional[str],
        Option(
            "--profiler",
            help="Profiler for finding bottlenecks in training (one of 'simple', 'advanced', 'pytorch')",
        ),
    ] = None,
    arch_fn: tpx.Annotated[
        str,
        Option(
            "-a",
            "--arch",
            help="Network architecture",
        ),
    ] = "simple_ani",
    arch_options: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "--ao",
            "--arch-option",
            help="Options passed to the arch function in the form key=value. Different arch functions accept different options",
        ),
    ] = None,
    optimizer: tpx.Annotated[
        str,
        Option(
            "-o",
            "--optimizer",
            help="Type of optimizer",
        ),
    ] = "AdamW",
    optimizer_options: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "--oo",
            "--optimizer-option",
            help="Options passed to the optimizer in the form key=value. Different optimizers accept different options (lr and wd are passed separately)",
        ),
    ] = None,
    scheduler: tpx.Annotated[
        str,
        Option(
            "-s",
            "--scheduler",
            help="Type of lr-scheduler",
        ),
    ] = "Plateau",
    scheduler_options: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "--so",
            "--scheduler-option",
            help="Options passed to the lr-scheduler in the form key=value. Different schedulers accept different options",
        ),
    ] = None,
    fold_idx: tpx.Annotated[
        tp.Optional[int],
        Option(
            "-i",
            "--fold-idx",
            help="Fold idx",
        ),
    ] = None,
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
            "--energies",
            help="Train with energies",
        ),
    ] = 1.0,
    forces: tpx.Annotated[
        float,
        Option(
            "-f",
            "--forces",
            help="Train with forces",
        ),
    ] = 0.0,
    dipoles: tpx.Annotated[
        float,
        Option(
            "-m",
            "--dipoles",
            help="Train with dipoles",
        ),
    ] = 0.0,
    atomic_charges_mbis: tpx.Annotated[
        float,
        Option(
            "--mbis",
            "--atomic-charges-mbis",
            help="Train with atomic charges",
        ),
    ] = 0.0,
    atomic_charges: tpx.Annotated[
        float,
        Option(
            "-q",
            "--atomic-charges",
            help="Train with atomic charges",
        ),
    ] = 0.0,
    total_charge: tpx.Annotated[
        float,
        Option(
            "--total-charge",
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
    max_epochs: tpx.Annotated[
        int,
        Option(
            "--max-epochs",
            help="Maximum number of epochs to train",
        ),
    ] = 200,
    early_stop_patience: tpx.Annotated[
        int,
        Option(
            "--early-stop-patience",
            help="Maximum number of epochs with no improvement in validation metric before early stopping",
        ),
    ] = 50,
    verbose: tpx.Annotated[
        bool,
        Option(
            "-v/ ",
            "--verbose/--no-verbose",
            help="Help string",
        ),
    ] = False,
) -> None:
    batched_dataset_path = select_subdirs((batch_name_or_idx,), kind=DataKind.BATCH)[0]
    ds_config_path = batched_dataset_path / "ds_config.json"
    ds_config = DatasetConfig.from_json_file(ds_config_path)
    ds_config.fold_idx = "train" if fold_idx is None else fold_idx
    if fold_idx is not None:
        name = f"{str(fold_idx).zfill(2)}-{name}"

    if debug:
        console.print("Debugging enabled:")
        if name == "train":
            _uuid = uuid.uuid4().hex[:8]
            console.print(f"    - Name set to 'debug-{_uuid}'")
            name = f"debug-{_uuid}"
        if max_epochs == 200:
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
        terms_and_factors[label if not sqrt_atoms else f"{label}SqrtAtoms"] = energies
    if forces > 0.0:
        terms_and_factors["Forces"] = forces
    if dipoles > 0.0:
        terms_and_factors["Dipoles"] = dipoles
    if atomic_charges > 0.0:
        terms_and_factors["AtomicCharges"] = atomic_charges
    if atomic_charges_mbis > 0.0:
        terms_and_factors["AtomicChargesMBIS"] = atomic_charges_mbis
    if total_charge > 0.0:
        terms_and_factors["TotalCharge"] = total_charge

    scheduler = parse_scheduler_str(scheduler)
    optimizer = parse_optimizer_str(optimizer)
    _optimizer_options = resolve_options(optimizer_options or (), optimizer)
    _optimizer_options.update({"lr": lr, "weight_decay": weight_decay})

    config = TrainConfig(
        name=name,
        debug=debug,
        ds=ds_config,
        monitor_label=monitor,
        accel=AccelConfig(
            max_batches_per_packet=100,
            limit=limit,
            deterministic=deterministic,
            detect_anomaly=detect_anomaly,
            max_epochs=max_epochs,
            early_stop_patience=early_stop_patience,
            profiler=profiler,
        ),
        model=ModelConfig(
            arch_fn=arch_fn,
            options=resolve_options(arch_options or (), arch_fn),
        ),
        loss=LossConfig(
            terms_and_factors=terms_and_factors,
        ),
        optim=OptimizerConfig(
            cls=optimizer,
            options=_optimizer_options,
        ),
        scheduler=SchedulerConfig(
            cls=scheduler,
            options=resolve_options(scheduler_options or (), scheduler),
        ),
    )
    train_lit_model(config, verbose=verbose)


@app.command(help="Fine tune a pretrained ANI model")
def ftune(
    batch_name_or_idx: tpx.Annotated[
        str,
        Argument(help="Name or idx of the batched dataset"),
    ],
    name_or_idx: tpx.Annotated[
        str,
        Option(
            "-t",
            "--train-run",
            help="Name or idx of the pretrained run, alternatively, ani1x:idx, ani2x:idx, etc. is also supported",
        ),
    ],
    profiler: tpx.Annotated[
        tp.Optional[str],
        Option(
            "--profiler",
            help="Profiler for finding bottlenecks in training (one of 'simple', 'advanced', 'pytorch')",
        ),
    ] = None,
    name: tpx.Annotated[
        str,
        Option(
            "-n",
            "--name",
            help="Name of the run",
        ),
    ] = "ftune",
    monitor: tpx.Annotated[
        str,
        Option(
            "--monitor",
            help="Label to monitor during training."
            " Format is 'valid_rmse_energies', or 'train_rmse_forces', etc."
            " If only one loss term is present, it is the validation RMSE of the corresponding loss label."
            " Otherwise it must be explicitly specified",
        ),
    ] = "valid_rmse_default",
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
            "--energies",
            help="Train with energies",
        ),
    ] = 1.0,
    forces: tpx.Annotated[
        float,
        Option(
            "-f",
            "--forces",
            help="Train with forces",
        ),
    ] = 0.0,
    dipoles: tpx.Annotated[
        float,
        Option(
            "-m",
            "--dipoles",
            help="Train with dipoles",
        ),
    ] = 0.0,
    atomic_charges: tpx.Annotated[
        float,
        Option(
            "-q",
            "--atomic-charges",
            help="Train with atomic charges",
        ),
    ] = 0.0,
    atomic_charges_mbis: tpx.Annotated[
        float,
        Option(
            "--mbis",
            "--atomic-charges-mbis",
            help="Train with atomic charges",
        ),
    ] = 0.0,
    total_charge: tpx.Annotated[
        float,
        Option(
            "--total-charge",
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
    max_epochs: tpx.Annotated[
        int,
        Option(
            "--max-epochs",
            help="Maximum number of epochs to train",
        ),
    ] = 200,
    early_stop_patience: tpx.Annotated[
        int,
        Option(
            "--early-stop-patience",
            help="Maximum number of epochs with no improvement in validation metric before early stopping",
        ),
    ] = 50,
    verbose: tpx.Annotated[
        bool,
        Option(
            "-v/ ",
            "--verbose/--no-verbose",
            help="Help string",
        ),
    ] = False,
    fold_idx: tpx.Annotated[
        tp.Optional[int],
        Option(
            "-i",
            "--fold-idx",
            help="Fold idx",
        ),
    ] = None,
    optimizer: tpx.Annotated[
        str,
        Option(
            "-o",
            "--optimizer",
            help="Type of optimizer",
        ),
    ] = "AdamW",
    optimizer_options: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "--oo",
            "--optimizer-option",
            help="Options passed to the optimizer in the form key=value. Different optimizers accept different options (lr and wd are passed separately)",
        ),
    ] = None,
    scheduler: tpx.Annotated[
        str,
        Option(
            "-s",
            "--scheduler",
            help="Type of lr-scheduler",
        ),
    ] = "Plateau",
    scheduler_options: tpx.Annotated[
        tp.Optional[tp.List[str]],
        Option(
            "--so",
            "--scheduler-option",
            help="Options passed to the lr-scheduler in the form key=value. Different schedulers accept different options",
        ),
    ] = None,
) -> None:
    batched_dataset_path = select_subdirs((batch_name_or_idx,), kind=DataKind.BATCH)[0]
    ds_config_path = batched_dataset_path / "ds_config.json"
    ds_config = DatasetConfig.from_json_file(ds_config_path)
    ds_config.fold_idx = "train" if fold_idx is None else fold_idx
    if fold_idx is not None:
        name = f"{str(fold_idx).zfill(2)}-{name}"

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

    if name_or_idx.split(":")[0] in ("ani1x", "ani2x", "ani1ccx", "anidr", "aniala"):
        pretrained_config = fetch_builtin_config(name_or_idx)
        raw_pretrained_state_dict_path = ""
        pretrained_name = name_or_idx
    else:
        pretrained_path = select_subdirs(
            (name_or_idx,),
            kind=DataKind.TRAIN,
        )[0]
        pretrained_name = pretrained_path.name
        pretrained_config = TrainConfig.from_json_file(pretrained_path / "config.json")

        raw_pretrained_state_dict_path = str(
            Path(pretrained_path, "best-model", "best.ckpt")
        )

        if not Path(raw_pretrained_state_dict_path).is_file():
            raise ValueError(
                f"{raw_pretrained_state_dict_path} is not a valid checkpoint"
            )

    terms_and_factors: tp.Dict[str, float] = {}
    if energies > 0.0:
        label = "EnergiesXC" if xc else "Energies"
        terms_and_factors[label if not sqrt_atoms else f"{label}SqrtAtoms"] = energies
    if forces > 0.0:
        terms_and_factors["Forces"] = forces
    if dipoles > 0.0:
        terms_and_factors["Dipoles"] = dipoles
    if atomic_charges > 0.0:
        terms_and_factors["AtomicCharges"] = atomic_charges
    if atomic_charges_mbis > 0.0:
        terms_and_factors["AtomicChargesMBIS"] = atomic_charges_mbis
    if total_charge > 0.0:
        terms_and_factors["TotalCharge"] = total_charge

    scheduler = parse_scheduler_str(scheduler)
    optimizer = parse_optimizer_str(optimizer)
    _optimizer_options = resolve_options(optimizer_options or (), optimizer)
    _optimizer_options.update({"lr": head_lr, "weight_decay": weight_decay})

    config = TrainConfig(
        name=name,
        ds=ds_config,
        monitor_label=monitor,
        accel=AccelConfig(
            max_batches_per_packet=100,
            limit=limit,
            deterministic=deterministic,
            detect_anomaly=detect_anomaly,
            max_epochs=max_epochs,
            early_stop_patience=early_stop_patience,
            profiler=profiler,
        ),
        model=pretrained_config.model,
        loss=LossConfig(
            terms_and_factors=terms_and_factors,
        ),
        optim=OptimizerConfig(
            cls=optimizer,
            options=_optimizer_options,
        ),
        scheduler=SchedulerConfig(
            cls=scheduler,
            options=resolve_options(scheduler_options or (), scheduler),
        ),
        ftune=FinetuneConfig(
            pretrained_name=pretrained_name,
            raw_state_dict_path=raw_pretrained_state_dict_path,
            num_head_layers=num_head_layers,
            backbone_lr=backbone_lr,
        ),
    )
    train_lit_model(config, verbose=verbose)


@app.command(help="Visualize train|ftune process with tensorboard")
def tb(
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
