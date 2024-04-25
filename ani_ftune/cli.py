r"""Command line interface entrypoints"""

import pickle
import shutil
import typing as tp
import typing_extensions as tpx
from pathlib import Path

from typer import Option, Typer

from ani_ftune.console import console
from ani_ftune.utils import load_state_dict
from ani_ftune.lit_training import train_from_scratch
from ani_ftune.config import (
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
)

app = Typer(
    rich_markup_mode="markdown",
    help=r"""## ANI-ftune

    Utility for generating a fine-tuned models from pre trained ANI style models,
    given a set of reference structures.
    """,
)


@app.command(help="Clean debug runs")
def clean(
    ftune: tpx.Annotated[
        bool,
        Option(
            "-f/-F",
            "--ftune/--no-ftune",
            help="Clean ftune config",
        ),
    ] = True,
    train: tpx.Annotated[
        bool,
        Option(
            "-p/-P",
            "--pretrain/--no-pretrain",
            help="Clean train config",
        ),
    ] = True,
) -> None:
    if train:
        if any(_DEBUG_TRAIN_PATH.iterdir()):
            shutil.rmtree(_DEBUG_TRAIN_PATH)
            _DEBUG_TRAIN_PATH.mkdir(exist_ok=True, parents=True)
            console.print("Cleaned all training debug runs")
        else:
            console.print("No debug training runs to clean")
    if ftune:
        if any(_DEBUG_FTUNE_PATH.iterdir()):
            shutil.rmtree(_DEBUG_FTUNE_PATH)
            _DEBUG_FTUNE_PATH.mkdir(exist_ok=True, parents=True)
            console.print("Cleaned all finetuning debug runs")
        else:
            console.print("No debug finetuning runs to clean")


def _select_run_path(
    name_or_idx: str,
    ftune: bool = False,
    debug: bool = False,
) -> Path:
    try:
        name = ""
        idx = int(name_or_idx)
    except ValueError:
        name = name_or_idx
        idx = None

    if debug:
        root = _DEBUG_FTUNE_PATH if ftune else _DEBUG_TRAIN_PATH
    else:
        root = _FTUNE_PATH if ftune else _TRAIN_PATH

    if idx is not None:
        try:
            path = sorted(root.iterdir())[idx]
        except IndexError:
            raise ValueError(
                f"Run {'ftune' if ftune else 'train'}-{idx} could not be found"
            ) from None
    else:
        for p in sorted(root.iterdir()):
            if p.name == name:
                path = p
                break
        else:
            raise ValueError(f"Run {name} could not be found") from None
    return path


@app.command(help="Continue a previously started training")
def restart(
    ftune_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-f",
            help="Name or idx of the run",
        ),
    ] = "",
    ptrain_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-p",
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
    if ftune_name_or_idx and ptrain_name_or_idx or not (ftune_name_or_idx or ptrain_name_or_idx):
        raise ValueError("One and only one of -f and -p should be specified")
    name_or_idx = ftune_name_or_idx or ptrain_name_or_idx
    path = _select_run_path(name_or_idx, ftune=bool(ftune_name_or_idx), debug=debug) / "config.pkl"
    if not path.is_file():
        raise ValueError(f"{path} is not a file dir")

    with open(path, mode="rb") as f:
        config = pickle.load(f)
    console.print(f"Restarting run {path.name}")
    train_from_scratch(config, restart=True)


@app.command(help="Display training and finetuning runs")
def ls() -> None:
    train = sorted(_TRAIN_PATH.iterdir())
    ftune = sorted(_FTUNE_PATH.iterdir())
    debug_train = sorted(_DEBUG_TRAIN_PATH.iterdir())
    debug_ftune = sorted(_DEBUG_FTUNE_PATH.iterdir())

    if train or debug_train:
        console.print("Training runs:")
        for j, p in enumerate(train):
            console.print(f"{j}. {p.name}", style="green")
        for j, p in enumerate(debug_train):
            console.print(f"(debug) {j}. {p.name}", style="yellow")
    else:
        console.print("(No training runs found)")

    console.print()
    if ftune or debug_ftune:
        console.print("Finetuning runs:")
        for j, p in enumerate(ftune):
            console.print(f"{j}. {p.name}", style="blue")
        for j, p in enumerate(debug_ftune):
            console.print(f"(debug) {j}. {p.name}", style="yellow")
    else:
        console.print("(No finetuning runs found)")


@app.command(help="Delete specific training or finetuning run")
def rm(
    ftune_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-f",
            help="Name or idx of the run",
        ),
    ] = "",
    ptrain_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-p",
            help="Name or idx of the run",
        ),
    ] = "",
    debug: tpx.Annotated[
        bool,
        Option(
            "-g/-G",
            "--debug/--no-debug",
            help="Remove a single debug run",
        ),
    ] = False,
) -> None:
    if ftune_name_or_idx and ptrain_name_or_idx or not (ftune_name_or_idx or ptrain_name_or_idx):
        raise ValueError("One and only one of -f and -p should be specified")
    name_or_idx = ftune_name_or_idx or ptrain_name_or_idx
    path = _select_run_path(name_or_idx, ftune=bool(ftune_name_or_idx), debug=debug)
    shutil.rmtree(path)
    console.print(f"Removed run {path.name}")


@app.command(help="Compare the params of a ftuned model and the original model")
def compare(
    pretrained_name_or_idx: tpx.Annotated[
        str,
        Option(
            "-p",
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
    pretrained_path = (
        _select_run_path(pretrained_name_or_idx, ftune=False, debug=debug)
        / "best-model"
    )
    ftuned_path = (
        _select_run_path(ftuned_name_or_idx, ftune=True, debug=debug) / "best-model"
    )

    pretrained_state_dict = load_state_dict(pretrained_path / "best.ckpt")
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
    original_model_path: tpx.Annotated[
        Path,
        Option(
            "-o",
            "--original-state-dict",
            help="Path to the pretrained state dict .pt or .ckpt",
        ),
    ],
    ftuned_model_path: tpx.Annotated[
        Path,
        Option(
            "-f",
            "--ftuned-state-dict",
            help="Path to the finetuned state dict .pt or .ckpt file",
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
    name: tpx.Annotated[
        str,
        Option(
            "--name",
            help="Name of the run",
        ),
    ] = "run",
    builder: tpx.Annotated[
        str,
        Option(
            "--builder",
            help="Builder function",
        ),
    ] = "FlexibleANI",
    dataset_name: tpx.Annotated[
        str,
        Option(
            "--dataset",
            help="Builtin dataset name",
        ),
    ] = "ANI1x",
    batch_size: tpx.Annotated[
        int,
        Option(
            "--batch-size",
            help="Batch size",
        ),
    ] = 2560,
    _fold_idx: tpx.Annotated[
        str,
        Option(
            "--fold-idx",
            help="Fold idx",
        ),
    ] = "single",
    train_frac: tpx.Annotated[
        float,
        Option(
            "--train-frac",
            help="Training set fraction",
        ),
    ] = 0.8,
    validation_frac: tpx.Annotated[
        float,
        Option(
            "--validation-frac",
            help="Validation set fraction",
        ),
    ] = 0.2,
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
    forces: tpx.Annotated[
        bool,
        Option(
            "-f/-F",
            "--force/--no-force",
            help="Train with forces",
        ),
    ] = False,
    dipoles: tpx.Annotated[
        bool,
        Option(
            "-d/-D",
            "--dipoles/--no-dipoles",
            help="Train with dipoles",
        ),
    ] = False,
    atomic_charges: tpx.Annotated[
        bool,
        Option(
            "-a/-A",
            "--atomic-charges/--no-atomic-charges",
            help="Train with atomic charges",
        ),
    ] = False,
    total_charge: tpx.Annotated[
        bool,
        Option(
            "-q/-Q",
            "--total-charge/--no-total-charge",
            help="Train with total charge",
        ),
    ] = False,
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
    use_cuda_ops: tpx.Annotated[bool, Option("--cuda-ops/--no-cuda-ops", help="Use cuda acceleration",),] = True,
    data_seed: tpx.Annotated[
        int,
        Option(
            "--data-seed",
            help="Seed for dataset prebatching",
        ),
    ] = 1234,
) -> None:
    fold_idx: tp.Union[str, int]
    try:
        fold_idx = int(_fold_idx)
    except ValueError:
        fold_idx = _fold_idx
    if debug:
        if limit is None:
            console.print("Setting train limit to 10 batches for debugging")
            limit = 10
        console.print("Setting deterministic training for debugging purposes")
        deterministic = True
        console.print("Setting anomaly detection for debugging purposes")
        detect_anomaly = True
    config = TrainConfig(
        name=name,
        debug=debug,
        ds=DatasetConfig(
            name=dataset_name,
            batch_size=batch_size,
            fold_idx=fold_idx,
            validation_frac=validation_frac,
            train_frac=train_frac,
            shuffle_seed=data_seed,
        ),
        accel=AccelConfig(
            max_batches_per_packet=100,
            limit=limit,
            deterministic=deterministic,
            detect_anomaly=detect_anomaly,
            use_cuda_ops=use_cuda_ops,
        ),
        model=ModelConfig(builder=builder),
        loss=LossConfig(
            terms_and_factors=(("Energies", 1.0),),
        ),
        optim=OptimizerConfig(lr=lr, weight_decay=weight_decay),
        scheduler=SchedulerConfig(),
    )
    train_from_scratch(config)


@app.command(help="Fine tune a pretrained ANI model")
def ftune(
    name_or_idx: tpx.Annotated[
        str,
        Option(
            "-p",
            help="Name or idx of the pretrained run, alternatively, ani1x:idx, ani2x:idx, etc. is also supported",
        ),
    ],
    _src_paths: tpx.Annotated[
        tp.Optional[tp.List[Path]],
        Option(
            "-s",
            "--data-path",
            help="Paths to data to fine-tune the model with",
        ),
    ] = None,
    dataset_name: tpx.Annotated[
        str,
        Option(
            "--dataset",
            help="Builtin dataset name",
        ),
    ] = "",
    name: tpx.Annotated[
        str,
        Option(
            "--name",
            help="Name of the run",
        ),
    ] = "ftune",
    num_head_layers: tpx.Annotated[
        int,
        Option(
            "-n",
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
    batch_size: tpx.Annotated[
        int,
        Option(
            "-b",
            "--batch",
            help="Batch size",
        ),
    ] = 64,
    train_frac: tpx.Annotated[
        float,
        Option(
            "--train-frac",
            help="Training set fraction",
        ),
    ] = 0.8,
    validation_frac: tpx.Annotated[
        float,
        Option(
            "--validation-frac",
            help="Validation set fraction",
        ),
    ] = 0.2,
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
    data_seed: tpx.Annotated[
        int,
        Option(
            "--data-seed",
            help="Seed for dataset prebatching",
        ),
    ] = 1234,
    debug: tpx.Annotated[
        bool,
        Option(
            "-g/-G",
            "--debug/--no-debug",
            help="Debug finetune run",
        ),
    ] = False,
    use_cuda_ops: tpx.Annotated[bool, Option("--cuda-ops/--no-cuda-ops", help="Use cuda acceleration",),] = True,
) -> None:
    src_paths = () if _src_paths is None else tuple(sorted(_src_paths))
    if (not (src_paths or dataset_name)) or (src_paths and dataset_name):
        raise ValueError(
            "One of src_paths or dataset_name must be specified, but not both"
        )
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
        from ani_ftune.model_builders import fetch_pretrained_config
        pretrained_config = fetch_pretrained_config(name_or_idx)
        pretrained_state_dict_path = None
    else:
        pretrained_path = _select_run_path(name_or_idx, ftune=False, debug=debug)
        pretrained_config_path = pretrained_path / "config.pkl"
        if not pretrained_config_path.is_file():
            raise ValueError(f"{pretrained_config_path} is not a valid config file")

        with open(pretrained_config_path, mode="rb") as f:
            pretrained_config = pickle.load(f).model

        pretrained_state_dict_path = (pretrained_path / "best-model") / "best.ckpt"

        if not pretrained_state_dict_path.is_file():
            raise ValueError(f"{pretrained_state_dict_path} is not a valid checkpoint")

    config = TrainConfig(
        name=f"{name}-from_{pretrained_config.ds.fold_idx}",
        ds=DatasetConfig(
            name=dataset_name,
            src_paths=src_paths,
            batch_size=batch_size,
            fold_idx="single",
            validation_frac=validation_frac,
            train_frac=train_frac,
            shuffle_seed=data_seed,
        ),
        accel=AccelConfig(
            max_batches_per_packet=100,
            limit=limit,
            deterministic=deterministic,
            detect_anomaly=detect_anomaly,
            use_cuda_ops=use_cuda_ops,
        ),
        model=pretrained_config.model,
        loss=LossConfig(
            terms_and_factors=(("Energies", 1.0),),
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
