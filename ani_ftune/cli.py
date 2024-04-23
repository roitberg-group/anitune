r"""Command line interface entrypoints"""

import pickle
import shutil
import typing as tp
import typing_extensions as tpx
from pathlib import Path

from typer import Option, Typer

from ani_ftune.utils import TrainKind
from ani_ftune.console import console
from ani_ftune.lit_training import train_from_scratch
from ani_ftune.configuration import (
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
            "--ftune/--no-ftune",
            help="Clean ftune config",
        ),
    ] = True,
    train: tpx.Annotated[
        bool,
        Option(
            "--train/--no-train",
            help="Clean train config",
        ),
    ] = True,
) -> None:
    if train:
        shutil.rmtree(_DEBUG_TRAIN_PATH)
        _DEBUG_TRAIN_PATH.mkdir(exist_ok=True, parents=True)
    if ftune:
        shutil.rmtree(_DEBUG_FTUNE_PATH)
        _DEBUG_FTUNE_PATH.mkdir(exist_ok=True, parents=True)


@app.command(help="Continue a previously started training")
def restart(
    name: tpx.Annotated[str, Option("-n", "--name", help="Name of training run",),] = "",
    idx: tpx.Annotated[tp.Optional[int], Option("-i" "--idx", help="Help string",),] = None,
    kind: tpx.Annotated[TrainKind, Option("-k", "--kind", help="Kind of training run",),] = TrainKind.TRAIN,
) -> None:
    if (idx is None and not name) or (idx is not None and name):
        raise ValueError("Either an index or a name should be specified, but not both")

    if kind is TrainKind.TRAIN:
        root = _TRAIN_PATH
    else:
        root = _FTUNE_PATH

    if idx is not None:
        try:
            path = sorted(root.iterdir())[idx] / "config.pkl"
        except IndexError:
            raise ValueError(f"Run {kind}-{idx} could not be found") from None
    else:
        for p in sorted(root.iterdir()):
            if p.name == name:
                path = p / "config.pkl"
                break
        else:
            raise ValueError(f"Run {name} could not be found") from None

    if not path.is_file():
        raise ValueError(f"{path} is not a valid file")

    with open(path / "config.pkl", mode="rb", encoding="utf-8") as f:
        config = pickle.load(f)
    train_from_scratch(config)


@app.command()
def ls() -> None:
    train = sorted(_TRAIN_PATH.iterdir())
    ftune = sorted(_FTUNE_PATH.iterdir())
    debug_train = sorted(_DEBUG_TRAIN_PATH.iterdir())
    debug_ftune = sorted(_DEBUG_FTUNE_PATH.iterdir())

    if train or debug_train:
        console.print("Training runs:")
        for j, p in enumerate(train):
            console.print(f"{j}. {p.name}", style="green")
        for p in debug_train:
            console.print(f"(debug). {p.name}", style="yellow")
    else:
        console.print("(No training runs found)")

    console.print()
    if ftune or debug_ftune:
        console.print("Finetuning runs:")
        for j, p in enumerate(ftune):
            console.print(f"{j}. {p.name}", style="blue")
        for p in debug_ftune:
            console.print(f"(debug). {p.name}", style="yellow")
    else:
        console.print("(No finetuning runs found)")


@app.command(help="Compare the params of a ftuned model and the original model")
def delta(
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
) -> None:
    raise NotImplementedError("Not implemented yet")


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
            "-g/-G" "--debug/--no-debug",
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
) -> None:
    fold_idx: tp.Union[str, int]
    try:
        fold_idx = int(_fold_idx)
    except ValueError:
        fold_idx = _fold_idx
    if debug:
        if limit is None:
            print("Setting train limit to 10 batches for debugging")
            limit = 10
    config = TrainConfig(
        name=name,
        debug=debug,
        ds=DatasetConfig(
            name=dataset_name,
            batch_size=batch_size,
            fold_idx=fold_idx,
            validation_frac=validation_frac,
            train_frac=train_frac,
        ),
        accel=AccelConfig(max_batches_per_packet=100, limit=limit),
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
    state_dict_path: tpx.Annotated[
        Path,
        Option(
            "-s",
            "--state-dict",
            help="Path to the pretrained jit-compiled model",
        ),
    ],
    src_paths: tpx.Annotated[
        tp.List[Path],
        Option(
            "-d",
            "--dataset",
            help="Paths to data to fine-tune the model with",
        ),
    ],
    name: tpx.Annotated[
        str,
        Option(
            "--name",
            help="Name of the run",
        ),
    ] = "ftune",
    builder: tpx.Annotated[
        str,
        Option(
            "--builder",
            help="Builder function",
        ),
    ] = "FlexibleANI",
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
    ] = 0.5e-3,
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
    limit: tpx.Annotated[
        tp.Optional[int],
        Option(
            "--limit",
            help="Limit number of batches or percent",
        ),
    ] = None,
    original_idx: tpx.Annotated[
        int,
        Option(
            "--idx",
            help="Original index of the model",
        ),
    ] = 0,
) -> None:
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

    # TODO: Do smth with the state dict
    config = TrainConfig(
        name=f"{name}-{original_idx}",
        ds=DatasetConfig(
            src_paths=tuple(sorted(src_paths)),
            batch_size=batch_size,
            fold_idx="single",
            validation_frac=validation_frac,
            train_frac=train_frac,
        ),
        accel=AccelConfig(max_batches_per_packet=100, limit=limit),
        model=ModelConfig(builder=builder),
        loss=LossConfig(
            terms_and_factors=(("Energies", 1.0),),
        ),
        optim=OptimizerConfig(
            lr=head_lr,
            weight_decay=weight_decay,
        ),
        scheduler=SchedulerConfig(),
        ftune=FinetuneConfig(
            state_dict_path=state_dict_path,
            num_head_layers=num_head_layers,
            backbone_lr=backbone_lr,
        ),
    )
    train_from_scratch(config)
