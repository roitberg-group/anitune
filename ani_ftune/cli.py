r"""Command line interface entrypoints"""

import typing as tp
import typing_extensions as tpx
from pathlib import Path

from typer import Option, Typer

from ani_ftune.lit_training import train_from_scratch
from ani_ftune.ftune import FinetuneSpec
from ani_ftune.configuration import (
    TrainConfig,
    DatasetConfig,
    AccelConfig,
    ModelConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
)

app = Typer(
    rich_markup_mode="markdown",
    help=r"""## ANI-ftune

    Utility for generating a fine-tuned models from pre trained ANI style models,
    given a set of reference structures.
    """,
)


@app.command(help="Compare the params of a ftuned model and the original model")
def delta(
    original_model_path: tpx.Annotated[
        Path,
        Option(
            "-om",
            "--om",
            "--original-model",
            help="Path to the pretrained jit-compiled model",
        ),
    ],
    ftuned_model_path: tpx.Annotated[
        Path,
        Option(
            "-fm",
            "--fm",
            "--ftuned-model",
            help="Path to the finetuned jit-compiled model",
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
            "-om" "--om",
            "--original-model",
            help="Path to the pretrained jit-compiled model",
        ),
    ],
    ftuned_model_path: tpx.Annotated[
        Path,
        Option(
            "-fm" "--fm",
            "--ftuned-model",
            help="Path to the finetuned jit-compiled model",
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
    name: tpx.Annotated[str, Option("--name", help="Name of the run",),] = "run",
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
            "--debug/--no-debug",
            help="Run debug",
        ),
    ] = False,
    limit: tpx.Annotated[tp.Optional[int], Option("--limit", help="Limit number of batches or percent",),] = None,
) -> None:
    fold_idx: tp.Union[str, int]
    try:
        fold_idx = int(_fold_idx)
    except ValueError:
        fold_idx = _fold_idx
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
    model_path: tpx.Annotated[
        Path,
        Option(
            "-m",
            "--model-path",
            help="Path to the pretrained jit-compiled model",
        ),
    ],
    data_paths: tpx.Annotated[
        tp.List[Path],
        Option(
            "-d",
            "--data-path",
            help="Paths to data to fine-tune the model with",
        ),
    ],
    _test_paths: tpx.Annotated[
        tp.Optional[tp.List[Path]],
        Option(
            "-t",
            "--test-path",
            help="Paths to test data",
        ),
    ] = None,
    num_head_layers: tpx.Annotated[
        int,
        Option(
            "-n",
            "--num-head-layers",
            help="Number of layers to use as model head",
        ),
    ] = 1,
    lrs: tpx.Annotated[
        tp.Tuple[float, float],
        Option(
            "-l",
            "--lrs",
            help="Tuple of learning rates for (head, body). Body lr may be 0",
        ),
    ] = (1e-3, 0.0),
    unfreeze_body_epoch_num: tpx.Annotated[
        int,
        Option(
            "-f",
            "--freeze-body-epoch",
            help="Epoch at which body is frozen, only used if body lr is nonzero",
        ),
    ] = 0,
    # Loss and optimizer specification
    l2_factor: tpx.Annotated[
        float,
        Option(
            "--l2-factor",
            help="L2 penalty for the loss",
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
    force_train_factor: tpx.Annotated[
        float,
        Option(
            "--force-train-factor",
            help="Factor for force training loss, may be 0",
        ),
    ] = 0.0,
) -> None:
    test_paths = [] if _test_paths is None else _test_paths
    ft_spec = FinetuneSpec(
        num_head_layers,
        head_lr=lrs[0],
        body_lr=lrs[1],
        unfreeze_body_epoch_num=unfreeze_body_epoch_num,
    )
    print(test_paths)
    print(ft_spec)
