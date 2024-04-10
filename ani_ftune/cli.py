r"""Command line interface entrypoints"""

import typing as tp
import typing_extensions as tpx
from pathlib import Path

from typer import Option, Typer

from ani_ftune.ftune import FinetuneSpec

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
