import argparse

def add_argument():
    """Parse command-line arguments and configure the training settings."""
    parser = argparse.ArgumentParser(description="CFD")

    # For train.
    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        help="number of total epochs (default: 2)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="output logging information at a given interval",
    )

    # For mixed precision training.
    parser.add_argument(
        "--dtype",
        default="fp32",
        type=str,
        choices=["fp16", "fp32", "bf16"],
        help="Datatype used for training",
    )

    args = parser.parse_args()
    return args