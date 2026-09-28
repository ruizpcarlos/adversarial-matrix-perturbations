import argparse
import torch


DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

def parse_args():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("n_samples", type=int, help="Number of samples to run.")
    args = parser.parse_args()

    if args.n_samples < 1:
        parser.error("n_samples must be >= 1")
    return args

def add_common_args(parser, *, dtype=False, weighted=False, matmul=False):
    parser.add_argument("n_samples", type=int, help="Number of samples to run.")
    if dtype:
        parser.add_argument("dtype", type=str.lower,
                            choices=["fp32", "float32", "bf16", "bfloat16"],
                            help="Data type of the experiment.")
    if weighted:
        parser.add_argument("--weighted", action=argparse.BooleanOptionalAction,
                            default=False)
    if matmul:
        parser.add_argument("--matmul", action=argparse.BooleanOptionalAction,
                            default=True,
                            help="Optimize using matrix multiplication (False uses a complete model)"
                                "Default: off.")
    return parser

# Use like:
# parser = argparse.ArgumentParser()
# add_common_args(parser)
# args = parser.parse_args()
