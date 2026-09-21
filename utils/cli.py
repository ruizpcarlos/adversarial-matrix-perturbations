import argparse
import torch


DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare random / SA / GA adversarial ULP perturbation search."
    )
    parser.add_argument("n_samples", type=int,
                        help="Number of random input matrices to sample.")
    parser.add_argument("dtype", type=str.lower, choices=list(DTYPES),
                        help="Data type of the experiment (fp32 or bf16).")
    parser.add_argument("--weighted", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Use wrap_score-weighted sampling of entries "
                             "(--weighted / --no-weighted). Default: off.")
    parser.add_argument("--matmul", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Optimize using matrix multiplication (False uses a complete model)"
                             "Default: off.")
    args = parser.parse_args()

    if args.n_samples < 1:
        parser.error("n_samples must be >= 1")
    return args
