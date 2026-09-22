"""
Shared fixtures for the whole suite.

Run from the project root with:  pytest tests/ -v
(these files assume adv_matrix.py, genetic_algorithm.py, simulated_annealing.py
and the utils/ package are importable from the current working directory).
"""
import numpy as np
import pytest
import torch

from adv_matrix import AdvPerturbation
from genetic_algorithm import AdversarialGeneticAlgorithm
from simulated_annealing import SimulatedAnnealingSearch


# ---------------------------------------------------------------------------
# CUDA -> CPU shim
# ---------------------------------------------------------------------------
# A lot of the interesting algorithmic logic in this codebase (compute_max_err,
# random_perturbation, AdversarialGeneticAlgorithm.release, ...) hardcodes
# `.to("cuda")` / `torch.cuda.is_available()` even though nothing about the
# logic itself is CUDA-specific -- it's there to compare CPU vs GPU numerics
# on real hardware. That makes it untestable in a CPU-only CI runner as
# written. This autouse fixture transparently redirects any `.to("cuda")` /
# `device="cuda"` call to `"cpu"` and reports `is_available()` as False, so
# we can unit test the *algorithm* (sorting, sampling, genome bookkeeping,
# search loops, budget accounting) without a GPU.
#
# This deliberately does NOT test CPU/GPU numeric divergence -- that still
# needs real hardware and is out of scope for unit tests.
@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    original_to = torch.Tensor.to

    def patched_to(self, *args, **kwargs):
        args = tuple("cpu" if a == "cuda" else a for a in args)
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", patched_to)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def make_float32(sign: int, exponent: int, mantissa: int) -> torch.Tensor:
    """Build a single-element float32 tensor from raw sign/exponent/mantissa
    bit-fields, so tests can target exact ULP-boundary cases in wrap_score
    instead of guessing which random float lands where. Built via a uint32
    buffer to avoid signed-int32 overflow when the sign bit is set."""
    bits = (sign << 31) | (exponent << 23) | mantissa
    arr = np.array([bits], dtype=np.uint32)
    return torch.from_numpy(arr).view(torch.float32)


# ---------------------------------------------------------------------------
# Small, fast fixtures shared across test modules
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_matrix():
    torch.manual_seed(0)
    return torch.randn(4, 4)


@pytest.fixture
def tiny_adv(tiny_matrix):
    """A small AdvPerturbation over a 4x4 @ 4x4 tensor product. Passing
    func_gpu explicitly (as a CPU tensor) avoids the constructor's internal
    default of `.to("cuda")`, on top of the force_cpu shim above."""
    func = torch.randn(4, 4)
    return AdvPerturbation(
        tiny_matrix, func, q=0.25,
        func_gpu=func.clone(),
        max_calls=4, budget_calls=8,
    )


@pytest.fixture
def tiny_ga():
    torch.manual_seed(0)
    m = torch.randn(4, 4)
    func = torch.randn(4, 4)
    return AdversarialGeneticAlgorithm(
        m, func, q=0.25,
        func_gpu=func.clone(),
        max_calls=2, budget_calls=8,
        pop_size=4, mating_pct=0.5,
    )


@pytest.fixture
def tiny_sa():
    torch.manual_seed(0)
    m = torch.randn(4, 4)
    func = torch.randn(4, 4)
    return SimulatedAnnealingSearch(
        m, func, q=0.25,
        func_gpu=func.clone(),
        max_calls=2, budget_calls=20,
    )
