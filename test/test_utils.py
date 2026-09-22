"""
Pure, deterministic numeric helpers in utils/utils.py. No CUDA, no model
loading -- these should be the fastest, most reliable tests in the suite.
"""
import numpy as np
import pytest
import torch

from conftest import make_float32
from utils.utils import (
    wrap_score,
    hash_tensor,
    product_err,
    vector_distance,
    pad_to_match,
    track_evol_,
    cum_stats,
    tensor_to_plotting_inputs,
    dict_to_plotting_data,
)

MAX_MANT_F32 = (1 << 23) - 1


class TestWrapScore:
    def test_one_step_from_wrap_scores_max_closeness(self):
        # sign=0 (positive), mantissa full -> steps to wrap = 1
        x = make_float32(sign=0, exponent=100, mantissa=MAX_MANT_F32)
        score = wrap_score(x, alpha=0.0, max_calls=32)
        assert torch.isclose(score, torch.tensor([1.0]))

    def test_far_from_wrap_scores_zero_closeness(self):
        # sign=0, mantissa=0 -> steps to wrap = max_mant + 1, far beyond window
        x = make_float32(sign=0, exponent=100, mantissa=0)
        score = wrap_score(x, alpha=0.0, max_calls=32)
        assert torch.isclose(score, torch.tensor([0.0]))

    def test_negative_sign_mirrors_positive(self):
        # sign=1, mantissa=0 -> steps to wrap = 1 (mirrors sign=0/mantissa=full)
        x = make_float32(sign=1, exponent=100, mantissa=0)
        score = wrap_score(x, alpha=0.0, max_calls=32)
        assert torch.isclose(score, torch.tensor([1.0]))

    def test_magnitude_bonus_orders_by_exponent(self):
        small = make_float32(sign=0, exponent=50, mantissa=MAX_MANT_F32)
        large = make_float32(sign=0, exponent=150, mantissa=MAX_MANT_F32)
        x = torch.cat([small, large])
        # alpha=1 -> pure bonus term (both entries have closeness=1)
        score = wrap_score(x, alpha=1.0, max_calls=32)
        assert score[0].item() == pytest.approx(0.0, abs=1e-6)
        assert score[1].item() == pytest.approx(1.0, abs=1e-6)

    def test_nonfinite_entries_score_zero(self):
        finite = make_float32(sign=0, exponent=100, mantissa=MAX_MANT_F32)
        inf_like = make_float32(sign=0, exponent=0xFF, mantissa=0)
        x = torch.cat([finite, inf_like])
        score = wrap_score(x, alpha=0.5, max_calls=32)
        assert score[1].item() == 0.0

    def test_bfloat16_window_is_fixed_mantissa_range(self):
        # bf16 window = 2**7 = 128 regardless of max_calls; just confirm the
        # dtype-specific branch runs and produces a valid score.
        x = torch.tensor([1.0], dtype=torch.bfloat16)
        score = wrap_score(x, alpha=0.5, max_calls=999)
        assert 0.0 <= score.item() <= 1.0


class TestHashTensor:
    def test_deterministic(self):
        t = torch.randn(3, 3)
        assert hash_tensor(t) == hash_tensor(t.clone())

    def test_sensitive_to_small_value_changes(self):
        t1 = torch.zeros(3)
        t2 = torch.zeros(3)
        t2[0] = 1e-6
        assert hash_tensor(t1) != hash_tensor(t2)

    def test_multi_tensor_order_matters(self):
        a = torch.randn(2)
        b = torch.randn(2)
        assert hash_tensor(a, b) != hash_tensor(b, a)

    def test_bfloat16_hashes_as_upcast_float32(self):
        # hash_tensor special-cases bfloat16 by upcasting to float32 before
        # hashing; confirm that's actually happening (not silently hashing
        # raw bf16 bytes), since the injector pipeline's whole notion of
        # "same activation" depends on this being consistent.
        t_bf16 = torch.randn(4).to(torch.bfloat16)
        assert hash_tensor(t_bf16) == hash_tensor(t_bf16.float())


class TestProductErrAndVectorDistance:
    # product_err chains torch.linalg.multi_dot over the matrix list, which
    # requires at least 2 matrices -- mirror the codebase's actual call shape
    # ([input] + weights) rather than a single matrix.
    def test_zero_error_for_identical_chains(self):
        a, b = torch.eye(3), torch.randn(3, 3)
        err = product_err([a, b], [a.clone(), b.clone()])
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_matches_manual_inf_norm_of_the_product(self):
        a, b = torch.eye(3), torch.eye(3)
        c = torch.eye(3)
        c[0, 0] += 0.5
        err = product_err([a, b], [a.clone(), c])
        expected = vector_distance(a @ b, a @ c)  # default ord=inf
        assert err == pytest.approx(expected, rel=1e-5)

    def test_custom_objective_fn_is_actually_used(self):
        a, b = torch.eye(3), torch.eye(3)
        c = torch.eye(3)
        c[0, 0] += 1.0
        calls = []

        def fake_objective(x, y):
            calls.append((x, y))
            return torch.tensor(42.0)

        result = product_err([a, b], [a.clone(), c], objective_fn=fake_objective)
        assert result == 42.0
        assert len(calls) == 1

    def test_vector_distance_ord_variants(self):
        x = torch.tensor([1.0, -3.0, 2.0])
        y = torch.tensor([0.0, 0.0, 0.0])
        assert vector_distance(x, y, ord=np.inf) == pytest.approx(3.0)
        assert vector_distance(x, y, ord=2) == pytest.approx(
            (1**2 + 3**2 + 2**2) ** 0.5
        )


class TestSmallPlottingHelpers:
    def test_pad_to_match_replicate_pads_shorter_tensors(self):
        t1 = torch.tensor([[1.0, 2.0, 3.0]])
        t2 = torch.tensor([[4.0, 5.0]])
        out = pad_to_match([t1, t2])
        assert out.shape == (2, 3)
        assert out[1].tolist() == [4.0, 5.0, 5.0]  # last value replicated

    def test_track_evol_fills_plateaus_between_checkpoints(self):
        Y = torch.tensor([1.0, 2.0, 3.0])
        n_calls = [0, 2, 5, 7]
        out = track_evol_(Y, n_calls)
        assert out.shape == (1, 7)
        assert out[0, 0:2].tolist() == [1.0, 1.0]
        assert out[0, 2:5].tolist() == [2.0, 2.0, 2.0]
        assert out[0, 5:7].tolist() == [3.0, 3.0]

    def test_cum_stats_shape_and_monotonic_running_max(self):
        y = torch.rand(5, 6)
        stats = cum_stats(y)
        assert stats.shape == (4, 6)  # [max, q99, q75, median] x m columns
        cummax_row = stats[0]
        assert torch.all(cummax_row[1:] >= cummax_row[:-1] - 1e-6)

    def test_tensor_to_plotting_inputs_buckets_by_power_of_two(self):
        y = torch.rand(3, 64)
        df, y_dist = tensor_to_plotting_inputs(y)
        assert set(df["n_calls"].unique()) == {32, 64}
        assert y_dist.shape[0] == y.numel()

    def test_dict_to_plotting_data_concatenates_all_groups(self):
        y_stats = {"0.1": torch.tensor([1.0, 2.0]), "0.5": torch.tensor([3.0])}
        df, y_hist = dict_to_plotting_data(y_stats)
        assert len(df) == 3
        assert sorted(y_hist.tolist()) == [1.0, 2.0, 3.0]
