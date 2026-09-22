"""
simulated_annealing.py::SimulatedAnnealingSearch. Covers the genome-drawing
primitives specific to SA, the temperature schedule, and one end-to-end
smoke test of search() under the force_cpu shim.
"""
import numpy as np
import pytest
import torch


class TestRandomGeneset:
    def test_has_correct_size_and_no_duplicates(self, tiny_sa):
        genome = tiny_sa.random_geneset()
        assert len(genome) == tiny_sa.n_perturbed
        assert len(set(genome.tolist())) == len(genome)
        assert np.all(genome >= 0) and np.all(genome < tiny_sa.total)

    def test_is_returned_pre_sorted(self, tiny_sa):
        genome = tiny_sa.random_geneset()
        assert list(genome) == sorted(genome.tolist())


class TestGenerateNewSol:
    def test_returns_indices_of_the_same_arity(self, tiny_sa):
        idx = tiny_sa._sample_entries(tiny_sa.n_perturbed)
        new_idx = tiny_sa.generate_new_sol(idx)
        assert len(new_idx) == len(idx)  # same number of index dimensions

    def test_returns_indices_within_bounds(self, tiny_sa):
        idx = tiny_sa._sample_entries(tiny_sa.n_perturbed)
        new_idx = tiny_sa.generate_new_sol(idx)
        genome = tiny_sa.indices_to_geneset(*new_idx)
        # NOTE: intentionally not asserting len(genome) == n_perturbed here.
        # See test_unweighted_sampling_can_yield_duplicate_positions below.
        assert len(genome) <= tiny_sa.n_perturbed
        assert np.all(genome >= 0) and np.all(genome < tiny_sa.total)

    def test_unweighted_sampling_can_yield_duplicate_positions(self, tiny_sa):
        """
        _sample_entries draws with replacement when unweighted
        (torch.randint), so a "genome" of n_perturbed indices can contain
        fewer than n_perturbed *unique* flat positions. generate_new_sol
        feeds such a genome into crossover_uniform, which converts genomes
        to Python sets -- so a duplicate in the input silently shrinks the
        effective genome size of the offspring for that step of the search.
        This is most visible on small matrices (as here); on realistic
        matrix sizes the birthday-paradox probability of a collision within
        one draw is much lower, but not zero. This test documents the
        current, observed behavior so a future change (e.g. switching genome
        construction to sampling without replacement) shows up here as an
        intentional change rather than a silent one.
        """
        torch.manual_seed(3)
        saw_duplicate = False
        for _ in range(50):
            idx = tiny_sa._sample_entries(tiny_sa.n_perturbed)
            genome = tiny_sa.indices_to_geneset(*idx)
            if len(set(genome.tolist())) < len(genome):
                saw_duplicate = True
                break
        assert saw_duplicate, (
            "Expected at least one duplicate-position draw over 50 samples "
            "given a small total search space; if this starts failing, "
            "sampling behavior may have changed."
        )


class TestSearchEndToEnd:
    def test_search_runs_and_produces_a_solution(self, tiny_sa):
        # Pin T0 directly: on a search space this small, init_temp's
        # sample-100-random-moves calibration can find zero "worse" moves,
        # leaving T0 as NaN (a real edge case in init_temp worth knowing
        # about, but orthogonal to what this test is checking).
        tiny_sa.T0 = 1.0
        Y, solution, temps, accept_probs = tiny_sa.search(L0=2, verbose=False)
        assert Y.ndim == 1
        assert len(Y) >= 1
        best_idx, best_calls = solution
        assert isinstance(best_calls, int)
        assert len(temps) == len(Y)

    def test_temperature_schedule_decays_geometrically_with_alpha(self, tiny_sa):
        tiny_sa.T0 = 1.0  # skip init_temp so the schedule is deterministic
        _, _, temps, _ = tiny_sa.search(L0=2, alpha=0.5, verbose=False)
        for prev, cur in zip(temps, temps[1:]):
            assert cur == pytest.approx(prev * 0.5)
