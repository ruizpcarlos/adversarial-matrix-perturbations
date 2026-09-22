"""
adv_matrix.py::AdvPerturbation -- the shared base class for both search
strategies. Focuses on the index/genome bookkeeping (easy to get an off-by-
one in) and constructor validation. compute_max_err / random_perturbation
are exercised too, via the force_cpu fixture, as smoke tests of the search
loop mechanics -- not as tests of real CPU/GPU numeric divergence.
"""
import numpy as np
import pytest
import torch

from adv_matrix import AdvPerturbation


class TestConstructorValidation:
    def test_rejects_q_outside_open_unit_interval(self, tiny_matrix):
        func = torch.randn(4, 4)
        with pytest.raises(ValueError):
            AdvPerturbation(tiny_matrix, func, q=1.5, func_gpu=func.clone())
        with pytest.raises(ValueError):
            AdvPerturbation(tiny_matrix, func, q=0.0, func_gpu=func.clone())

    def test_rejects_shape_mismatch(self, tiny_matrix):
        bad_func = torch.randn(5, 5)
        with pytest.raises(ValueError):
            AdvPerturbation(tiny_matrix, bad_func, q=0.1, func_gpu=bad_func.clone())

    def test_rejects_non_tensor_non_module_func(self, tiny_matrix):
        with pytest.raises(TypeError):
            AdvPerturbation(tiny_matrix, "not a tensor or module", q=0.1)


class TestGenesetRoundtrip:
    def test_indices_geneset_indices_is_stable(self, tiny_adv):
        # indices -> geneset -> indices -> geneset should be a fixed point,
        # even if the intermediate index *tuples* aren't identical (order
        # isn't guaranteed), because geneset is always returned sorted.
        rng = np.random.default_rng(1)
        for _ in range(20):
            rows = rng.integers(0, 4, size=3)
            cols = rng.integers(0, 4, size=3)
            genome = tiny_adv.indices_to_geneset(rows, cols)
            rows2, cols2 = tiny_adv.geneset_to_indices(genome)
            recon = tiny_adv.indices_to_geneset(rows2, cols2)
            assert sorted(genome.tolist()) == sorted(recon.tolist())


class TestMutateGeneset:
    def test_preserves_size_and_introduces_no_duplicates(self, tiny_adv):
        genome = np.array([0, 1, 2], dtype=np.int64)
        mutated = tiny_adv.mutate_geneset(genome.copy(), n_mutations=1)
        assert len(mutated) == len(genome)
        assert len(set(mutated.tolist())) == len(mutated)

    def test_mutates_exactly_n_positions(self, tiny_adv):
        genome = np.array([0, 1, 2], dtype=np.int64)
        mutated = tiny_adv.mutate_geneset(genome.copy(), n_mutations=1)
        # symmetric difference of size 2 == exactly one old value replaced
        # by exactly one new value
        assert len(set(genome.tolist()) ^ set(mutated.tolist())) == 2


class TestCrossoverUniform:
    def test_offspring_stay_within_the_parents_universe(self, tiny_adv):
        g1 = np.array([0, 1, 2, 3], dtype=np.int64)
        g2 = np.array([2, 3, 4, 5], dtype=np.int64)
        o1, o2 = tiny_adv.crossover_uniform(g1, g2)
        universe = set(g1.tolist()) | set(g2.tolist())
        assert set(o1.tolist()) <= universe
        assert set(o2.tolist()) <= universe

    def test_offspring_sizes_match_their_parents(self, tiny_adv):
        g1 = np.array([0, 1, 2, 3], dtype=np.int64)
        g2 = np.array([2, 3, 4, 5], dtype=np.int64)
        o1, o2 = tiny_adv.crossover_uniform(g1, g2)
        assert len(o1) == len(g1)
        assert len(o2) == len(g2)


class TestSampleEntries:
    def test_unweighted_shape_and_bounds(self, tiny_adv):
        rows, cols = tiny_adv._sample_entries(num_samples=5, weighted=False)
        assert rows.shape == (5,)
        assert cols.shape == (5,)
        assert torch.all(rows < 4) and torch.all(cols < 4)

    def test_weighted_draws_without_replacement(self):
        torch.manual_seed(0)
        m = torch.randn(4, 4)
        func = torch.randn(4, 4)
        adv = AdvPerturbation(
            m, func, q=0.5, func_gpu=func.clone(),
            max_calls=4, budget_calls=8, weighted_sampling=True,
        )
        rows, cols = adv._sample_entries(num_samples=6, weighted=True)
        flat = (rows * adv.n_latent + cols).tolist()
        assert len(flat) == len(set(flat))

    def test_unweighted_sampling_allows_duplicate_positions(self, tiny_adv):
        """
        NOTE: unlike the weighted path, unweighted _sample_entries uses
        torch.randint, i.e. sampling WITH replacement. A drawn "genome" of
        k positions can therefore contain fewer than k unique flat indices.
        This matters downstream: crossover_uniform (in
        SimulatedAnnealingSearch.generate_new_sol and the GA's
        create_offspring) converts genomes to Python sets, so a duplicate in
        the input silently shrinks the effective genome size of the
        offspring. This test just pins down that duplicates are possible
        with the current sampling method -- see
        test_simulated_annealing.py::test_unweighted_sampling_can_yield_duplicate_positions
        for the downstream consequence.
        """
        torch.manual_seed(3)
        saw_duplicate = False
        for _ in range(50):
            rows, cols = tiny_adv._sample_entries(num_samples=tiny_adv.n_perturbed)
            flat = (rows * tiny_adv.n_latent + cols).tolist()
            if len(set(flat)) < len(flat):
                saw_duplicate = True
                break
        assert saw_duplicate


class TestComputeMaxErrAndRandomPerturbation:
    """These need real (patched-to-CPU) forward passes, so they're closer to
    smoke tests than pure unit tests -- they check the search loop's control
    flow and bookkeeping (call counts, caching, output shapes), not numeric
    correctness of any particular error value."""

    def test_compute_max_err_runs_and_returns_sane_types(self, tiny_adv):
        calls, err = tiny_adv.compute_max_err()
        assert isinstance(calls, int)
        assert isinstance(err, float)
        assert 1 <= calls <= tiny_adv.max_calls

    def test_full_perturbation_err_is_cached_and_does_not_recount_calls(self, tiny_adv):
        tiny_adv.compute_max_err.reset()
        err1 = tiny_adv.full_perturbation_err
        count_after_first = tiny_adv.compute_max_err.call_count
        err2 = tiny_adv.full_perturbation_err  # cached_property -> no new call
        assert err1 == err2
        assert tiny_adv.compute_max_err.call_count == count_after_first

    def test_random_perturbation_returns_shape_and_best_perturbation(self, tiny_adv):
        y, max_pert = tiny_adv.random_perturbation(early_stopping=True)
        assert y.ndim == 2
        assert isinstance(max_pert, dict)
