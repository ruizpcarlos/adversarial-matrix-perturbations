"""
genetic_algorithm.py::AdversarialGeneticAlgorithm. Most tests manipulate
`population`/`fitness` directly rather than running a real search, so they
isolate the bookkeeping logic (sorting, history recording, memory release)
from the expensive/GPU-flavored parts already covered in test_adv_matrix.py.
One end-to-end smoke test exercises search() itself via the force_cpu shim.
"""
import numpy as np
import pytest


class TestSortGeneration:
    def test_orders_by_descending_error_then_ascending_calls(self, tiny_ga):
        # fitness tuples are (calls_to_max, err); best = highest err,
        # tie-break by fewer calls
        fake_pop = ["a", "b", "c", "d"]
        fake_fit = [(2, 1.0), (1, 5.0), (1, 5.0), (3, 0.5)]
        tiny_ga.population.append(fake_pop)
        tiny_ga.fitness.append(fake_fit)

        tiny_ga.sort_generation()

        assert tiny_ga.fitness[-1][0] == (1, 5.0)
        assert tiny_ga.fitness[-1][:2] == [(1, 5.0), (1, 5.0)]
        assert set(tiny_ga.population[-1][:2]) == {"b", "c"}

    def test_truncates_to_pop_size(self, tiny_ga):
        fake_pop = list(range(10))
        fake_fit = [(i, float(10 - i)) for i in range(10)]
        tiny_ga.population.append(fake_pop)
        tiny_ga.fitness.append(fake_fit)

        tiny_ga.sort_generation()

        assert len(tiny_ga.population[-1]) == tiny_ga.pop_size
        assert len(tiny_ga.fitness[-1]) == tiny_ga.pop_size


class TestMatingProbabilities:
    def test_returns_a_valid_probability_distribution(self, tiny_ga):
        tiny_ga.fitness.append([(1, 3.0), (2, 3.0), (1, 1.0), (1, 0.0)])
        probs = tiny_ga.mating_probabilities()
        assert probs.shape[0] == tiny_ga.mating_pop
        assert np.all(probs >= 0)
        assert probs.sum() == pytest.approx(1.0)

    def test_equal_scores_give_uniform_probabilities(self, tiny_ga):
        tiny_ga.fitness.append([(1, 2.0), (1, 2.0)])
        tiny_ga.mating_pop = 2
        probs = tiny_ga.mating_probabilities()
        assert probs[0] == pytest.approx(probs[1])


class TestRecordGeneration:
    def test_appends_best_of_each_generation_to_history(self, tiny_ga):
        tiny_ga.population.append(["gen0_a", "gen0_b"])
        tiny_ga.fitness.append([(1, 5.0), (2, 4.0)])
        tiny_ga._record_generation()
        assert tiny_ga.history == [(1, 5.0)]

        tiny_ga.population.append(["gen1_a", "gen1_b"])
        tiny_ga.fitness.append([(1, 6.0), (2, 4.5)])
        tiny_ga._record_generation()

        assert tiny_ga.history == [(1, 5.0), (1, 6.0)]

    def test_frees_previous_generations_population_and_fitness(self, tiny_ga):
        tiny_ga.population.append(["gen0_a"])
        tiny_ga.fitness.append([(1, 5.0)])
        tiny_ga._record_generation()

        tiny_ga.population.append(["gen1_a"])
        tiny_ga.fitness.append([(1, 6.0)])
        tiny_ga._record_generation()

        assert tiny_ga.population[-2] is None
        assert tiny_ga.fitness[-2] is None
        assert tiny_ga.population[-1] is not None  # current gen kept

    def test_keep_full_history_flag_disables_freeing(self, tiny_ga):
        tiny_ga.keep_full_history = True
        tiny_ga.population.append(["gen0"])
        tiny_ga.fitness.append([(1, 1.0)])
        tiny_ga._record_generation()

        tiny_ga.population.append(["gen1"])
        tiny_ga.fitness.append([(1, 2.0)])
        tiny_ga._record_generation()

        assert tiny_ga.population[-2] is not None
        assert tiny_ga.fitness[-2] is not None


class TestRelease:
    def test_clears_population_fitness_history_and_gpu_weights(self, tiny_ga):
        tiny_ga.population.append(["x"])
        tiny_ga.fitness.append([(1, 1.0)])
        tiny_ga.history.append((1, 1.0))
        assert tiny_ga.weights_gpu is not None  # sanity check before release

        tiny_ga.release()

        assert tiny_ga.population == []
        assert tiny_ga.fitness == []
        assert tiny_ga.history == []
        assert tiny_ga.weights_gpu is None


class TestSearchEndToEnd:
    def test_search_runs_and_produces_a_solution(self, tiny_ga):
        tiny_ga.search(early_stopping=True, verbose=False)
        assert tiny_ga.solution is not None
        best_idx, best_calls = tiny_ga.solution
        assert isinstance(best_calls, int)
