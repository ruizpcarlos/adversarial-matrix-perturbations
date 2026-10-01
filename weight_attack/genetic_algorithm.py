import gc
import time
import itertools
from typing import Optional

import numpy as np
import torch

from .adv_layer import AdvLayerPerturbation


class AdvLayerGeneticAlgorithm(AdvLayerPerturbation):
    """
    Genetic algorithm over sets of weight indices (genome = sorted flat indices into W).

    Batch handling, scoring and fitness (A -> best_calls -> ULPs -> B) live in
    AdvLayerPerturbation: resample(), fitness(), test_score(). A fresh (A, B) pair
    is drawn once per generation and shared by all its members, so every member
    (mating pool included) is re-scored each generation.
    """

    def __init__(self, *args,
                 pop_size: int = 50,
                 mating_pct: float = 0.4,
                 mating_temp: float = 0.05,
                 offspring_mutation_rate: Optional[float] = None,  # None -> self.mutation_rate
                 keep_full_history: bool = False,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.n_generations = n_generations = max(1, self.budget_calls // pop_size)
        self.stop_counter  = max(10, 1 + n_generations // 2)
        self.pop_size      = pop_size
        self.mating_pct    = mating_pct
        self.mating_pop    = int(mating_pct * pop_size)
        self.mating_temp   = mating_temp
        self.offspring_mutation_rate = offspring_mutation_rate

        self.keep_full_history = keep_full_history

        self.population   = []
        self.fitness_hist = []   # per generation: [(best_calls, fitness), ...]
        self.history      = []   # [(best_calls, best_fitness), ...] one per generation
        self.ulp_calls    = []   # cumulative n_evals after each generation
        self.best_C       = []

        self.solution = None

    # NOTE: `fitness` is the base-class method (score one genome), so the per-generation
    # fitness lists live in `self.fitness_hist`.

    ##################################################################
    #                    SCORING
    ##################################################################
    def score_generation(self, genomes, agg=None):
        """Draws a fresh (A, B) pair, returns [(best_calls, fitness), ...] aligned with genomes."""
        self.resample()
        out = []
        for g in genomes:
            fit, n_ulp, _ = self.fitness(g, agg)
            out.append((n_ulp, fit))
        return out

    ##################################################################
    #                    GENETIC ALGORITHM
    ##################################################################
    def init_population(self, agg=None):

        first_gen = [self.random_geneset() for _ in range(self.pop_size)]
        gen_fitness = self.score_generation(first_gen, agg)

        self.population.append(first_gen)
        self.fitness_hist.append(gen_fitness)
        self.sort_generation()
        self.ulp_calls.append(self.n_evals)
        self._record_generation()

    def _record_generation(self):
        """Save this generation's summary, and — unless keep_full_history is set —
        free the *previous* generation's full population/fitness arrays."""
        best_calls, best_fit = self.fitness_hist[-1][0]
        self.history.append((best_calls, best_fit))

        test = self.test_score(self.population[-1][0], best_calls)
        self.best_C.append(test)

        if not self.keep_full_history and len(self.population) > 1:
            self.population[-2]   = None
            self.fitness_hist[-2] = None

    def sort_generation(self, gen=-1):
        current_gen = self.population[gen]
        gen_fitness = self.fitness_hist[gen]

        # Sorts by descending fitness 1st, asc calls 2nd
        scored_pop = sorted(
                            zip(current_gen, gen_fitness),
                            key=lambda x: (-x[1][1], x[1][0])
                        )
        current_gen, gen_fitness = zip(*scored_pop[:self.pop_size])

        self.population[gen]   = list(current_gen)
        self.fitness_hist[gen] = list(gen_fitness)

    def create_offspring(self, parent1, parent2):
        return self.recombine(parent1, parent2,
                              mutation_rate=self.offspring_mutation_rate)

    def mating_probabilities(self):
        # Boltzmann distribution over the mating pool's fitness.
        scores = np.array(
            [fit[1] for fit in self.fitness_hist[-1][:self.mating_pop]],
            dtype=float,
        )
        scaled  = (scores - scores.max()) / self.mating_temp
        w_probs = np.exp(scaled)
        return w_probs / w_probs.sum()

    def evolve_generation(self, agg=None):

        mating_pool = self.population[-1][:self.mating_pop]
        mating_prob = self.mating_probabilities()

        n_pairs = self.pop_size // 2
        pairs   = [
            [mating_pool[i] for i in self.rng.choice(len(mating_pool), size=2, p=mating_prob)]
            for _ in range(n_pairs)
        ]

        offspring = list(itertools.chain.from_iterable(
                            self.create_offspring(p1, p2) for p1, p2 in pairs
                        ))

        new_gen = offspring + mating_pool

        # Whole generation is re-scored on a fresh (A, B) pair
        gen_fitness = self.score_generation(new_gen, agg)

        self.population.append(new_gen)
        self.fitness_hist.append(gen_fitness)
        self.ulp_calls.append(self.n_evals)

        self.sort_generation()
        self._record_generation()

    def search(self, early_stopping=True, verbose=False, agg=None):

        self.n_evals   = 0
        self.ulp_calls = [0]
        self.history.clear()
        self.population.clear()
        self.fitness_hist.clear()

        start_t = time.time()
        self.init_population(agg)
        total_t = time.time() - start_t

        if verbose:
            aux = self.fitness_hist[0][0]
            print(f"Initialized 1st generation ({total_t:.3f}s)-- ",
                  f"fitness = {aux[1]:.4f}, ",
                  f"ulp calls = {aux[0]}")

        counter  = 0
        j        = 1
        best_fit = self.history[0][1]

        while (self.n_evals + self.pop_size + self.mating_pop <= self.budget_calls
               and j < self.n_generations
               and counter < self.stop_counter):

            start_t = time.time()
            self.evolve_generation(agg)
            total_t = time.time() - start_t

            n_calls, fit = self.fitness_hist[-1][0]

            j += 1

            # Fitness is noisy (batches change), so compare against the best seen so far
            if early_stopping:
                counter = 0 if fit > best_fit else counter + 1
            best_fit = max(best_fit, fit)

            if verbose:
                print(f"Evolved {j} generations ({counter}) in {total_t:.3f}s -- ",
                      f"fitness = {fit:.4f}, ulp calls = {n_calls}")

        self.n_generations = j
        self.solution = (self.population[-1][0],
                         self.fitness_hist[-1][0][0])   # (genome, best_calls)

        return self.solution

    def track_max(self, score:str):
        assert score in["train", "val"], "Scored set name must be 'train' or 'val'"

        if score=='train':
            max_fit = [0] + [fit for _, fit in self.history]
        else:
            max_fit = [0] + [fit for fit in self.best_C]

        idx_aux = self.ulp_calls
        y_max   = torch.zeros(self.ulp_calls[-1])

        for k in range(self.n_generations):
            y_max[idx_aux[k]: idx_aux[k+1]] = max_fit[k]

        return y_max.unsqueeze(0)

    ##################################################################
    #                  VALIDATION / OUTPUT
    ##################################################################
    def validate(self, top_k=1, batch_size=None, agg=None):
        """Re-score the top_k individuals of the last generation on the held-out batch C."""
        return [
            dict(calls=fit[0],
                 fitness_B=fit[1],
                 fitness_C=self.test_score(genome, fit[0], batch_size=batch_size, agg=agg))
            for genome, fit in zip(self.population[-1][:top_k], self.fitness_hist[-1][:top_k])
        ]

    def final_score(self, batch_size=1024):
        """Estimate on a fresh, larger batch that never influenced the search."""
        genome, n_ulp = self.solution
        return self.test_score(genome, n_ulp, batch_size=batch_size, fresh=True)

    def compute_adv_weights(self):
        genome, n_calls = self.solution
        return self.apply_ulps(genome, n_calls)

    def release(self):
        """Explicitly drop this instance's large in-memory state."""
        self.population.clear()
        self.fitness_hist.clear()
        self.history.clear()
        self.batch_A = self.batch_B = self.batch_C = None
        self.h_cpu = self.h_gpu = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
