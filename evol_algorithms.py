import time
import torch
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np

from torch.linalg import vector_norm, multi_dot
from adv_matrix import AdvPerturbation, _nextafter


class AdversarialGeneticAlgorithm(AdvPerturbation):

    def __init__(self,
                input_matrix,
                weights,
                p,
                max_iter = 256,
                n_generations = 10, 
                pop_size=50, 
                mating_pct=0.2):

        super().__init__(input_matrix, weights, p, max_iter)

        self.n_generations = n_generations
        self.pop_size      = pop_size
        self.mating_pct    = mating_pct
        self.mating_pop    = int(mating_pct*pop_size)

        self.population = []
        self.fitness    = []
        self.ulp_calls  = [0]
         
        if self.population is None:
            self.population = []
            self.fitness    = []
            self.init_population()

    # def __init__(self, input_matrix, weights, p,
    #              max_iter = 1024,
    #              n_generations = 10, pop_size=50, mating_pop=10):

    #     self.input_matrix = input_matrix

    #     if isinstance(weights, torch.Tensor):
    #         weights = [weights]

    #     # Dimension check using consecutive pairs
    #     _matrices = [input_matrix] + weights
    #     for a, b in zip(_matrices, _matrices[1:]):
    #         if a.shape[-1] != b.shape[-2]:
    #             raise ValueError(
    #                 f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)} — "
    #                 f"dim {a.shape[-1]} != {b.shape[-2]}"
    #             )

    #     self.weights     = weights
    #     self.weights_gpu = [m.to("cuda") for m in self.weights]

    #     self.n_input    = input_matrix.shape[0]
    #     self.n_latent   = input_matrix.shape[1]

    #     self.p          = p
    #     self.max_iter   = max_iter
    #     self.n_gens     = n_generations
    #     self.pop_size   = pop_size
    #     self.mating_pop = mating_pop

    #     self.population = None
    #     self.fitness    = None

    #     if self.population is None:
    #         self.population = []
    #         self.fitness    = []
    #         self.init_population()

    def init_population(self):

        first_gen = [self._sample_entries() for _ in range(self.pop_size)]
        gen_error = [self.compute_max_err(idx) for idx in first_gen]

        #
        self.population.append(first_gen)
        self.fitness.append(gen_error)
        self.sort_generation()
        self.ulp_calls.append(_nextafter.call_count)


    def sort_generation(self, gen=-1):
        # Ranks last generation (by default)
        current_gen = self.population[gen]
        gen_fitness = self.fitness[gen]

        # Sorts by descending error 1st, asc calls to max 2nd
        scored_pop = sorted(
                            zip(current_gen, gen_fitness),
                            key=lambda x: (-abs(x[1][1]), x[1][0])
                        )
        # Keeps only pop_size individuals - for new gens
        current_gen, gen_fitness = zip(*scored_pop[:self.pop_size])

        self.population[gen] = list(current_gen)
        self.fitness[gen]    = list(gen_fitness)

    def compute_max_err(self, indices):

        # def chain_matmul(m, w):
        #     return reduce(torch.matmul, w, m)

        M_    = self.input_matrix.clone()
        M_gpu = M_.to("cuda")
        infty = torch.tensor(torch.inf)

        # weights_cpu = self.weights
        mat_cpu     = [None] + self.weights
        mat_gpu     = [None] + self.weights_gpu

        abs_err      = 0
        max_error    = 0
        calls_to_max = 1

        for i in range(self.max_iter):

            # M_[indices] = nextafter(M_[indices], 1)
            # torch wrapped in counter
            M_[indices] = _nextafter(M_[indices], infty)

            M_gpu.copy_(M_, non_blocking=True)

            mat_cpu[0] = M_
            mat_gpu[0] = M_gpu

            y_cpu  = multi_dot(mat_cpu)
            y_gpu  = multi_dot(mat_gpu)
            y_diff = (y_cpu - y_gpu.cpu()).ravel().squeeze()

            if len(y_diff.shape) > 0:
                y_ = vector_norm(y_diff, ord=np.inf).item()
            else:
                y_ = y_diff.item()

            if abs(y_) > abs_err:
                calls_to_max = i+1
                abs_err      = abs(y_)
                max_error    = y_

        return calls_to_max, max_error

    ################################
    ###    MUTATION FUNCTIONS    ###
    ################################
    def index_to_binary_string(self, rows, cols):
        m = self.n_input
        n = self.n_latent

        flat = rows * n + cols          # row-major flat indices
        bits = torch.zeros(m * n, dtype=torch.int)
        bits[flat] = 1

        return ''.join(bits.numpy().astype(str))

    def binary_string_to_index(self, s):
        n = self.n_input
        flat = torch.tensor([i for i, b in enumerate(s) if b == '1'])
        return flat % n, flat // n

    def crossover_uniform(self, s1, s2):

        s1, s2 = list(s1), list(s2)

        # Separate differing positions by type
        zero_one = [i for i in range(len(s1)) if s1[i] == '0' and s2[i] == '1']
        one_zero = [i for i in range(len(s1)) if s1[i] == '1' and s2[i] == '0']

        # Swap the same number from each group
        n_swap = min(len(zero_one), len(one_zero)) // 2
        swap   = random.sample(zero_one, n_swap) + random.sample(one_zero, n_swap)

        for i in swap:
            s1[i], s2[i] = s2[i], s1[i]

        return ''.join(s1), ''.join(s2)

    def mutate_binary_string(self, s, n_mutations=1):
        s = list(s)
        ones  = [i for i, b in enumerate(s) if b == '1']
        zeros = [i for i, b in enumerate(s) if b == '0']

        to_clear = random.sample(ones,  n_mutations)
        to_set   = random.sample(zeros, n_mutations)

        for i in to_clear: s[i] = '0'
        for i in to_set:   s[i] = '1'
        return ''.join(s)

    def recombine(self, s1, s2):

        o1, o2 = self.crossover_uniform(s1, s2)
        o1 = self.mutate_binary_string(o1)
        o2 = self.mutate_binary_string(o2)

        return o1, o2

    def create_offspring(self, parent1, parent2):

        x_str = self.index_to_binary_string(*parent1)
        y_str = self.index_to_binary_string(*parent2)

        xy1, xy2 = self.recombine(x_str, y_str)

        xy1 = self.binary_string_to_index(xy1)
        xy2 = self.binary_string_to_index(xy2)

        return xy1, xy2

    def mating_probabilities(self):

        # Computes mating probabilities for last generation
        w_probs = [abs(err[1]) for err in self.fitness[-1]][:self.mating_pop]
        w_probs = np.array(w_probs)
        probs   = w_probs/w_probs.sum()
        return probs

    def evolve_generation(self):

        # Score and select the mating pool
        mating_pool   = self.population[-1][:self.mating_pop]
        mating_scores = self.fitness[-1][:self.mating_pop]
        mating_prob   = self.mating_probabilities()

        # Generate offspring from all pairs in the mating pool
        n_pairs = self.pop_size // 2
        pairs   = [random.choices(mating_pool,
                                  weights=mating_prob,
                                  k=2)
                    for _ in range(n_pairs)]

        # Flatten
        offspring = list(itertools.chain.from_iterable(
                            self.create_offspring(p1, p2) for p1, p2 in pairs
                        ))

        new_gen   = offspring + mating_pool
        gen_error = [self.compute_max_err(idx) for idx in offspring] + mating_scores

        self.population.append(new_gen)
        self.fitness.append(gen_error)
        self.ulp_calls.append(_nextafter.call_count)

        # Sort and trim to keep pop_size individuals
        self.sort_generation()

    def generation_plot(self, gen=-1):
        _scores = self.fitness[gen]
        _best   = _scores[:self.mating_pop]

        plt.title(f"Generation {len(self.population)}")
        plt.xlabel("ULP calls")
        plt.ylabel("Max Error")
        plt.scatter([x[0] for x in _scores],
                    [x[1] for x in _scores])
        plt.scatter([x[0] for x in _best],
                    [x[1] for x in _best],
                    marker = '*')
        plt.show()

    def search(self, early_stopping=True, verbose=False, print_plots = False):

        _nextafter.reset()
        self.init_population()

        if verbose:
            print(f"Initialized 1st generation -- ",
                  f"max error = {self.fitness[-1][0][1]:.3f}, ",
                  f"ulp calls = {self.fitness[-1][0][0]}")

        if print_plots:
            self.generation_plot()

        counter   = 0
        j = 1

        while _nextafter.call_count < self.max_ulp_calls and counter < 7:

            start_t = time.time()
            self.evolve_generation()
            total_t = time.time()-start_t
            n_calls, err = self.fitness[-1][0]
            prev_calls, prev_err = self.fitness[-2][0]

            if early_stopping: # If early_stopping is False, the counter never grows
                counter = 0 if (err > prev_err or n_calls < prev_calls) else counter+1
                
            if verbose:
                print(f"Evolved {j+1} generations ({counter}) in {total_t:.3f}s -- ",
                      f"max error = {err:.4e}, ulp calls = {n_calls}")
                # print(_nextafter.call_count)
            if print_plots:
                self.generation_plot()
            
            j+=1

        self.n_generations = j


    def track_max(self):

        max_err = [0] + [abs(x[0][1]) for x in self.fitness]
        idx_aux = self.ulp_calls
        y_max   = torch.zeros(self.ulp_calls[-1])

        for k in range(self.n_generations):
            y_max[idx_aux[k]: idx_aux[k+1]] = max_err[k]

        return y_max.unsqueeze(0)