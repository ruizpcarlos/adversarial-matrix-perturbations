import time
import torch
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np

import torchvision.models as models

# from torch.linalg import vector_norm, multi_dot
from tqdm import tqdm

from adv_matrix import AdvPerturbation
from utils.utils import save_dict_to_pickle
from utils.plotting_utils import generation_plot


class AdversarialGeneticAlgorithm(AdvPerturbation):

    def __init__(self,
                input_matrix,
                func,
                c, 
                q,
                max_calls = 256,
                n_generations = 10, 
                pop_size=50, 
                mating_pct=0.2,
                deterministic_selection:bool=False):

        super().__init__(input_matrix, func, c, max_calls)

        assert n_generations*pop_size <= c, "Invalid combination of parameters"

        self.q  = q
        self._p = int(q*input_matrix.numel())
                
        self.n_generations = n_generations
        self.pop_size      = pop_size
        self.mating_pct    = mating_pct
        self.mating_pop    = int(mating_pct*pop_size)

        self.deterministic_selection = deterministic_selection

        self.population = []
        self.fitness    = []
        self.ulp_calls  = [0]
         
        # if self.population is None:
        #     self.population = []
        #     self.fitness    = []
        #     self.init_population()

    def init_population(self):

        first_gen = [self._sample_entries(self._p) for _ in range(self.pop_size)]
        gen_error = [self.compute_max_err(idx) for idx in first_gen]

        #
        self.population.append(first_gen)
        self.fitness.append(gen_error)
        self.sort_generation()
        self.ulp_calls.append(self.compute_max_err.call_count)


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

    ################################
    ###    MUTATION FUNCTIONS    ###
    ################################
    # @staticmethod
    # def _strides(shape):
    #     strides = [1] * len(shape)
    #     for i in range(len(shape) - 2, -1, -1):
    #         strides[i] = strides[i + 1] * shape[i + 1]
    #     return strides

    # def index_to_binary_string(self, *indices):
    #     strides = self._strides(self.input_shape)
    #     flat = torch.zeros_like(indices[0])
    #     for idx, stride in zip(indices, strides):
    #         flat = flat + idx * stride

    #     total = 1
    #     for d in self.input_shape:
    #         total *= d

    #     bits = torch.zeros(total, dtype=torch.int)
    #     bits[flat] = 1
    #     return ''.join(bits.numpy().astype(str))
    
    # def binary_string_to_index(self, s):
    #     strides = self._strides(self.input_shape)
    #     flat = torch.tensor([i for i, b in enumerate(s) if b == '1'])

    #     indices = []
    #     remainder = flat.clone()
    #     for stride in strides:
    #         indices.append(remainder // stride)
    #         remainder = remainder % stride
    #     return tuple(indices)
    
    # def index_to_binary_string(self, rows, cols):
    #     m = self.n_input
    #     n = self.n_latent

    #     flat = rows * n + cols          # row-major flat indices
    #     bits = torch.zeros(m * n, dtype=torch.int)
    #     bits[flat] = 1

    #     return ''.join(bits.numpy().astype(str))

    # def binary_string_to_index(self, s):
    #     n = self.n_input
    #     flat = torch.tensor([i for i, b in enumerate(s) if b == '1'])
    #     return flat % n, flat // n

    # def crossover_uniform(self, s1, s2):

    #     s1, s2 = list(s1), list(s2)

    #     # Separate differing positions by type
    #     zero_one = [i for i in range(len(s1)) if s1[i] == '0' and s2[i] == '1']
    #     one_zero = [i for i in range(len(s1)) if s1[i] == '1' and s2[i] == '0']

    #     # Swap the same number from each group
    #     n_swap = min(len(zero_one), len(one_zero)) // 2
    #     swap   = random.sample(zero_one, n_swap) + random.sample(one_zero, n_swap)

    #     for i in swap:
    #         s1[i], s2[i] = s2[i], s1[i]

    #     return ''.join(s1), ''.join(s2)

    # def mutate_binary_string(self, s, n_mutations=1):
    #     s = list(s)
    #     ones  = [i for i, b in enumerate(s) if b == '1']
    #     zeros = [i for i, b in enumerate(s) if b == '0']

    #     to_clear = random.sample(ones,  n_mutations)
    #     to_set   = random.sample(zeros, n_mutations)

    #     for i in to_clear: s[i] = '0'
    #     for i in to_set:   s[i] = '1'
    #     return ''.join(s)

    # -------- replaces crossover_uniform --------
    # def crossover_uniform(self, g1, g2):
    #     s1, s2 = set(np.asarray(g1).tolist()), set(np.asarray(g2).tolist())

    #     one_zero = list(s1 - s2)   # on in g1, off in g2
    #     zero_one = list(s2 - s1)   # on in g2, off in g1

    #     n_swap = min(len(zero_one), len(one_zero)) // 2
    #     to_s1 = set(random.sample(zero_one, n_swap))  # move into s1
    #     to_s2 = set(random.sample(one_zero, n_swap))  # move into s2

    #     new_s1 = (s1 - to_s2) | to_s1
    #     new_s2 = (s2 - to_s1) | to_s2

    #     return (np.array(sorted(new_s1), dtype=np.int64),
    #             np.array(sorted(new_s2), dtype=np.int64))

    # def recombine(self, g1, g2):
    #     o1, o2 = self.crossover_uniform(g1, g2)
    #     o1 = self.mutate_geneset(o1)
    #     o2 = self.mutate_geneset(o2)
    #     return o1, o2
    
    # def recombine(self, s1, s2):

    #     o1, o2 = self.crossover_uniform(s1, s2)
    #     o1 = self.mutate_binary_string(o1)
    #     o2 = self.mutate_binary_string(o2)

    #     return o1, o2

    def create_offspring(self, parent1, parent2):

        bx = self.indices_to_geneset(*parent1)
        by = self.indices_to_geneset(*parent2)

        xy1, xy2 = self.recombine(bx, by)

        xy1 = self.geneset_to_indices(xy1)
        xy2 = self.geneset_to_indices(xy2)

        return xy1, xy2

    def mating_probabilities(self):

        # Computes mating probabilities for last generation
        w_probs = [abs(err[1]) for err in self.fitness[-1]]
        if self.deterministic_selection:
            w_probs = w_probs[:self.mating_pop]
        w_probs = np.array(w_probs)
        probs   = w_probs/w_probs.sum()
        return probs

    def evolve_generation(self):

        # Score and select the mating pool
        if self.deterministic_selection:
            mating_pool   = self.population[-1][:self.mating_pop]
            mating_scores = self.fitness[-1][:self.mating_pop]
        else:
            mating_pool   = self.population[-1]
            mating_scores = self.fitness[-1]
                    
        mating_prob = self.mating_probabilities()

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
        self.ulp_calls.append(self.compute_max_err.call_count)

        # Sort and trim to keep pop_size individuals
        self.sort_generation()


    def generation_plot(self, gen=-1):
        scores = self.fitness[gen]
        best   = scores[:self.mating_pop]
        n_gen  = len(self.population)
        y_line = self.full_perturbation_err

        generation_plot(scores, best, n_gen, y_line)
    

    def search(self, early_stopping=True, verbose=False, print_plots = False):

        self.compute_max_err.reset()
        self.init_population()

        if verbose:
            aux = self.fitness[0][0]
            print(f"Initialized 1st generation -- ",
                  f"max error = {aux[1]:.3e}, ",
                  f"ulp calls = {aux[0]}")

        if print_plots:
            self.generation_plot()

        counter = 0
        j       = 1
        # pop_set = len(set(self.fitness[-1]))

        while (self.compute_max_err.call_count <= self.c 
               and j<self.n_generations
               # and pop_set > 1
               and counter < 7):

            start_t = time.time()
            self.evolve_generation()
            total_t = time.time()-start_t

            n_calls, err         = self.fitness[-1][0]
            prev_calls, prev_err = self.fitness[-2][0]
            # pop_set              = len(set(self.fitness[-1]))

            j+=1

            if early_stopping: # If early_stopping is False, the counter never grows
                counter = 0 if (err > prev_err or n_calls < prev_calls) else counter+1
                
            if verbose:
                print(f"Evolved {j} generations ({counter}) in {total_t:.3f}s -- ",
                      f"max error = {err:.4e}, ulp calls = {n_calls}")
                # print(self.compute_max_err.call_count)
            if print_plots:
                self.generation_plot()

        self.n_generations = j


    def track_max(self):

        max_err = [0] + [abs(x[0][1]) for x in self.fitness]
        idx_aux = self.ulp_calls
        y_max   = torch.zeros(self.ulp_calls[-1])

        for k in range(self.n_generations):
            y_max[idx_aux[k]: idx_aux[k+1]] = max_err[k]

        return y_max.unsqueeze(0)



if __name__ == "__main__":

    n_test     = 1   # number of repeated runs per (pop_size, p) configuration
    model_name = "ResNet"
    seed       = 420

    random.seed(seed)
    torch.manual_seed(seed)
    # ------------------------------------------------------------------
    # EXPERIMENT INPUTS
    # ------------------------------------------------------------------
    if model_name.upper().startswith("EFF"):
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).eval()
        W     = torch.transpose(model.classifier[1].weight.data, 0, 1)
    else:
        model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1).eval()
        W     = torch.transpose(model.fc.weight.data, 0, 1)
                
    n_latent = W.shape[0]

    W0    = torch.randn(n_latent, n_latent)  
    X     = torch.randn(n_test, n_latent, n_latent)
    X_img = torch.randn(n_test, 1, 3, 224, 224)
    
    c = 1000                      # total budget of calls to compute_max_err

    # ------------------------------------------------------------------
    # Grid search over population size and perturbation fraction
    # ------------------------------------------------------------------
    pop_sizes = [20, 50, 80, 100]
    q_values  = [0.05, 0.1, 0.15, 0.2]

    targets = []
    print(f"Computing target errors of the sample")
    for _x in X:
        targ_aux = AdvPerturbation(_x, W, c)
        err = targ_aux.full_perturbation_err
        targets.append(err)

    results = []
    fname   = f"grid_search_{model_name}_{seed}.pkl"

    for pop_size, q in itertools.product(pop_sizes, q_values):
        print(f"Running test for pop_size={pop_size}, q={q:.2f}")

        # n_generations must satisfy: n_generations * pop_size <= c
        n_generations = max(1, c // pop_size)

        run_errs      = []
        run_calls     = []
        run_times     = []
        run_gens      = []
        run_err_ratio = []

        for trial in tqdm(range(n_test)):
            X_test     = X[trial]
            target_err = targets[trial]

            print(f"{trial+1} - Full matrix perturbation = {target_err:.4e}")

            ga = AdversarialGeneticAlgorithm(
                input_matrix=X_test,
                func=W,
                c=c,
                q=q,
                n_generations=n_generations,
                pop_size=pop_size,
            )

            start_t = time.time()
            ga.search(early_stopping=True, verbose=False, print_plots=False)
            elapsed = time.time() - start_t

            best_calls, best_err = ga.fitness[-1][0]

            err_pct = abs(best_err)/target_err

            run_errs.append(abs(best_err))
            run_err_ratio.append(err_pct)          
            run_calls.append(ga.ulp_calls[-1])
            run_times.append(elapsed)
            run_gens.append(ga.n_generations)

            print(f"pop_size={pop_size:>4} | q={q:>5.2f} | trial={trial+1}/{n_test} | "
                  f"gens={ga.n_generations:>3} | "
                  f"best_err={abs(best_err):.4e} ({100*err_pct:.2f}%) | "
                  f"ulp_calls={best_calls:>6} | "
                  f"time={elapsed:.2f}s")

        run_errs      = np.array(run_errs)
        run_err_ratio = np.array(run_err_ratio)
        success_pct   = (run_errs >= target_err).sum()/n_test

        results.append({
            "pop_size":       pop_size,
            "p":              q,
            "n_test":         n_test,
            "target_err":     target_err,
            # "mean_err":       run_errs.mean(),
            "std_err":        run_errs.std(),
            "success_pct":    success_pct,
            "mean_err_pct":   run_err_ratio.mean(),
            "err_dist":       run_errs,
            "mean_ulp_calls": float(np.mean(run_calls)),
            "mean_time_s":    float(np.mean(run_times)),
            "max_gens":       n_generations,
            "mean_gens":      float(np.mean(run_gens)),
        })

        save_dict_to_pickle(results, filename=fname)

        print(f"  -> max_err={run_errs.max():.4e} (std={run_errs.std():.4e}) "
              f"{(100*success_pct):.2f}% success over {n_test} trials\n")

    # ------------------------------------------------------------------
    # Best configuration (highest mean max error achieved)
    # ------------------------------------------------------------------
    best = max(results, key=lambda r: r["mean_err_pct"])
    print("Best configuration:")
    print(f"  pop_size = {best['pop_size']}, q = {best['q']} "
          f"-> {(100*success_pct):.2f}% success w/ mean_err = {best['mean_err_pct']:.4e}"
          f" (std = {best['std_err']:.4e}) "
          f"over {n_test} trials "
          f"(mean ulp_calls = {best['mean_ulp_calls']:.0f}, "
          f"mean time = {best['mean_time_s']:.2f}s)")

    # ------------------------------------------------------------------
    # Heatmap of mean best error over the (pop_size, q) grid
    # ------------------------------------------------------------------
    err_grid = np.array([r["mean_err_pct"] for r in results]).reshape(
        len(pop_sizes), len(q_values)
    )

    plt.figure()
    plt.imshow(err_grid, aspect="auto", origin="lower")
    plt.colorbar(label=f"Mean best max error (n_test={n_test})")
    plt.xticks(range(len(q_values)), q_values)
    plt.yticks(range(len(pop_sizes)), pop_sizes)
    plt.xlabel("q (perturbation %)")
    plt.ylabel("Population Size (P)")
    plt.title("Grid search: pop_size vs q")
    plt.tight_layout()
    plt.show()


    succcess_grid = np.array([r["success_pct"] for r in results]).reshape(
            len(pop_sizes), len(q_values)
        )

    plt.figure()
    plt.imshow(succcess_grid, aspect="auto", origin="lower")
    plt.colorbar(label=f"Success percentage (n_test={n_test})")
    plt.xticks(range(len(q_values)), q_values)
    plt.yticks(range(len(pop_sizes)), pop_sizes)
    plt.xlabel("q (perturbation %)")
    plt.ylabel("Population Size (P)")
    plt.title("Grid search: pop_size vs q")
    plt.tight_layout()
    plt.show()
