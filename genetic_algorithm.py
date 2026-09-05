import time
import sys
import gc
import torch
import random
import itertools
import numpy as np
import threading
import torchvision.models as models

from concurrent.futures import ThreadPoolExecutor

from adv_matrix import AdvPerturbation
from utils.utils import save_dict_to_pickle
from utils.plotting_utils import generation_plot


_call_lock = threading.Lock()

class AdversarialGeneticAlgorithm(AdvPerturbation):

    def __init__(self,
                input_matrix,
                func,
                c, 
                q,
                func_gpu=None,
                max_calls = 32,
                n_generations = 10, 
                pop_size=50, 
                mating_pct=0.4,
                keep_full_history=False):

        super().__init__(input_matrix, func, c, func_gpu, max_calls)

        assert n_generations*pop_size <= c, "Invalid combination of parameters"

        self.q  = q
        self._p = int(q*input_matrix.numel())
                
        self.n_generations = n_generations
        self.stop_counter  = max(1+n_generations//2, 10)
        # self.stop_counter  = 20
        self.pop_size      = pop_size
        self.mating_pct    = mating_pct
        self.mating_pop    = int(mating_pct*pop_size)

        # evolve_generation()/mating_probabilities() only ever read the last
        # entry of population/fitness. By default (keep_full_history=False)
        # we drop older generations' full genesets/index arrays as soon as
        # a new generation replaces them, keeping only a lightweight
        # (ulp_calls, best_err) summary in self.history for track_max()/plots.
        # Set keep_full_history=True if you need every generation's full
        # population retained (uses much more memory, scales with
        # n_generations * pop_size).
        self.keep_full_history = keep_full_history

        self.population = []
        self.fitness    = []
        self.history    = []   # [(ulp_calls, best_err), ...] one per generation
        self.ulp_calls  = [0]
         
        # if self.population is None:
        #     self.population = []
        #     self.fitness    = []
        #     self.init_population()

    def init_population(self):

        first_gen = [self._sample_entries(self._p) for _ in range(self.pop_size)]
        # gen_error = [self.compute_max_err(idx) for idx in first_gen]
        with ThreadPoolExecutor(max_workers=8) as ex:
            gen_error = list(ex.map(self._compute_max_err_threadsafe, first_gen))

        #
        self.population.append(first_gen)
        self.fitness.append(gen_error)
        self.sort_generation()
        self.ulp_calls.append(self.compute_max_err.call_count)
        self._record_generation()


    def _record_generation(self):
        """Save this generation's (ulp_calls, best_err) summary, and — unless
        keep_full_history is set — free the *previous* generation's full
        population/fitness arrays, since nothing reads them again once the
        next generation exists."""
        best_calls, best_err = self.fitness[-1][0]
        self.history.append((best_calls, best_err))

        if not self.keep_full_history and len(self.population) > 1:
            self.population[-2] = None
            self.fitness[-2]    = None

    def _compute_max_err_threadsafe(self, idx):
        result = self.compute_max_err.func(idx)   # bypass CallTracker's own increment (not thread-safe as-is)
        with _call_lock:
            self.compute_max_err.call_count += 1
        return result   


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


    def create_offspring(self, parent1, parent2):

        bx = self.indices_to_geneset(*parent1)
        by = self.indices_to_geneset(*parent2)

        xy1, xy2 = self.recombine(bx, by)

        xy1 = self.geneset_to_indices(xy1)
        xy2 = self.geneset_to_indices(xy2)

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

        with ThreadPoolExecutor(max_workers=8) as ex:
            gen_error = list(ex.map(self._compute_max_err_threadsafe, offspring))

        gen_error = gen_error + mating_scores

        self.population.append(new_gen)
        self.fitness.append(gen_error)
        self.ulp_calls.append(self.compute_max_err.call_count)

        # Sort and trim to keep pop_size individuals
        self.sort_generation()
        self._record_generation()


    def generation_plot(self, gen=-1):
        scores = self.fitness[gen]
        best   = scores[:self.mating_pop]
        n_gen  = len(self.population)
        y_line = self.full_perturbation_err

        generation_plot(scores, best, n_gen, y_line)
    

    def search(self, early_stopping=True, verbose=False, print_plots = False):

        self.compute_max_err.reset()

        start_t = time.time()
        self.init_population()
        total_t = time.time()-start_t

        if verbose:
            aux = self.fitness[0][0]
            print(f"Initialized 1st generation ({total_t:.3f}s)-- ",
                  f"max error = {aux[1]:.3e}, ",
                  f"ulp calls = {aux[0]}")

        if print_plots:
            self.generation_plot()

        counter = 0
        j       = 1
        # pop_set = len(set(self.fitness[-1]))

        while (self.compute_max_err.call_count <= self.c 
               and j<self.n_generations
               and counter<self.stop_counter):

            start_t = time.time()
            self.evolve_generation()
            total_t = time.time()-start_t

            n_calls, err         = self.fitness[-1][0]
            prev_calls, prev_err = self.history[-2]
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

        max_err = [0] + [abs(err) for _, err in self.history]
        idx_aux = self.ulp_calls
        y_max   = torch.zeros(self.ulp_calls[-1])

        for k in range(self.n_generations):
            y_max[idx_aux[k]: idx_aux[k+1]] = max_err[k]

        return y_max.unsqueeze(0)

    def release(self):
        """Explicitly drop this instance's large in-memory state instead of
        waiting on refcounting / the cyclic GC to reclaim it. Call this
        (and then `del` the instance) once you're done with a search() run,
        especially if you're creating many instances in a loop (e.g. one
        per sample in a benchmarking script)."""
        self.population.clear()
        self.fitness.clear()
        self.history.clear()

        # Drop the GPU copy of the weights uploaded in AdvPerturbation's
        # __init__, if present, and let PyTorch's caching allocator reclaim
        # the underlying CUDA blocks.
        if getattr(self, "weights_gpu", None) is not None:
            del self.weights_gpu
            self.weights_gpu = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        gc.collect()



if __name__ == "__main__":

    n_test = int(sys.argv[1])   # number of repeated runs per (pop_size, p) configuration
    data   = sys.argv[2]

    model_name = "ResNet"
    seed       = 420
    dtype      = torch.bfloat16 if data.upper().startswith("BF") else torch.float32

    random.seed(seed)
    torch.manual_seed(seed)
    # ------------------------------------------------------------------
    # EXPERIMENT INPUTS
    # ------------------------------------------------------------------
    if model_name.upper().startswith("EFF"):
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).eval()
        W     = torch.transpose(model.classifier[1].weight.data, 0, 1).to(dtype)
    else:
        model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1).eval()
        W     = torch.transpose(model.fc.weight.data, 0, 1).to(dtype)
                
    n_latent = W.shape[0]

    W0    = torch.randn(n_latent, n_latent,
                        dtype=dtype)  
    X     = torch.randn(n_test, n_latent, n_latent,
                        dtype=dtype)
    X_img = torch.randn(n_test, 1, 3, 224, 224,
                        dtype=dtype)

    MAX_CALLS      = 32 # if data.upper().startswith("BF") else 128
    c              = 1000  # total budget of calls to compute_max_err
    early_stopping = (not data.upper().startswith("BF")) # Deactivate early stopping for bf16

    # ------------------------------------------------------------------
    # Grid search over population size and perturbation fraction
    # ------------------------------------------------------------------
    pop_sizes = [20, 50, 100]
    q_values  = [0.05, 0.1]
        
    targets = []
    print(f"Computing target errors of the sample ({dtype})")
    for _x in X:
        targ_aux = AdvPerturbation(_x, W, c, max_calls=MAX_CALLS)
        err = targ_aux.full_perturbation_err
        targets.append(err)

    results = []
    fname   = f"grid_search_{model_name}_{data}_{seed}.pkl"

    for pop_size, q in itertools.product(pop_sizes, q_values):
        print(f"Running test for pop_size={pop_size}, q={q:.2f}")

        # n_generations must satisfy: n_generations * pop_size <= c
        n_generations = max(1, c // pop_size)

        run_errs      = []
        run_calls     = []
        run_times     = []
        run_gens      = []
        run_err_ratio = []

        for trial in range(n_test):
            X_test     = X[trial]
            target_err = targets[trial]

            print(f"{trial+1} - Full matrix perturbation = {target_err:.4e}")

            ga = AdversarialGeneticAlgorithm(
                input_matrix=X_test,
                func=W,
                c=c,
                q=q,
                max_calls=MAX_CALLS, 
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

            print(# f"pop_size={pop_size:>4} | q={q:>5.2f} |"
                  f" trial={trial+1}/{n_test} | gens={ga.n_generations:>3} | "
                  f"best_err={abs(best_err):.4e} ({100*err_pct:.2f}%) | "
                  f"ulp_calls={best_calls:>6} | "
                  f"n calls = {ga.ulp_calls[-1]} ({100*(ga.ulp_calls[-1]/c):.2f}% of call budget)  in {elapsed:.2f}s")

        run_errs      = np.array(run_errs)
        run_err_ratio = np.array(run_err_ratio)
        success_pct   = (run_err_ratio >= 1).sum()/n_test

        results.append({
            "pop_size":       pop_size,
            "q":              q,
            "n_test":         n_test,
            "target_err":     np.array(targets),
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

        print(f"  -> mean err% ={run_err_ratio.mean():.4e} (std={run_errs.std():.4e}) "
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

    # # ------------------------------------------------------------------
    # # Heatmap of mean best error over the (pop_size, q) grid
    # # ------------------------------------------------------------------
    # err_grid = np.array([r["mean_err_pct"] for r in results]).reshape(
    #     len(pop_sizes), len(q_values)
    # )

    # plt.figure()
    # plt.imshow(err_grid, aspect="auto", origin="lower")
    # plt.colorbar(label=f"Mean best max error (n_test={n_test})")
    # plt.xticks(range(len(q_values)), q_values)
    # plt.yticks(range(len(pop_sizes)), pop_sizes)
    # plt.xlabel("q (perturbation %)")
    # plt.ylabel("Population Size (P)")
    # plt.title("Grid search: pop_size vs q")
    # plt.tight_layout()
    # plt.show()


    # succcess_grid = np.array([r["success_pct"] for r in results]).reshape(
    #         len(pop_sizes), len(q_values)
    #     )

    # plt.figure()
    # plt.imshow(succcess_grid, aspect="auto", origin="lower")
    # plt.colorbar(label=f"Success percentage (n_test={n_test})")
    # plt.xticks(range(len(q_values)), q_values)
    # plt.yticks(range(len(pop_sizes)), pop_sizes)
    # plt.xlabel("q (perturbation %)")
    # plt.ylabel("Population Size (P)")
    # plt.title("Grid search: pop_size vs q")
    # plt.tight_layout()
    # plt.show()
