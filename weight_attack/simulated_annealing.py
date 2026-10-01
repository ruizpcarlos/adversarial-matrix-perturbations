import numpy as np
from tqdm import tqdm

from adv_layer import AdvLayerPerturbation


class SimulatedAnnealingLayerSearch(AdvLayerPerturbation):
    """
    Batch handling, scoring and fitness (A -> best_calls -> ULPs -> B) live in
    AdvLayerPerturbation: resample(), draw_test_batch(), fitness(), test_score().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.T0 = None
        self.solution = None
        self.log = None

        self.ulp_calls  = [0]    # cumulative n_evals after each generation
        

    # ------------------------------------------------------------------
    # SA pieces
    # ------------------------------------------------------------------
    # def neighbor(self, idx):
    #     return self.mutate_geneset(idx)                    # uses self.mutation_rate

    def init_temp(self, target_prob=0.9, n_samples=50, verbose=False):
        if verbose:
            print(f"Sampling {n_samples} moves to set initial temp")
        if self.batch_A is None:
            self.resample()
        cur = self.random_geneset()
        cur_fit, _, _ = self.fitness(cur)
        deltas = []

        for _ in tqdm(range(n_samples), desc="Sampling moves", disable=not verbose):
            cand = self.mutate_geneset(cur)
            fit, _, _ = self.fitness(cand)
            deltas.append(fit - cur_fit)
            cur, cur_fit = cand, fit                       # random walk
        worse = np.array([d for d in deltas if d < 0])
        self.T0 = worse.mean() / np.log(target_prob) if worse.size else 1e-2
        self.n_evals = 0                                   # calibration is not charged

    def search(self, L0=5, alpha=0.85, beta=1.03,
               T_min=1e-8,
               patience=25,
               resample_every=None,
               test_batch_size=1024,
               verbose=False):

        self.resample()
        self.draw_test_batch(test_batch_size)              # fixed, never used for decisions
        if self.T0 is None:
            self.init_temp(verbose=verbose)
        T = self.T0
        self.n_evals = 0

        cur = self.random_geneset()
        cur_fit, cur_n, _ = self.fitness(cur)
        best_idx, best_n, best_fit = cur, cur_n, cur_fit

        log = {k: [] for k in ("T", "evals", "cur_B", "best_B", "best_C")}

        def record():
            test = self.test_score(best_idx, best_n)
            for k, v in zip(log, (T, self.n_evals, cur_fit, best_fit, test)):
                log[k].append(v)

        record()
        k, stale, L = 0, 0, L0

        while T > T_min and stale < patience and self.n_evals < self.budget_calls:
            # Fitness values are only comparable within one (A, B) pair,
            # so after a resample, re-anchor cur and best.
            if resample_every and k > 0 and k % resample_every == 0:
                self.resample()
                cur_fit, cur_n, _ = self.fitness(cur)
                best_fit, best_n, _ = self.fitness(best_idx)  # evaluate re-runs: best_n may change

            improved = False
            for _ in range(L):
                if self.n_evals >= self.budget_calls:
                    break
                cand = self.mutate_geneset(cur)
                fit, n_ulp, _ = self.fitness(cand)
                d = fit - cur_fit
                if d >= 0 or self.rng.random() < np.exp(d / T):
                    cur, cur_fit, cur_n = cand, fit, n_ulp
                if fit > best_fit:
                    best_idx, best_n, best_fit = cand, n_ulp, fit
                    improved = True

            stale = 0 if improved else stale + 1
            k += 1
            T *= alpha
            L = int(L0 * beta ** k)
            if verbose and k % 10 == 0:
                print(f"k={k} T={T:.2e} best={best_fit:.4f} n_ulp={best_n} evals={self.n_evals}")

            record()

        self.solution = (best_idx, best_n)
        self.log = log
        return best_fit, self.solution

    def final_score(self, batch_size=1024):
        """Estimate on a fresh, larger batch that never influenced the search."""
        idx, n_ulp = self.solution
        return self.test_score(idx, n_ulp, batch_size=batch_size, fresh=True)

    def compute_adv_weights(self):
        idx, n_ulp = self.solution
        return self.apply_ulps(idx, n_ulp)
