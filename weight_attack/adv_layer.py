# %%
import copy
import random
import pickle
import gc

import torch
import numpy as np
from datasets import load_dataset

# import torchvision.models as models
# from huggingface_hub import login
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable

# from functools import cached_property, update_wrapper
from utils.utils import vector_distance, wrap_score


class AdvLayerPerturbation:

    def __init__(self,
                 model: nn.Module,
                 model_name: str,
                 q:float,
                 max_calls:int = 32,
                 budget_calls: int = 1_000,
                 objective_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
                 seed: int = 420,
                 weighted_sampling: bool = False,
                 mutation_rate:float = 0.01,
                 batch_size:int = 256):

        # self.dataset     = dataset

        fe, clf    = self.split_model(model, model_name) # SEPARATE MODEL INTO FEATURE EXTRACTION AND CLASSIFIER
        lin        = clf[1] if isinstance(clf, nn.Sequential) else clf   # EfficientNet vs ResNet

        self._extractor = fe.eval()
        self.W_cpu     = lin.weight.detach().clone()
        self.b_cpu     = lin.bias.detach().clone()
        self.W_gpu     = self.W_cpu.to("cuda")
        self.b_gpu     = self.b_cpu.to("cuda")

        self.features = None

        if not 0 < q < 1:
            raise ValueError(f"q must be in [0, 1], got {q}")
        self.q  = q
        self.n_perturbed = int(q*self.W_cpu.numel())

        self.INFTY = torch.tensor(torch.inf)

        self.n_classes, self.n_features = self.W_cpu.shape

        # self.n_data = self.dataset.shape[0]
        self.total  = int(self.W_cpu.numel())

        self.objective_fn = (
                        objective_fn if objective_fn is not None
                        else vector_distance
                        )

        self.budget_calls = budget_calls # Controls the number of calls to compute_max_err
        self.max_calls   = max_calls
        self.total_calls = self.budget_calls*max_calls

        self.rng       = np.random.default_rng(seed)
        self.batch_rng = np.random.default_rng(seed + 1) 

        self.weighted_sampling = weighted_sampling
        self.sample_weights    = None
        self._cdf              = None
        if weighted_sampling:
            self._build_sampling_weights(eps = 0.75)

        self.batch_size = batch_size
        self.mutation_rate = mutation_rate
        self.pools    = None

        # Batch snapshots (h_cpu, h_gpu, e0): A = search, B = fitness, C = held-out test
        self.features_val = None
        self.batch_A = self.batch_B = self.batch_C = None
        self.n_evals = 0          # number of fitness() calls (budget unit for every algorithm)


    def split_model(self, model:nn.Module, name: str):
        
        if name.upper().startswith("EFF"):
            fe = nn.Sequential(model.features, model.avgpool, nn.Flatten(1))
            clf = model.classifier
        else:
            fe = nn.Sequential(*list(model.children())[:-1], nn.Flatten(1))
            clf = model.fc
        return fe, clf


    @torch.inference_mode()
    def _extract(self, source, batch_size, device):
        if torch.is_tensor(source):
            batches = (source[i:i + batch_size] for i in range(0, len(source), batch_size))
        else:  # DataLoader yielding (x, y) or x
            batches = (b[0] if isinstance(b, (list, tuple)) else b for b in source)
        return torch.cat([self._extractor(x.to(device)).cpu() for x in batches])


    def extract_features(self, train, val=None, batch_size=256, device="cuda"):
        self._extractor.to(device)
        self.features = self._extract(train, batch_size, device)
        self.n_data   = self.features.shape[0]
        if val is not None:
            self.features_val = self._extract(val, batch_size, device)
            self.n_val        = self.features_val.shape[0]
        self._extractor = None          # drop the backbone only after both splits are done
        self._flush()
        return self.features, self.features_val

    def set_features(self, train, val=None):
        """
        Helper to set features if they have been precomputed and read from cache.
        """
        self.features, self.features_val = train, val
        self.n_data = train.shape[0]
        self.n_val = None if val is None else val.shape[0]
        self._extractor = None


    @staticmethod
    def _flush():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        
    ############################################################
    #        CANDIDATE DRAWING
    ###########################################################

    # def _build_sampling_weights(self, alpha: float=0.1, eps:float= 0.1) -> torch.Tensor:
    #     score = wrap_score(self.W_cpu.reshape(-1),
    #                     alpha=alpha,
    #                        max_calls=self.max_calls).double()
    #     n, total = score.numel(), score.sum()
    #     uniform = torch.full_like(score, 1.0 / n)
    #     if total <= 0:                       # nothing can wrap: fall back to uniform
    #         return uniform
    #     # Mixture: with prob (1-eps) follow the score, with prob eps sample uniformly
    #     return (1 - eps) * score / total + eps * uniform

    # def _draw_flat(self, num_samples: int, weighted=None):
    #     weighted = self.weighted_sampling if weighted is None else weighted
    #     if not weighted or self.sample_weights is None:
    #         return torch.randperm(self.total)[:num_samples]
    #     return torch.multinomial(self.sample_weights, num_samples, replacement=False)


    # def _sample_entries(self, num_samples=1, weighted=None):
    #     flat_idx = self._draw_flat(num_samples, weighted)
    #     return (flat_idx // self.n_classes, flat_idx % self.n_classes)

    def _build_sampling_weights(self, alpha=0.1, eps=0.1):
        score = wrap_score(self.W_cpu.reshape(-1),
                           alpha=alpha,
                           max_calls=self.max_calls).double()
        n, score_sum = score.numel(), score.sum()
        if score_sum <= 0:
            w = torch.full_like(score, 1.0 / n)
        else:
            w = (1 - eps) * score / score_sum + eps / n         # mixture with uniform
        self.sample_weights = w
        cdf = np.cumsum(w.numpy())
        cdf /= cdf[-1]                                       # guard against rounding drift
        self._cdf = cdf                                      # built once, reused by every draw

    #######################################################
    #       CANDIDATE SCORING
    #######################################################
    def split_pools(self, fracs=(0.5, 0.5)):
        """Partition the TRAIN features into disjoint pools A (search) and B (fitness)."""
        assert self.features is not None, "No features have been extracted"
        assert len(fracs) == 2 and abs(sum(fracs) - 1) < 1e-9, "fracs must have 2 entries summing to 1"

        perm = self.batch_rng.permutation(self.n_data)
        n_a  = int(fracs[0] * self.n_data)
        self.pools = {"A": perm[:n_a], "B": perm[n_a:]}
        assert not (set(self.pools["A"].tolist()) & set(self.pools["B"].tolist()))


    @torch.inference_mode()
    def set_batch(self, pool, batch_size=None):
        """
        pool: "A" | "B" -> drawn from disjoint partitions of the train features
            "C"       -> drawn from the validation features
        Returns the (h_cpu, h_gpu, e0) snapshot and loads it into self.
        """
        if pool == "C":
            assert self.features_val is not None, "No validation features: pass val= to extract_features"
            src, candidates = self.features_val, np.arange(self.n_val)
        elif pool in ("A", "B"):
            assert self.features is not None, "No features have been extracted"
            if self.pools is None:
                self.split_pools()
            src, candidates = self.features, self.pools[pool]
        else:
            raise ValueError(f"unknown pool {pool!r}")

        bs  = min(self.batch_size if batch_size is None else batch_size, candidates.size)
        sel = self.batch_rng.choice(candidates, size=bs, replace=False)
        self.batch_idx = sel

        self.h_cpu = src[torch.from_numpy(sel)]
        self.h_gpu = self.h_cpu.to("cuda")
        y_c = F.linear(self.h_cpu, self.W_cpu, self.b_cpu)
        y_g = F.linear(self.h_gpu, self.W_gpu, self.b_gpu).cpu()
        self.e0 = (y_c - y_g).abs().amax(dim=1)
        return (self.h_cpu, self.h_gpu, self.e0)


    ############################################################
    #   BATCH HANDLING (shared by every search algorithm)
    ############################################################
    # A batch is a snapshot (h_cpu, h_gpu, e0). e0 is batch-specific, so a score is
    # only meaningful on the batch it was computed with.
    #   A: search batch  (feeds evaluate -> best_calls)
    #   B: fitness batch (scores the candidate after best_calls ULPs)
    #   C: held-out test batch (never used for decisions)
    def load_batch(self, batch):
        self.h_cpu, self.h_gpu, self.e0 = batch

    def resample(self):
        """Draw a fresh (A, B) pair from the disjoint train pools."""
        self.batch_A = self.set_batch("A")
        self.batch_B = self.set_batch("B")

    def draw_test_batch(self, batch_size=None):
        """Draw the fixed held-out batch C (call once; resample() does not touch it)."""
        self.batch_C = self.set_batch("C", batch_size)
        return self.batch_C

    @torch.inference_mode()
    def apply_ulps(self, idx, n_ulp):
        """Perturbed CPU copy of W with n_ulp nextafter steps applied at idx."""
        W    = self.W_cpu.clone()
        flat = W.view(-1)
        i    = torch.from_numpy(np.asarray(idx, dtype=np.int64))
        v    = flat[i]
        for _ in range(n_ulp):
            v = torch.nextafter(v, self.INFTY)
        flat[i] = v
        return W

    @torch.inference_mode()
    def score_fixed(self, idx, n_ulp, agg=None):
        """Apply exactly n_ulp nextafter steps to idx; score on the currently loaded batch."""
        agg = agg or self.default_agg
        W_c = self.apply_ulps(idx, n_ulp)
        W_g = W_c.to("cuda")
        y_g = F.linear(self.h_gpu, W_g, self.b_gpu)
        y_c = F.linear(self.h_cpu, W_c, self.b_cpu)
        e   = (y_c - y_g.cpu()).abs().amax(dim=1)
        return agg(e, self.e0)

    def fitness(self, idx, agg=None):
        """
        1. evaluate() on batch A -> best_calls
        2. apply best_calls ULP steps
        3. score on batch B      -> fitness
        Returns (fitness, best_calls, score_A). Counts one eval toward the budget.
        """
        if self.batch_A is None or self.batch_B is None:
            self.resample()
        self.load_batch(self.batch_A)
        n_ulp, score_A = self.evaluate(idx, agg)
        self.load_batch(self.batch_B)
        fit = self.score_fixed(idx, n_ulp, agg)
        self.n_evals += 1
        return fit, n_ulp, score_A

    @torch.inference_mode()
    def test_score(self, idx, n_ulp, batch_size=None, fresh=False, agg=None):
        """
        Score on held-out data. Default: the fixed batch C (drawn on first use).
        fresh=True draws a new, unstored C batch (e.g. a larger one for a final estimate).
        """
        if fresh:
            batch = self.set_batch("C", batch_size)
        else:
            if self.batch_C is None:
                self.draw_test_batch(batch_size)
            batch = self.batch_C
        self.load_batch(batch)
        return self.score_fixed(idx, n_ulp, agg)

    @staticmethod
    def default_agg(e, e0):
        frac = (e > e0).float().mean()
        gain = ((e - e0).mean() / (e0.mean() + 1e-30))
        return frac.item() + 1e-3 * gain.item()


    @torch.inference_mode()
    def evaluate(self, idx, agg=None):
        agg = agg or self.default_agg

        flat_idx_c = torch.from_numpy(idx)                 # int64
        flat_idx_g = flat_idx_c.to("cuda")
        W_c, W_g   = self.W_cpu.clone(), self.W_gpu.clone()
        Wc, Wg = W_c.view(-1), W_g.view(-1)             # views onto the same storage
        v = Wc[flat_idx_c]

        best_score, best_calls = -float("inf"), 0

        for n in range(1, self.max_calls):
            v = torch.nextafter(v, self.INFTY)
            Wc[flat_idx_c] = v
            Wg[flat_idx_g] = v.to("cuda")

            y_g = F.linear(self.h_gpu, W_g, self.b_gpu)    # launches asynchronously
            y_c = F.linear(self.h_cpu, W_c, self.b_cpu)    # overlaps with the GPU work
            e   = (y_c - y_g.cpu()).abs().amax(dim=1)      # (N,), syncs here

            score = agg(e, self.e0)
            if score > best_score:
                best_score, best_calls = score, n

        return best_calls, best_score


    ##################################################################
    #                 MUTATION FUNCTIONS
    ##################################################################
    @staticmethod
    def _not_in_sorted(cand, sorted_arr):
        """Boolean mask: which cand are NOT in sorted_arr (binary search, O(m log k))."""
        if sorted_arr is None or sorted_arr.size == 0:
            return np.ones(cand.shape, dtype=bool)
        pos = np.searchsorted(sorted_arr, cand)
        pos[pos == sorted_arr.size] = sorted_arr.size - 1
        return sorted_arr[pos] != cand


    def _draw_distinct(self, n, exclude=None, weighted=None):
        """n distinct flat indices, none in `exclude` (sorted array). Returns sorted int64.
        Drawing with replacement and keeping first occurrences is exactly successive
        sampling without replacement, so the weighted case matches the old multinomial."""
        weighted = self.weighted_sampling if weighted is None else weighted
        use_w    = weighted and self._cdf is not None
        out      = np.empty(0, dtype=np.int64)

        while out.size < n:
            need = n - out.size
            m    = int(need * 1.25) + 8
            if use_w:
                cand = np.searchsorted(self._cdf, self.rng.random(m), side="right")
                np.clip(cand, 0, self.total - 1, out=cand)
            else:
                cand = self.rng.integers(0, self.total, size=m)
            cand = cand.astype(np.int64)

            _, first = np.unique(cand, return_index=True)    # dedupe, keep order of appearance
            cand = cand[np.sort(first)]
            cand = cand[self._not_in_sorted(cand, exclude)]
            if out.size:
                cand = cand[~np.isin(cand, out)]
            out = np.concatenate([out, cand[:need]])

        return np.sort(out)


    def _sample_entries(self, num_samples=1, weighted=None):
        return self._draw_distinct(num_samples, weighted=weighted)   # flat indices only


    def random_geneset(self, weighted=None):
        return self._draw_distinct(self.n_perturbed, weighted=weighted)


    # ---- genetic operators ----
    def mutate_geneset(self, genome, n_mutations=None):
        """Replace n_mutations entries. Default: mutation_rate * k (at least 1)."""
        genome = np.array(genome, dtype=np.int64)            # copy: never mutate the parent
        k = genome.size
        n = max(1, int(round(self.mutation_rate * k))) if n_mutations is None else n_mutations
        n = min(n, k)

        clear_pos = self.rng.choice(k, size=n, replace=False)
        new_vals  = self._draw_distinct(n, exclude=genome)   # new entries are not already active
        genome[clear_pos] = new_vals
        genome.sort()
        return genome


    def crossover_uniform(self, g1, g2):
        """g1, g2: sorted, unique, same length. Swaps half of the differing entries."""
        one_zero = np.setdiff1d(g1, g2, assume_unique=True)  # active in g1 only
        zero_one = np.setdiff1d(g2, g1, assume_unique=True)  # active in g2 only
        n_swap   = min(one_zero.size, zero_one.size) // 2

        to_s1 = self.rng.choice(zero_one, size=n_swap, replace=False)
        to_s2 = self.rng.choice(one_zero, size=n_swap, replace=False)

        new1 = np.union1d(np.setdiff1d(g1, to_s2, assume_unique=True), to_s1)
        new2 = np.union1d(np.setdiff1d(g2, to_s1, assume_unique=True), to_s2)
        return new1, new2                                    # union1d output is sorted and unique


    def recombine(self, g1, g2, mutation_rate=None):
        o1, o2 = self.crossover_uniform(g1, g2)
        n1 = None if mutation_rate is None else max(1, int(round(mutation_rate * o1.size)))
        return self.mutate_geneset(o1, n1), self.mutate_geneset(o2, n1)