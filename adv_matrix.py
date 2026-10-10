import torch
import random
import copy
# import threading
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from typing import Union, Optional, Tuple, Callable
from functools import cached_property

from utils.objective_func import vector_distance
from utils.utils import CallTracker, product_err, wrap_score
from utils.plotting_utils import plot_max

FuncType = Union[torch.Tensor, nn.Module]

@CallTracker
def _nextafter(input, other):
    return torch.nextafter(input, other)

# @CallTracker
# def nextafter(x: torch.Tensor, offset: torch.Tensor):
#     MAX_EXPONENT = 0xFF
#     MAX_MANTISSA = 0x7FFFFF

#     if not torch.is_tensor(x):
#         x = torch.tensor(x, dtype=torch.float32)

#     if not torch.is_tensor(offset):
#         offset = torch.tensor(offset, dtype=torch.int32)

#     x, offset = torch.broadcast_tensors(torch.clone(x), torch.clone(offset))

#     assert not torch.is_floating_point(offset)
#     assert x.dtype == torch.float32
#     assert (-MAX_MANTISSA <= offset).all() and (offset <= MAX_MANTISSA).all()

#     x_int = x.view(torch.int32)

#     # Extract the sign (1 bit), exponent (8 bits), and mantissa (23 bits)
#     sign = (x_int >> 31) & 1
#     exponent = (x_int >> 23) & MAX_EXPONENT
#     mantissa = x_int & MAX_MANTISSA

#     # If
#     #   "sign is positive and offset is positive"
#     # or
#     #   "sign is negative and offset is negative"
#     # treat the calculation as an addition that can overflow. Otherwise, it's
#     # a substraction that can underflow.
#     sign_is_positive = sign == 0
#     offset = torch.where(sign_is_positive, offset, -offset)

#     underflow = (mantissa + offset) < 0
#     overflow = (mantissa + offset) > MAX_MANTISSA

#     zero_pass = exponent == 0
#     to_inf = exponent == MAX_EXPONENT

#     # OVERFLOW
#     # Handle "regular" overflow. We add the mantissa and the offset modulo the
#     # max mantissa value and increase the exponent by one
#     mantissa[overflow] = (mantissa[overflow] + offset[overflow]) & MAX_MANTISSA
#     exponent[overflow & ~to_inf] += 1

#     # If the exponent was already maximal, we set the mantissa to zero
#     # (max exponent + non-zero mantissa is a NaN)
#     # This is different from how torch.nextafter handles it. They return a NaN
#     # value for some reason.
#     mantissa[overflow & to_inf] = 0

#     # UNDERFLOW
#     # Handle "regular" underflow.
#     underflow_r = underflow & ~zero_pass
#     underflow_z = underflow & zero_pass

#     mantissa[underflow_r] = (mantissa[underflow_r] + offset[underflow_r]) & MAX_MANTISSA
#     exponent[underflow_r] -= 1

#     # Underflow past zero
#     mantissa[underflow_z] = -offset[underflow_z] - mantissa[underflow_z]
#     sign[underflow_z] = 1 - sign[underflow_z]

#     mantissa[~overflow & ~underflow] = (
#         mantissa[~overflow & ~underflow] + offset[~overflow & ~underflow]
#     )

#     sign = sign & 0x1
#     exponent = exponent & MAX_EXPONENT
#     mantissa = mantissa & MAX_MANTISSA

#     return ((sign << 31) | (exponent << 23) | mantissa).view(torch.float32)


class AdvPerturbation:

    def __init__(self, 
                 input_matrix: torch.Tensor, 
                 func: FuncType, 
                 q:float,
                 func_gpu: Optional[FuncType] = None,
                 max_calls:int = 32,
                 budget_calls: int = 1_000,
                 objective_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 mutation_rate:float = 0.01,
                 weighted_sampling: bool = False,
                 seed:int = 420):

        self.input_matrix = input_matrix

        if isinstance(func, torch.Tensor):
            self.tensor_prod = True
            # Dimension check using consecutive pairs
            # _matrices = [input_matrix] + func
            # for a, b in zip(_matrices, _matrices[1:]):
            if input_matrix.shape[-1] != func.shape[-2]:
                raise ValueError(
                    f"Shape mismatch: {tuple(input_matrix.shape)} vs {tuple(func.shape)} — "
                    f"dim {input_matrix.shape[-1]} != {func.shape[-2]}"
                    )
            self.weights     = [func]
            self.weights_gpu = [func_gpu] if func_gpu is not None else [m.to("cuda") for m in self.weights]      
            self.nn          = None
            self.nn_gpu      = None
        elif isinstance(func, nn.Module):
            self.tensor_prod = False
            try:
                with torch.no_grad():
                    func.eval()
                    func(input_matrix)
            except Exception as e:
                raise ValueError(f"Input tensor is not a valid input for func: {e}")
            self.weights     = None
            self.weights_gpu = None
            self.nn          = func
            self.nn_gpu      = copy.deepcopy(self.nn).eval().to("cuda")        
        else:
            raise TypeError(f"Received a {type(func)} as func: must be either torch.Tensor or nn.Module.")

        if not 0 < q < 1:
            raise ValueError(f"q must be in [0, 1], got {q}")
        self.q  = q
        self.n_perturbed = int(q*input_matrix.numel())
                                    
        self.INFTY = torch.tensor(torch.inf)
        
        self.input_shape    = input_matrix.shape
        # self.strides        = self._strides(self.input_shape)
        self.total          = int(input_matrix.numel())
        self.mutation_rate = mutation_rate
            
        if self.tensor_prod: # Input is a matrix   
            self.n_input  = input_matrix.shape[0]
            self.n_latent = input_matrix.shape[1]
        else: # Input is "image-like": (1, C, H, W)
            self.n_input  = input_matrix.shape[2] # Height
            self.n_latent = input_matrix.shape[3] # Width

        self.objective_fn = (
                        objective_fn if objective_fn is not None
                        else vector_distance
                        )
        
        self.budget_calls = budget_calls # Controls the number of calls to compute_max_err
        self.max_calls   = max_calls
        self.total_calls = self.budget_calls*max_calls


        self.rng  = np.random.default_rng(seed)
        self._cdf = None
        self._w = None
        if weighted_sampling:
            self._build_sampling_weights()
        

        # Wrap the function to count calls
        self.compute_max_err = CallTracker(self.compute_max_err)


    @cached_property
    def baseline_err(self):
        """
        Calculates the error  
        d (y_gpu, y_cpu) for the non perturbed matrix
        """
        _, err = self.compute_max_err(ulp_calls=0)
        self.compute_max_err.reset()
        return err
        
    
    # def flat_to_3d(self, idx):
    #     aux_idx = idx % (self.n_latent**2)
    #     c = idx // (self.n_latent**2)
    #     h = aux_idx // self.n_latent
    #     w = aux_idx % self.n_latent
    #     return c, h, w

    # def _build_sample_weights(self, alpha: float=0.1, eps:float= 0.1) -> torch.Tensor:
    #     score = wrap_score(self.input_matrix.reshape(-1),
    #                     alpha=alpha, max_calls=self.max_calls).double()
    #     n, total = score.numel(), score.sum()
    #     uniform = torch.full_like(score, 1.0 / n)

    #     if total <= 0:                       # nothing can wrap: fall back to uniform
    #         return uniform
    #     # Mixture: with prob (1-eps) follow the score, with prob eps sample uniformly
    #     return (1 - eps) * score / total + eps * uniform

    def _build_sampling_weights(self, alpha=0.1, eps=0.1):
        score = wrap_score(self.input_matrix.reshape(-1), alpha=alpha,
                        max_calls=self.max_calls).double()
        n, score_sum = score.numel(), score.sum()
        w = (torch.full_like(score, 1.0 / n) if score_sum <= 0
            else (1 - eps) * score / score_sum + eps / n)
        self.sample_weights = w
        self._w   = w.numpy().copy()          # pristine copy, never mutated
        cdf = np.cumsum(self._w)
        cdf /= cdf[-1]
        self._cdf = cdf


    # def _draw_flat(self, num_samples: int, weighted=None):
    #     weighted = self.weighted_sampling if weighted is None else weighted
    #     if not weighted or self.sample_weights is None:
    #         return torch.randperm(self.total)[:num_samples]
    #     return torch.multinomial(self.sample_weights, num_samples, replacement=False)


    def _draw_distinct(self, n, exclude=None, weighted=None, cdf=None):
        """n distinct flat indices, none in `exclude` (sorted array). Returns sorted int64.
        Drawing with replacement and keeping first occurrences is exactly successive
        sampling without replacement, so the weighted case matches the old multinomial."""
        
        weighted = self.weighted_sampling if weighted is None else weighted
        cdf      = self._cdf if cdf is None else cdf
        use_w    = weighted and cdf is not None
        out      = np.empty(0, dtype=np.int64)
        
        while out.size < n:
            need = n - out.size
            m    = int(need * 1.25) + 8
            if use_w:
                cand = np.searchsorted(cdf, self.rng.random(m), side="right")
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
    

    # def _sample_entries(self, num_samples=1, weighted=None):
    #     flat_idx = self._draw_flat(num_samples, weighted)

    #     if not self.tensor_prod:
    #         indices = self.flat_to_3d(flat_idx)
    #         return (torch.zeros(num_samples, dtype=int), *indices)
    #     else:
    #         return (flat_idx // self.n_latent, flat_idx % self.n_latent)
    
    def _sample_entries(self, num_samples=1, weighted=None, cdf=None):
        return self._draw_distinct(num_samples, weighted=weighted, cdf=cdf)

    def _product_err(self, mat_cpu, mat_gpu):
        err = product_err(mat_cpu, mat_gpu, self.objective_fn)
        return err.item()
    
    @torch.inference_mode()
    def model_err(self, x_cpu, x_gpu):
        y_cpu  = self.nn(x_cpu)
        y_gpu  = self.nn_gpu(x_gpu)

        d = self.objective_fn(y_cpu, y_gpu.cpu())
        
        return d.item()


    def random_perturbation(self, 
                            step=1, 
                            weighted=False, 
                            early_stopping = True, 
                            verbose=False):

        if weighted and not self.weighted_sampling:
            raise ValueError(f"Passed weighted={weighted}  but there are no weights available for sampling")

        cdf = self._cdf if weighted else None
        w   = self._w.copy() if weighted else None
        
        _nextafter.reset()

        X_    = self.input_matrix.clone()
        X_gpu = X_.to("cuda")
        X_flat = X_.view(-1)

        if self.tensor_prod:
            mat_cpu = [None] + self.weights
            mat_gpu = [None] + self.weights_gpu

        n_iter     = self.total_calls // step
        n_stopping = int(n_iter/2)
        
        y       = np.zeros(n_iter)
        counts  = {}
        maxed   = set()
        max_err = -1
        stale   = 0      
        n_done  = n_iter
        exhausted = False

        iterator = tqdm(range(n_iter)) if verbose else range(n_iter)

        for i in iterator:

            idx = self._sample_entries(weighted=weighted, cdf=cdf)
            key = (int(idx[0]),)
            if not weighted and len(maxed) >= self.total:   # unweighted guard
                n_done = i
                break

            counts[key] = counts.get(key, 0) + step
            if counts[key] >= self.max_calls:
                maxed.add(key)
                if weighted:
                    w[idx[0]] = 0.0
                    if w.sum() <= 0:
                        exhausted = True                     # nothing left to sample
                    else:
                        cdf = np.cumsum(w)
                        cdf /= cdf[-1]

            for _ in range(step):
                X_flat[idx] = _nextafter(X_flat[idx], self.INFTY)
            
            X_gpu.copy_(X_, non_blocking=True)

            if self.tensor_prod:
                mat_cpu[0] = X_
                mat_gpu[0] = X_gpu
                _y = product_err(mat_cpu, mat_gpu)
            else:
                _y = self.model_err(X_, X_gpu)

            y[i] = _y

            if _y > max_err:
                max_err  = _y
                max_pert = {k: v for k, v in counts.items()}
                stale    = 0
            else:
                stale += 1

            if exhausted:
                n_done = i + 1
                break

            # NEW: early stopping
            if early_stopping and stale >= n_stopping:
                n_done = i + 1
                if verbose:
                    print(f"Early stop at iteration {n_done}/{n_iter}: "
                        f"no new max in {n_stopping} iterations "
                        f"(max err = {max_err:.4e})")
                break

        y = y[:n_done]  # NEW: drop the unused tail of the preallocated array

        return torch.Tensor(y).unsqueeze(0), max_pert
    
    
    def compute_max_err(self, 
                        ulp_calls: Optional[int] = None,
                        # indices: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[int, float]:
                        indices: Optional[np.array] = None) -> Tuple[int, float]:

        X_    = self.input_matrix.clone()
        X_gpu = X_.to("cuda")

        if indices is not None:
            idx = torch.from_numpy(indices)  # int64
            X_flat = X_.view(-1)

        if self.tensor_prod:
            mat_cpu = [X_] + self.weights
            mat_gpu = [X_gpu] + self.weights_gpu
            max_err = self._product_err(mat_cpu, mat_gpu)
        else:
            max_err = self.model_err(X_, X_gpu)
                
        calls_to_max = 0
        ulp_calls = self.max_calls-1 if ulp_calls is None else ulp_calls
        
        for i in range(ulp_calls):

            if indices is not None:
                X_flat[idx] = _nextafter(X_flat[idx], self.INFTY)
            else:
                X_.copy_(_nextafter(X_, self.INFTY))

            X_gpu.copy_(X_, non_blocking=True)

            if self.tensor_prod:
                mat_cpu[0] = X_
                mat_gpu[0] = X_gpu
                _err = self._product_err(mat_cpu, mat_gpu)
            else:
                _err = self.model_err(X_, X_gpu)
                            
            if _err > max_err:
                calls_to_max = i+1
                max_err   = _err

        return calls_to_max, max_err
    
    def compute_adv_input(self) -> torch.Tensor:
        X_adv        = self.input_matrix.clone()
        idx, n_calls = self.solution

        X_flat = X_adv.view(-1)                       # shares storage with X_adv
        idx  = torch.as_tensor(idx, dtype=torch.long)

        for _ in range(n_calls):
            X_flat[idx] = torch.nextafter(X_flat[idx], self.INFTY)

        self.X_adv = X_adv
        return X_adv

    #####################################
    #                 PLOTTING
    #####################################
    def plot_max(self, y, y_hist, fname=None, show_zero=False):
        plot_max(y, y_hist, 
                 labels=['random'],
                 n_latent=self.n_input*self.n_latent,
                 p= self.q,
                 fname=fname,
                 show_zero=show_zero)
        
    ################################
    ###    MUTATION FUNCTIONS    ###
    ################################
    @staticmethod
    def _not_in_sorted(cand, sorted_arr):
        """Boolean mask: which cand are NOT in sorted_arr (binary search, O(m log k))."""
        if sorted_arr is None or sorted_arr.size == 0:
            return np.ones(cand.shape, dtype=bool)
        pos = np.searchsorted(sorted_arr, cand)
        pos[pos == sorted_arr.size] = sorted_arr.size - 1
        return sorted_arr[pos] != cand


    def random_geneset(self, weighted=None):
        return self._draw_distinct(self.n_perturbed, weighted=weighted)


    # @staticmethod
    # def _strides(shape):
    #     strides = [1] * len(shape)
    #     for i in range(len(shape) - 2, -1, -1):
    #         strides[i] = strides[i + 1] * shape[i + 1]
    #     return strides
    
    # # -------- replaces index_to_binary_string --------
    # def indices_to_geneset(self, *indices):
    #     """Multi-dim indices -> sorted int64 array of flat positions (the genome)."""
    #     idx = [np.asarray(i, dtype=np.int64) for i in indices]
    #     flat = np.zeros_like(idx[0])
    #     strides =self._strides(self.input_shape)
    #     for i, stride in zip(idx, strides):
    #         flat += i * stride
    #     return np.sort(flat)

    # # -------- replaces binary_string_to_index --------
    # def geneset_to_indices(self, genome):
    #     """Sorted int64 array of flat positions -> multi-dim index tuple."""
    #     flat = np.asarray(genome, dtype=np.int64)
    #     remainder = flat.copy()
    #     indices = []
    #     for stride in self.strides:
    #         indices.append(remainder // stride)
    #         remainder %= stride
    #     return tuple(indices)
    
    # -------- replaces mutate_binary_string --------
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
    
    # def mutate_geneset(self, genome, n_mutations=1):
    #     """genome: 1D int array of active flat positions (length k << total)."""
    #     genome = np.asarray(genome, dtype=np.int64)
    #     k = len(genome)

    #     # pick which currently-on positions to clear
    #     clear_pos = np.random.choice(k, size=n_mutations, replace=False)

    #     active = set(genome.tolist())
    #     new_vals = set()
    #     while len(new_vals) < n_mutations:
    #         for cand in self._draw_flat(4 * n_mutations).tolist():
    #             if cand not in active and cand not in new_vals:
    #                 new_vals.add(cand)
    #                 if len(new_vals) == n_mutations:
    #                     break

    #     genome[clear_pos] = list(new_vals)
    #     return np.sort(genome)

    def crossover_uniform(self, g1, g2):
        s1, s2 = set(np.asarray(g1).tolist()), set(np.asarray(g2).tolist())
    
        one_zero = list(s1 - s2)   # on in g1, off in g2
        zero_one = list(s2 - s1)   # on in g2, off in g1
    
        n_swap = min(len(zero_one), len(one_zero)) // 2
        to_s1 = set(random.sample(zero_one, n_swap))  # move into s1
        to_s2 = set(random.sample(one_zero, n_swap))  # move into s2
    
        new_s1 = (s1 - to_s2) | to_s1
        new_s2 = (s2 - to_s1) | to_s2
    
        return (np.array(sorted(new_s1), dtype=np.int64),
                np.array(sorted(new_s2), dtype=np.int64))

    def recombine(self, g1, g2, mutation_rate=None):
        o1, o2 = self.crossover_uniform(g1, g2)
        n1 = None if mutation_rate is None else max(1, int(round(mutation_rate * o1.size)))
        return self.mutate_geneset(o1, n1), self.mutate_geneset(o2, n1)
    
    # def recombine(self, g1, g2):
    #     o1, o2 = self.crossover_uniform(g1, g2)
    #     o1 = self.mutate_geneset(o1)
    #     o2 = self.mutate_geneset(o2)
    #     return o1, o2