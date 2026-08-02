import torch
import random
import numpy as np
from tqdm import tqdm
# from torch.linalg import vector_norm, matrix_norm, multi_dot
from adv_matrix import AdvPerturbation, _nextafter


class SimulatedAnnealingSearch(AdvPerturbation):

    def __init__(self,
                 input_matrix: torch.Tensor,
                 func,
                 c,
                 p,
                 max_calls = 256):

        super().__init__(input_matrix, func, c, max_calls)

        self.p  = p
        self._p = int(p*input_matrix.numel())
                

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

    # def mutate_binary_string(self, s, n_mutations=1):
    #     s = list(s)
    #     ones  = [i for i, b in enumerate(s) if b == '1']
    #     zeros = [i for i, b in enumerate(s) if b == '0']

    #     to_clear = random.sample(ones,  n_mutations)
    #     to_set   = random.sample(zeros, n_mutations)

    #     for i in to_clear: s[i] = '0'
    #     for i in to_set:   s[i] = '1'
    #     return ''.join(s)

    # def recombine(self, s1):

    #     # max_perm = self.p//2
    #     for _ in range(random.randint(1, 9)):
    #         o1 = self.mutate_geneset(s1)
    #         s1 = o1
    #     return o1

    def random_geneset(self):
        """Random genome via rejection sampling — better when total >> k."""
        chosen = set()
        while len(chosen) < self._p:
            chosen.add(random.randrange(self.total))
        return np.array(sorted(chosen), dtype=np.int64)

    # def random_geneset(self):
    #     positions = random.sample(range(self.total), self._p)
    #     bitmask = 0
    #     for p in positions:
    #         bitmask |= (1 << p)
    #     return bitmask

    def generate_new_sol(self, index):

        x_str = self.indices_to_geneset(*index)
        # x1 = self.recombine(x_str)
        x1, _ = self.recombine(x_str, self.random_geneset())
        x1 = self.mutate_geneset(x1)

        return self.geneset_to_indices(x1)

    def init_temp(self, q=0.9, n_samples=100, verbose=False):
        """
        Calculates a initial temperature that will allow q% of 
        'worse' solutions to be accepted
        """

        obj_delta = np.zeros(n_samples)
        idx       = self._sample_entries(self._p)
        _, err    = self.compute_max_err(idx)

        if verbose:
            pbar = tqdm(range(n_samples)) 
            pbar.set_description(f"Sampling {n_samples} moves to set init temperature")
        else:
            pbar = range(n_samples)

        for i in pbar:
            idx     = self.generate_new_sol(idx)
            _, _err = self.compute_max_err(idx)

            obj_delta[i] = err-_err

            err = _err

        T0  = -np.mean(obj_delta[obj_delta>0])/np.log(q)

        return T0


    def search(self, L, *,
               alpha = 0.8,
               tol = 1e-15,
               early_stopping = True,
               #decay = "linear",
               verbose=False):

        self.compute_max_err.reset()
        self.ulp_calls = [0]

        k       = 0
        T       = self.init_temp(verbose=verbose) # Initialize temperature
        counter = 0
                
        iter_idx             = self._sample_entries(self._p)
        iter_calls, iter_err = self.compute_max_err(iter_idx)

        Y = [iter_err]
        self.ulp_calls.append(self.compute_max_err.call_count)

        # Init the best sol
        best_idx   = iter_idx
        best_err   = iter_err
        best_calls = iter_calls

        if verbose:
            print(f"Starting search w/ T0 = {T:.3e}")

        while (T > tol and counter < 250 
               and self.compute_max_err.call_count < self.c
               ):

            accept_prob = None

            for _ in range(L):

                idx = self.generate_new_sol(iter_idx)
                n_calls, _err = self.compute_max_err(idx)
                _err = abs(_err)

                if (_err > iter_err 
                    # or (_err == iter_err and n_calls < iter_calls)
                    ):
                    iter_err   = _err
                    iter_idx   = idx
                    iter_calls = n_calls
                else:
                    delta = (_err - iter_err) / T
                    accept_prob = np.exp(delta)
                    if random.uniform(0, 1) < accept_prob:
                        iter_err   = _err
                        iter_idx   = idx
                        iter_calls = n_calls

                # Update the global best, independent of acceptance criteria above
                if (_err > best_err 
                    or (_err == best_err and n_calls < best_calls)
                    ):
                    best_idx   = idx
                    best_err   = _err
                    best_calls = n_calls
                    # break

            self.ulp_calls.append(self.compute_max_err.call_count)

            Y.append(iter_err)

            if early_stopping:
                if best_err > getattr(self, "_last_best_err", -np.inf):
                    counter = 0
                else:
                    counter += 1
                self._last_best_err = best_err

            k += 1
            T = alpha*T
                        
            if verbose and k%10==0:
                print(f"k = {k} ({counter}): ", 
                    f"max error = {best_err:.3e} , ulp_calls = {best_calls} --",
                    f"iter error = {iter_err:.3e} - temp={T:.3e}, p<{accept_prob:.3e}")

        if verbose:
            print(f"Terminated in {len(Y)} iterations w/ error = {best_err:.4e}")

        Y = torch.Tensor(Y).unsqueeze(0)

        solution = (best_idx, best_calls)

        return Y, solution, self.ulp_calls