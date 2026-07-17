import torch
import random
import numpy as np
# from torch.linalg import vector_norm, matrix_norm, multi_dot
from adv_matrix import AdvPerturbation, _nextafter


class SimulatedAnnealingSearch(AdvPerturbation):

    def __init__(self,
                 input_matrix: torch.Tensor,
                 weights,
                 p,
                 max_calls = 256):

        super().__init__(input_matrix, weights, p, max_calls)

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

    def mutate_binary_string(self, s, n_mutations=1):
        s = list(s)
        ones  = [i for i, b in enumerate(s) if b == '1']
        zeros = [i for i, b in enumerate(s) if b == '0']

        to_clear = random.sample(ones,  n_mutations)
        to_set   = random.sample(zeros, n_mutations)

        for i in to_clear: s[i] = '0'
        for i in to_set:   s[i] = '1'
        return ''.join(s)

    def recombine(self, s1):

        # max_perm = self.p//2
        for _ in range(random.randint(1, 9)):
            o1 = self.mutate_binary_string(s1)
            s1 = o1
        return o1

    def generate_new_sol(self, index):

        x_str = self.index_to_binary_string(*index)
        # x1 = self.recombine(x_str)
        x1 = self.mutate_binary_string(x_str)

        return self.binary_string_to_index(x1)

    def search(self, c, L, *,
               cooling_r = 0.8,
               tol = 1e-15,
               early_stopping = True,
               #decay = "linear",
               verbose=False):

        _nextafter.reset()
        self.ulp_calls = [0]

        iter_idx  = self._sample_entries()
        iter_calls, iter_err = self.compute_max_err(iter_idx)

        Y = [iter_err]
        k = 0
        counter = 0
        self.ulp_calls.append(_nextafter.call_count)

        # Init the best sol
        best_idx   = iter_idx
        best_err   = iter_err
        best_calls = iter_calls

        # def num_neighbors(c) -> int:
        #     if c > 1e-7:
        #         return 2
        #     else:
        #         return 8


        while c > tol and counter < 50 and _nextafter.call_count < self.max_ulp_calls:

            # L = num_neighbors(c)
            for _ in range(L):

                idx = self.generate_new_sol(iter_idx)
                n_calls, _err = self.compute_max_err(idx)
                _err = abs(_err)

                if (_err > iter_err) or (_err == iter_err and n_calls < iter_calls):
                    iter_err   = _err
                    iter_idx   = idx
                    iter_calls = n_calls
                else:
                    temp = (_err-best_err)/c
                    if random.uniform(0, 1) < np.exp(temp):
                        iter_err   = _err
                        iter_idx   = idx
                        iter_calls = n_calls

                # Update the global best, independent of acceptance criteria above
                if (_err > best_err) or (_err == best_err and n_calls < best_calls):
                    best_idx   = idx
                    best_err   = _err
                    best_calls = n_calls
                    break

            self.ulp_calls.append(_nextafter.call_count)

            Y.append(iter_err)

            if early_stopping:
                if Y[-1] > Y[-2]:
                    counter = 0
                else:
                    counter +=1

            c = cooling_r*c
            k += 1

            if verbose and k%10==0:
                print(f"k = {k} ({counter}): ", 
                    f"max error = {best_err:.3e} , ulp_calls = {best_calls} -- ",
                    f"iter error = {iter_err:.3e} - temp={c:.3e}, p<{np.exp(temp):.3e}")

        if verbose:
            print(f"Terminated in {len(Y)} iterations w/ error = {best_err:.4e}")

        Y = torch.Tensor(Y).unsqueeze(0)

        solution = (best_idx, best_calls)

        return Y, solution, self.ulp_calls