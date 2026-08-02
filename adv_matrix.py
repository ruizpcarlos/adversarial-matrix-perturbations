import torch
import random
import copy
import functools
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.linalg import vector_norm, multi_dot
from utils.utils import product_err, plot_max


class CallTracker:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.func(*args, **kwargs)

    def reset(self):
        self.call_count = 0

@CallTracker
def _nextafter(input, other):
    return torch.nextafter(input, other)

@CallTracker
def nextafter(x: torch.Tensor, offset: torch.Tensor):
    MAX_EXPONENT = 0xFF
    MAX_MANTISSA = 0x7FFFFF

    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    if not torch.is_tensor(offset):
        offset = torch.tensor(offset, dtype=torch.int32)

    x, offset = torch.broadcast_tensors(torch.clone(x), torch.clone(offset))

    assert not torch.is_floating_point(offset)
    assert x.dtype == torch.float32
    assert (-MAX_MANTISSA <= offset).all() and (offset <= MAX_MANTISSA).all()

    x_int = x.view(torch.int32)

    # Extract the sign (1 bit), exponent (8 bits), and mantissa (23 bits)
    sign = (x_int >> 31) & 1
    exponent = (x_int >> 23) & MAX_EXPONENT
    mantissa = x_int & MAX_MANTISSA

    # If
    #   "sign is positive and offset is positive"
    # or
    #   "sign is negative and offset is negative"
    # treat the calculation as an addition that can overflow. Otherwise, it's
    # a substraction that can underflow.
    sign_is_positive = sign == 0
    offset = torch.where(sign_is_positive, offset, -offset)

    underflow = (mantissa + offset) < 0
    overflow = (mantissa + offset) > MAX_MANTISSA

    zero_pass = exponent == 0
    to_inf = exponent == MAX_EXPONENT

    # OVERFLOW
    # Handle "regular" overflow. We add the mantissa and the offset modulo the
    # max mantissa value and increase the exponent by one
    mantissa[overflow] = (mantissa[overflow] + offset[overflow]) & MAX_MANTISSA
    exponent[overflow & ~to_inf] += 1

    # If the exponent was already maximal, we set the mantissa to zero
    # (max exponent + non-zero mantissa is a NaN)
    # This is different from how torch.nextafter handles it. They return a NaN
    # value for some reason.
    mantissa[overflow & to_inf] = 0

    # UNDERFLOW
    # Handle "regular" underflow.
    underflow_r = underflow & ~zero_pass
    underflow_z = underflow & zero_pass

    mantissa[underflow_r] = (mantissa[underflow_r] + offset[underflow_r]) & MAX_MANTISSA
    exponent[underflow_r] -= 1

    # Underflow past zero
    mantissa[underflow_z] = -offset[underflow_z] - mantissa[underflow_z]
    sign[underflow_z] = 1 - sign[underflow_z]

    mantissa[~overflow & ~underflow] = (
        mantissa[~overflow & ~underflow] + offset[~overflow & ~underflow]
    )

    sign = sign & 0x1
    exponent = exponent & MAX_EXPONENT
    mantissa = mantissa & MAX_MANTISSA

    return ((sign << 31) | (exponent << 23) | mantissa).view(torch.float32)


class AdvPerturbation:

    def __init__(self, input_matrix: torch.Tensor, func, p, 
                 max_calls = 256):

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
            self.nn          = None
            # print("RUNNING W TENSOR MULTIPLICATION")
        elif isinstance(func, nn.Module):
            self.tensor_prod = False
            try:
                with torch.no_grad():
                    func.eval()
                    func(input_matrix)
            except Exception as e:
                raise ValueError(f"Input tensor is not a valid input for func: {e}")
            self.weights     = None
            self.nn          = func#.eval()
            self.tensor_prod = False
            # print("RUNNING W CALLABLE TORCH MODULE")
            
        else:
            raise TypeError(f"Received a {type(func)} as func: must be either torch.Tensor or nn.Module.")
                        
        # # Dimension check using consecutive pairs
        # _matrices = [input_matrix] + weights
        # for a, b in zip(_matrices, _matrices[1:]):
        #     if a.shape[-1] != b.shape[-2]:
        #         raise ValueError(
        #             f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)} — "
        #             f"dim {a.shape[-1]} != {b.shape[-2]}"
        #         )

        self.weights_gpu = None if self.weights is None else [m.to("cuda") for m in self.weights]
        self.nn_gpu      = None if self.nn is None else copy.deepcopy(self.nn).eval().to("cuda") 

        self.p  = p
        self._p = int(p*input_matrix.numel())

        self.input_shape = input_matrix.shape
        self.strides = self._strides(self.input_shape)
        self.total = int(input_matrix.numel())
            
        if self.tensor_prod: # Input is a matrix   
            self.n_input  = input_matrix.shape[0]
            self.n_latent = input_matrix.shape[1]
        else: # Input is "image-like": (1, C, H, W)
            self.n_input  = input_matrix.shape[2] # Height
            self.n_latent = input_matrix.shape[3] # Width
        
        self.max_calls    = max_calls
        self.total_calls = self._p*max_calls
    
    def flat_to_3d(self, idx):
        aux_idx = idx % (self.n_latent**2)
        c = idx // (self.n_latent**2)
        h = aux_idx // self.n_latent
        w = aux_idx % self.n_latent
        return c, h, w


    def _sample_entries(self, num_samples):

        flat_idx = torch.randint(self.input_matrix.numel(), 
                                (num_samples,)
                                )

        if not self.tensor_prod:
            indices = self.flat_to_3d(flat_idx)
            return (torch.zeros(num_samples, dtype=int),
                        *indices)
        else:
            return (flat_idx // self.n_latent, flat_idx % self.n_latent)

    
    # def multi_dot_err(self, mat_cpu, mat_gpu):
    #     """
    #     Used to calculate the error for matrix multiplication:
    #     mat_cpu: list of matrices in CPU device
    #     mat_gpu: list of matrices hosted in GPU
    #     """
    #     y_cpu  = multi_dot(mat_cpu)
    #     y_gpu  = multi_dot(mat_gpu)
    #     y_diff = (y_cpu - y_gpu.cpu()).ravel().squeeze()

    #     if len(y_diff.shape) > 0:
    #         _y = vector_norm(y_diff, ord=np.inf).item()
    #     else:
    #         _y = y_diff.item()

    #     return _y
    
    def model_err(self, x_cpu, x_gpu):
        """
        Used to calculate the error for forward pass multiplication:
        mat_cpu: list of matrices in CPU device
        mat_gpu: list of matrices hosted in GPU
        """
        with torch.no_grad():
            y_cpu  = self.nn(x_cpu)
            y_gpu  = self.nn_gpu(x_gpu)
        
        y_diff = (y_cpu - y_gpu.cpu()).ravel().squeeze()

        if len(y_diff.shape) > 0:
            _y = vector_norm(y_diff, ord=np.inf).item()
        else:
            _y = y_diff.item()

        return _y


    def random_perturbation(self, step=1,  pool_ulp=False, verbose =False):

        X_    = self.input_matrix.clone()
        X_gpu = X_.to("cuda")

        if self.tensor_prod:
            mat_cpu  = [None] + self.weights
            mat_gpu  = [None] + self.weights_gpu
        
        abs_err    = -1
        aux_div    = (step*self._p) if pool_ulp else step
        aux_sample = self._p if pool_ulp else 1
        n_iter     = self.total_calls//aux_div
        infty      = torch.tensor(torch.inf)

        y = np.zeros(n_iter)
        perturbation_dict = dict()

        iterator = tqdm(range(n_iter)) if verbose else range(n_iter)

        for i in iterator:

            idx = self._sample_entries(aux_sample)
           
            if idx in perturbation_dict:
                if perturbation_dict[idx] >= self.max_calls:
                    continue
                else:
                    perturbation_dict[idx] += step     
            else:
                perturbation_dict.update({idx: step})

            # idx = self._sample_entries()
            # for i in range(self._p):
            #     j = (idx[0][i].item(), idx[1][i].item())
            #     if j in perturbation_dict:
            #         perturbation_dict[j] += step
            #     else:
            #         perturbation_dict.update({j: step})
 
            if step > 1:
                for _ in range(step):
                    X_[idx] = _nextafter(X_[idx], infty)
            else:
                X_[idx] = _nextafter(X_[idx], infty)

            X_gpu.copy_(X_, non_blocking=True)

            if self.tensor_prod:
                mat_cpu[0] = X_
                mat_gpu[0] = X_gpu

            if not self.tensor_prod:
                _y = self.model_err(X_, X_gpu)
            else:
                _y = product_err(mat_cpu, mat_gpu)

            y[i] = _y

            if abs(_y) > abs_err:
                abs_err  = abs(_y)
                max_pert = perturbation_dict.copy()
                
        return torch.Tensor(y).unsqueeze(0), max_pert
    

    def compute_max_err(self, indices):

        X_    = self.input_matrix.clone()
        X_gpu = X_.to("cuda")
        infty = torch.tensor(torch.inf)

        if self.tensor_prod:
            mat_cpu  = [None] + self.weights
            mat_gpu  = [None] + self.weights_gpu
        
        abs_err      = 0
        max_error    = 0
        calls_to_max = 1

        for i in range(self.max_calls):

            # M_[indices] = nextafter(M_[indices], 1)
            # torch wrapped in counter
            X_[indices] = _nextafter(X_[indices], infty)

            X_gpu.copy_(X_, non_blocking=True)

            if self.tensor_prod:
                mat_cpu[0] = X_
                mat_gpu[0] = X_gpu

            if not self.tensor_prod:
                _err = self.model_err(X_, X_gpu)
            else:
                _err = product_err(mat_cpu, mat_gpu)

            if abs(_err) > abs_err:
                calls_to_max = i+1
                abs_err      = abs(_err)
                max_error    = _err

        return calls_to_max, max_error
    
    #####################################
    #                 PLOTTING
    #####################################
    def plot_max(self, y, y_hist, fname=None, show_zero=False):
        plot_max(y, y_hist, 
                 labels=['random'],
                 n_latent=self.n_input*self.n_latent,
                 p= self.p,
                 fname=fname,
                 show_zero=show_zero)
        
    ################################
    ###    MUTATION FUNCTIONS    ###
    ################################
    @staticmethod
    def _strides(shape):
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return strides
    
    # -------- replaces index_to_binary_string --------
    def indices_to_geneset(self, *indices):
        """Multi-dim indices -> sorted int64 array of flat positions (the genome)."""
        idx = [np.asarray(i, dtype=np.int64) for i in indices]
        flat = np.zeros_like(idx[0])
        strides =self._strides(self.input_shape)
        for i, stride in zip(idx, strides):
            flat += i * stride
        return np.sort(flat)

    # -------- replaces binary_string_to_index --------
    def geneset_to_indices(self, genome):
        """Sorted int64 array of flat positions -> multi-dim index tuple."""
        flat = np.asarray(genome, dtype=np.int64)
        remainder = flat.copy()
        indices = []
        for stride in self.strides:
            indices.append(remainder // stride)
            remainder %= stride
        return tuple(indices)
    
    # -------- replaces mutate_binary_string --------
    def mutate_geneset(self, genome, n_mutations=1):
        """genome: 1D int array of active flat positions (length k << total)."""
        genome = np.asarray(genome, dtype=np.int64)
        k = len(genome)

        # pick which currently-on positions to clear
        clear_pos = np.random.choice(k, size=n_mutations, replace=False)

        active = set(genome.tolist())
        new_vals = set()
        # rejection sampling — fast because k << total
        while len(new_vals) < n_mutations:
            cand = random.randrange(self.total)
            if cand not in active and cand not in new_vals:
                new_vals.add(cand)

        genome[clear_pos] = list(new_vals)
        return np.sort(genome)

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
    
    # def mutate_binary_string(self, s, n_mutations=1):
    #     s = list(s)
    #     ones  = [i for i, b in enumerate(s) if b == '1']
    #     zeros = [i for i, b in enumerate(s) if b == '0']

    #     to_clear = random.sample(ones,  n_mutations)
    #     to_set   = random.sample(zeros, n_mutations)

    #     for i in to_clear: s[i] = '0'
    #     for i in to_set:   s[i] = '1'
    #     return ''.join(s)