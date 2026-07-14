import torch
import functools
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from torch.linalg import vector_norm, multi_dot


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

    def __init__(self, input_matrix: torch.Tensor, weights, p, 
                 max_iter = 256):

        self.input_matrix = input_matrix

        if isinstance(weights, torch.Tensor):
            weights = [weights]

        # Dimension check using consecutive pairs
        _matrices = [input_matrix] + weights
        for a, b in zip(_matrices, _matrices[1:]):
            if a.shape[-1] != b.shape[-2]:
                raise ValueError(
                    f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)} — "
                    f"dim {a.shape[-1]} != {b.shape[-2]}"
                )

        self.weights     = weights
        self.weights_gpu = [m.to("cuda") for m in self.weights]

        self.n_input  = input_matrix.shape[0]
        self.n_latent = input_matrix.shape[1]

        self.p             = p
        self.max_iter      = max_iter
        self.max_ulp_calls = p*max_iter

    def _sample_entries(self):
        flat_idx = torch.randperm(self.input_matrix.numel())[:self.p]
        return (flat_idx // self.n_latent, flat_idx % self.n_latent)

    def random_perturbation(self, step=1, verbose =False):

        M_       = self.input_matrix.clone()
        M_gpu    = M_.to("cuda")
        mat_cpu  = [None] + self.weights
        mat_gpu  = [None] + self.weights_gpu

        abs_err   = -1
        # max_error = 0
        flat_len  = self.n_input*self.n_latent
        n_iter    = self.p*self.max_iter//step
        infty     = torch.tensor(torch.inf)

        y = np.zeros(n_iter)
        perturbation_dict = dict()

        iterator = tqdm(range(n_iter)) if verbose else range(n_iter)

        for i in iterator:

            flat_idx = np.random.choice(flat_len)
            idx = (flat_idx // self.n_latent, flat_idx % self.n_latent)

            if idx in perturbation_dict:
                perturbation_dict[idx] += 1
            else:
                perturbation_dict.update({idx: 1})

            # idx = self._sample_entries()
            # for i in range(self.p):
            #     j = (idx[0][i].item(), idx[1][i].item())
            #     if j in perturbation_dict:
            #         perturbation_dict[j] += step
            #     else:
            #         perturbation_dict.update({j: step})
 
            if step > 1:
                for _ in range(step):
                    M_[idx] = _nextafter(M_[idx], infty)
            else:
                M_[idx] = _nextafter(M_[idx], infty)

            M_gpu.copy_(M_, non_blocking=True)

            mat_cpu[0] = M_
            mat_gpu[0] = M_gpu

            y_cpu  = multi_dot(mat_cpu)
            y_gpu  = multi_dot(mat_gpu)
            y_diff = (y_cpu - y_gpu.cpu()).ravel().squeeze()

            if len(y_diff.shape) > 0:
                _y = vector_norm(y_diff, ord=np.inf).item()
            else:
                _y = y_diff.item()

            y[i] = _y

            if abs(_y) > abs_err:
                abs_err  = abs(_y)
                max_pert = perturbation_dict.copy()
                # max_error    = y_

        return torch.Tensor(y).unsqueeze(0), max_pert
    

    def compute_max_err(self, indices):

        M_    = self.input_matrix.clone()
        M_gpu = M_.to("cuda")
        infty = torch.tensor(torch.inf)

        mat_cpu = [None] + self.weights
        mat_gpu = [None] + self.weights_gpu

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