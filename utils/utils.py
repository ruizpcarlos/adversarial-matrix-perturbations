import hashlib
import pickle
import platform
from functools import partial

import torch
import psutil  # pip install psutil
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.linalg import vector_norm, multi_dot
import torchvision.models as models



_FORMATS = {
    torch.float32:  (torch.int32, 23, 8),
    torch.bfloat16: (torch.int16,  7, 8),
}


def print_sys_specs():
  
    print("Processor:", platform.processor())
    print("Architecture:", platform.machine())
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    print("CPU Frequency:", psutil.cpu_freq())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())
        print("Memory allocated:", torch.cuda.memory_allocated(0))
        print("Memory reserved:", torch.cuda.memory_reserved(0))
    else:
        print("GPU not available: Connect to a GPU environment")


def load_model_and_weights(model_name:str, dtype:torch.dtype = torch.float32):

    if model_name.upper().startswith("EFF"):
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).eval()
        W     = torch.transpose(model.classifier[1].weight.data, 0, 1).to(dtype)
    else:
        model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1).eval()
        W     = torch.transpose(model.fc.weight.data, 0, 1).to(dtype)

    return model, W


def hash_tensor(*tensors: torch.Tensor):
    h = hashlib.sha256()
    cpu = torch.device("cpu")
    for t in tensors:
        if t.device != cpu:
            t = t.cpu()

        if t.dtype == torch.bfloat16:
            h.update(t.float().detach().numpy().tobytes())
        else:
            h.update(t.detach().numpy().tobytes())
    return h.digest()


def save_dict_to_pickle(data_dict: dict, filename):

    with open(filename, 'wb') as fp:
        pickle.dump(data_dict, fp, 
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"--- Saved results in {filename}")


def track_evol_(Y, n_calls):

    max_err = Y.squeeze()
    idx_aux = n_calls

    y_max = torch.zeros(n_calls[-1])

    for k in range(max_err.shape[0]):
        y_max[idx_aux[k]: idx_aux[k+1]] = max_err[k]

    return y_max.unsqueeze(0)


def pad_to_match(tensors):
    max_len = max(t.size(1) for t in tensors)
    padded_list = [F.pad(t, (0, max_len - t.size(1)), mode='replicate') for t in tensors]
    return torch.cat(padded_list, dim=0)


def wrap_score(x: torch.Tensor, alpha: float = 0.5, max_calls: int = 32) -> torch.Tensor:
    """
    Per-entry score in [0, 1]. High = easy to wrap AND large magnitude.

    closeness : 1 when a single nextafter wraps, decreasing linearly to 0 once
                steps > window.
    bonus     : normalised exponent (larger |x| -> larger bonus).
    score     : closeness * ((1 - alpha) + alpha * bonus)
                alpha=0 -> pure closeness; alpha=1 -> closeness * bonus.
    """
    int_dtype, m_bits, e_bits = _FORMATS[x.dtype]
    max_mant = (1 << m_bits) - 1
    max_exp  = (1 << e_bits) - 1

    # fp32: only entries that can wrap within the budget score > 0
    # bf16: full mantissa range (2^7) -> score grows linearly over the binade
    window = max_calls if x.dtype == torch.float32 else (1 << m_bits)

    x_int    = x.view(int_dtype)
    sign     = (x_int >> (m_bits + e_bits)) & 1
    exponent = (x_int >> m_bits) & max_exp
    mantissa = x_int & max_mant

    steps = torch.where(sign == 0, max_mant - mantissa + 1, mantissa + 1)

    # a) closeness: steps=1 -> 1.0, steps=window -> 1/window, steps>window -> 0
    closeness = (window - steps + 1).clamp(min=0).float() / window

    # b) magnitude bonus: min-max normalised exponent over finite entries
    finite = exponent < max_exp
    e = exponent.float()
    lo, hi = e[finite].min(), e[finite].max()
    bonus = (e - lo) / (hi - lo).clamp(min=1.0)

    score = closeness * ((1 - alpha) + alpha * bonus)
    return torch.where(finite, score, torch.zeros_like(score))


def vector_distance(x:torch.Tensor, y:torch.Tensor, ord:float=np.inf) -> float:

    diff = (x-y).ravel().squeeze()

    return torch.linalg.vector_norm(diff, ord=ord).item()

def product_err(mat_cpu, mat_gpu, objective_fn=None):
    """
    Used to calculate the error for matrix multiplication:
    mat_cpu: list of matrices in CPU device
    mat_gpu: list of matrices hosted in GPU
    objective_fn: callable(y_diff: Tensor) -> Tensor (scalar-valued).
                  Defaults to inf-norm (max abs difference), matching the
                  previous hardcoded behavior.
    """
    if objective_fn is None:
        objective_fn = vector_distance

    y_cpu  = multi_dot(mat_cpu)
    y_gpu  = multi_dot(mat_gpu)

    return objective_fn(y_cpu, y_gpu.cpu())

 
# def product_err(mat_cpu, mat_gpu):
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


def cum_stats(y:torch.Tensor):

    m = y.shape[1]

    iter_max   = torch.cummax(y.max(dim=0).values.squeeze(),
                            dim=0).values
    acc_q90    = torch.tensor([torch.quantile(y[:, :k].float(), 0.99) for k in range(1, m+1)])
    acc_q75    = torch.tensor([torch.quantile(y[:, :k].float(), 0.75) for k in range(1, m+1)])
    acc_median = torch.tensor([torch.quantile(y[:, :k].float(), 0.5) for k in range(1, m+1)])

    # cum_sum = y.cumsum(dim=1).sum(dim=0)  # cumulative sum over columns, summed across rows
    # counts = torch.arange(1, m + 1) * n_rows
    # cum_mean = cum_sum / counts

    return torch.stack([iter_max, acc_q90, acc_q75, acc_median])


def tensor_to_plotting_inputs(y:torch.Tensor):

    n_calls = y.shape[1]
    end     = int(np.log2(n_calls))+1
    arrays  = {2**k: y[:, :2**k].ravel().numpy() for k in range(5, end)}

    df = pd.DataFrame({
        "n_calls": np.concatenate([np.full(len(v), j) for j, v in arrays.items()]),
        "error": np.concatenate(list(arrays.values()))
    })

    y_dist = arrays[n_calls]

    return df, y_dist


def dict_to_plotting_data(y_stats):

    df = pd.DataFrame({
                "q": np.concatenate([np.full(len(err), q) for q, err in y_stats.items()]),
                "error": np.concatenate(list(y_stats.values()))
                })

    y_hist = df.error.values

    return df, y_hist
