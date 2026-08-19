import torch
import hashlib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.linalg import vector_norm, multi_dot

from matplotlib.lines import Line2D


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


def product_err(mat_cpu, mat_gpu):
    """
    Used to calculate the error for matrix multiplication:
    mat_cpu: list of matrices in CPU device
    mat_gpu: list of matrices hosted in GPU
    """
    y_cpu  = multi_dot(mat_cpu)
    y_gpu  = multi_dot(mat_gpu)
    y_diff = (y_cpu - y_gpu.cpu()).ravel().squeeze()

    if len(y_diff.shape) > 0:
        _y = vector_norm(y_diff, ord=np.inf).item()
    else:
        _y = y_diff.item()

    return _y


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
