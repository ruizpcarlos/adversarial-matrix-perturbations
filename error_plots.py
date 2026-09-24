import pickle
import torch
import random
import copy
import itertools
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.linalg import vector_norm

from utils.cli import parse_args#, DTYPES
from utils.utils import product_err, cum_stats, tensor_to_plotting_inputs, dict_to_plotting_data, print_sys_specs, load_model_and_weights
from utils.plotting_utils import *



class DeltaNormPlots:

    def __init__(self, n_latent: int, seed:int):

        random.seed(seed)
        torch.manual_seed(seed)

        self.n_latent = n_latent

        self.X = torch.randn(self.n_latent, self.n_latent)
        
        self.INFTY = torch.tensor(torch.inf)
        
        
    def sample_entries(self, num_samples=1):
    
        flat_idx = torch.randint(self.X.numel(), 
                                 (num_samples,)
                                 )
    
        return (flat_idx // self.n_latent, flat_idx % self.n_latent)
    

    def perturbation_norm(self, q=None, n_calls=256, n_calls_bf16 = 128):

        M = self.X
    
        if q is not None:
            _p  = int(q*M.numel())
            idx = self.sample_entries(_p)

        M_aux      = M.clone()
        M_bf16     = M.clone().to(dtype=torch.bfloat16)
        M_bf16_aux = M.clone().to(dtype=torch.bfloat16)

        delta_fp32 = torch.zeros(2, n_calls)
        delta_bf16 = torch.zeros(2, n_calls_bf16)

        for i in range(n_calls):

            if i < n_calls_bf16:
                if q is not None:
                    M_bf16_aux[idx] = torch.nextafter(M_bf16_aux[idx], self.INFTY)
                else:
                    M_bf16_aux      = torch.nextafter(M_bf16_aux, self.INFTY)

                delta_bf16[0, i] = vector_norm(M_bf16-M_bf16_aux, ord=np.inf).item()
                delta_bf16[1, i] = vector_norm(M_bf16-M_bf16_aux).item()
                
                
            if q is not None:
                M_aux[idx] = torch.nextafter(M_aux[idx], self.INFTY)
            else:
                M_aux      = torch.nextafter(M_aux, self.INFTY)

            delta_fp32[0, i] = vector_norm(M-M_aux, ord=np.inf).item()
            delta_fp32[1, i] = vector_norm(M-M_aux).item()
            
        return delta_fp32, delta_bf16


    def perturbation_size_plots(self, n_calls, n_calls_bf16, qs_list=None):

        DEFAULT_FNAME = "perturbation_norm"

        if qs_list is None:
            fname = DEFAULT_FNAME + ".png"
            delta_f32, delta_bf16 = self.perturbation_norm(n_calls=n_calls, 
                                                           n_calls_bf16=n_calls_bf16)
        else:
            delta_f32  = torch.zeros(len(qs_list),
                                     2,
                                     n_calls)
            delta_bf16 = torch.zeros(len(qs_list),
                                     2,
                                     n_calls_bf16)

            for i, q in enumerate(qs_list):
                aux_f32, aux_bf16 = self.perturbation_norm(q=q, n_calls=n_calls, n_calls_bf16=n_calls_bf16)
                delta_f32[i, :]   = aux_f32
                delta_bf16[i, :]  = aux_bf16

            fname = DEFAULT_FNAME + "_q.png"

        delta_growth_plots(delta_f32, delta_bf16, 
                            qs_list=qs_list, 
                            fname=fname)
        print(f"Plot saved to {fname}")



class ErrorPlots:

    def __init__(self, model_name: str, seed:int, n_samples:int=1):
    
        random.seed(seed)
        torch.manual_seed(seed)
    
        self.INFTY  = torch.tensor(torch.inf)
        self.DTYPES = [torch.bfloat16, torch.float32]

        self.n_samples  = n_samples
        self.func_names = ["random",
                           f"{model_name}_clf",
                           model_name
                           ]

        model, W   = load_model_and_weights(model_name)
        self.model = model
        self.W     = W
            
        self.n_latent = self.W.shape[0]
        self.W0       = torch.randn(self.n_latent, self.n_latent)

        self.init_func_dict()
        
        # Experiment inputs
        self.X0    = torch.randn(self.n_samples, self.n_latent, self.n_latent)
        self.X_img = torch.randn(self.n_samples, 1, 3, 224, 224)


    def init_func_dict(self):
        aux_dict = dict(zip(self.func_names, 
                            [self.W0, self.W, self.model])
                            )
        self.functions = {}
        self.functions_gpu = {}

        for name, dtype in itertools.product(self.func_names, self.DTYPES):

            src = aux_dict[name]

            if isinstance(src, nn.Module):
                cpu_version = copy.deepcopy(src).to(dtype)
            else:
                cpu_version = src.to(dtype)  

            self.functions[(name, dtype)] = cpu_version

            self.functions_gpu[(name, dtype)] = copy.deepcopy(cpu_version).to('cuda') \
                if isinstance(cpu_version, nn.Module) else cpu_version.to('cuda')


    def model_err(self, x_cpu, x_gpu, func, func_gpu):
        """
        Used to calculate the error for forward pass multiplication:
        mat_cpu: list of matrices in CPU device
        mat_gpu: list of matrices hosted in GPU
        """

        with torch.no_grad():
            y_cpu  = func(x_cpu)
            y_gpu  = func_gpu(x_gpu)

        y_diff = (y_cpu - y_gpu.cpu()).ravel().squeeze()

        if len(y_diff.shape) > 0:
            _y = vector_norm(y_diff, ord=np.inf).item()
        else:
            _y = abs(y_diff.item())

        return _y

    def compute_arch_diff(self, M, func_name, indices=None, n_calls=256):

        dtype = M.dtype
        X_    = M.clone()
        X_gpu = X_.to("cuda")
        
        func     = self.functions[(func_name, dtype)]
        func_gpu = self.functions_gpu[(func_name, dtype)]

        if isinstance(func, torch.Tensor):
            weights_cpu = [func]
            weights_gpu = [func_gpu]

            _matrices = [X_, func]
            # Dimension check: matrix[i] cols must match matrix[i+1] rows
            for i in range(len(_matrices) - 1):
                if _matrices[i].shape[-1] != _matrices[i + 1].shape[-2]:
                    raise ValueError(
                        f"Shape mismatch at position {i} and {i+1}: "
                        f"{tuple(_matrices[i].shape)} vs {tuple(_matrices[i+1].shape)} —"
                        f"dim {_matrices[i].shape[-1]} != {_matrices[i+1].shape[-2]}"
                    )
            mat_cpu     = [X_] + weights_cpu
            mat_gpu     = [X_gpu] + weights_gpu
            tensor_prod = True
        elif isinstance(func, nn.Module):
            tensor_prod = False

        iters = n_calls-1 if n_calls>1 else n_calls

        if n_calls > 1:
            Y     = torch.zeros(n_calls)
            Y[0]  = product_err(mat_cpu, mat_gpu) if tensor_prod else self.model_err(X_, X_gpu, func, func_gpu)
            iters = n_calls-1
        else:
            Y     = torch.zeros(n_calls+1)
            Y[0]  = product_err(mat_cpu, mat_gpu) if tensor_prod else self.model_err(X_, X_gpu, func, func_gpu)
            iters = n_calls

        for i in range(iters):

            if indices is not None:
                X_[indices] = torch.nextafter(X_[indices], self.INFTY)
            else:
                X_ = torch.nextafter(X_, self.INFTY)

            X_gpu.copy_(X_, non_blocking=True)

            if tensor_prod:
                mat_cpu[0] = X_
                mat_gpu[0] = X_gpu
                _err = product_err(mat_cpu, mat_gpu)
            else:
                _err = self.model_err(X_, X_gpu, func, func_gpu)

            Y[i+1] = _err

        return Y#.unsqueeze(0)

    def baseline_err(self, X, func_name, idx=None):

        y_ = self.compute_arch_diff(X, func_name, 
                                    n_calls=0)

        return y_.squeeze().item()


    def error_distribution(self, n_calls, n_calls_bf16,
                            func_names=None, verbose=False):

        if isinstance(func_names, str):
            func_names = [func_names]
                  
        func_aux = self.func_names if not func_names else func_names

        shape_bf16 = (len(func_aux),
                    self.n_samples,
                    n_calls_bf16
                    )
        shape_fp32 = (len(func_aux),
                    self.n_samples,
                    n_calls
                    )

        y_dist_bf16 = torch.zeros(*shape_bf16)
        y_dist_fp32 = torch.zeros(*shape_fp32)

        call_list = [n_calls_bf16, n_calls]
        y_list    = [y_dist_bf16, y_dist_fp32]

        for k, name in enumerate(func_aux):

            if verbose:
                print(f"Computing errors for {name}")

            _func_check_aux = self.functions[(name, self.DTYPES[0])]

            if isinstance(_func_check_aux, torch.Tensor):
                X = self.X0
            elif isinstance(_func_check_aux, nn.Module):
                X = self.X_img
                
            for j, dtype in enumerate(self.DTYPES):

                X = X.clone().to(dtype)

                if verbose and self.n_samples>1:
                    pbar = enumerate(tqdm(X, desc=f"Proccesing {dtype} data", unit="samples"))
                else:
                    pbar = enumerate(X)

                for i, x in pbar:
                    Y  = self.compute_arch_diff(x, name, n_calls=call_list[j])

                    y_list[j][k, i, :] = Y.ravel()

        return y_list
    

    def plot_distributions(self, y_dist):

        fname = "dtype_model_err.png"

        plot_error_histograms(y_dist,  self.func_names, self.DTYPES, fname=fname)


    def plot_err_signals(self, y_hist, funcname, show_zero=False):

        idx = self.func_names.index(funcname)

        err_f32  = y_hist[0][idx]
        err_bf16 = y_hist[1][idx]
        
        fname = f"error_signals_{funcname}.png"
        
        iter_error((err_bf16, err_f32),
                        show_zero=show_zero,
                        fname=fname)


    def stats_plots(self, y_dist:list[torch.Tensor], func_names):

        if isinstance(func_names, str):
            func_names = [func_names]
        for y in y_dist:
            assert y.shape[0]==len(func_names), "Inputdims and number of functions do not match!!!"
                    
        y_bf16 = y_dist[0]
        y_fp32 = y_dist[1]

        for k, name in enumerate(func_names):

            err_bf16 = y_bf16[k]
            err_fp32 = y_fp32[k]
            
            stats_bf16 = cum_stats(err_bf16)
            stats_fp32 = cum_stats(err_fp32)
            
            bf16_data = tensor_to_plotting_inputs(err_bf16)
            fp32_data = tensor_to_plotting_inputs(err_fp32)

            fname1 = f"acc_distributions_{name}.png"
            iter_error_distributions((bf16_data, fp32_data),
                                        fname=fname1)
            
            fname2 = f"cum_stats_{name}.png"
            plot_cum_stats([stats_fp32, stats_bf16],
                                        fname=fname2)
            


class qErrorPlots(ErrorPlots):

    def __init__(self, model_name: str, seed:int, list_q, idx_samples:int=30):

        super().__init__(model_name, seed)

        self.list_q      = list_q
        self.idx_samples = idx_samples
        self.X_img       = torch.randn(1, 3, 224, 224) #OVERWRITE X_img (this class samples over indices, not images)


    def flat_to_3d(self, idx, stride=None):

        stride = self.n_latent if stride is None else stride

        aux_idx = idx % (stride**2)
        c = idx // (stride**2)
        h = aux_idx // stride
        w = aux_idx % stride
        return c, h, w
    
    def sample_entries(self, X, num_samples=1):
            
        flat_idx = torch.randperm(X.numel())[:num_samples]

        if X.dim()>2:
            indices = self.flat_to_3d(flat_idx, X.shape[2])
            return (torch.zeros(num_samples, dtype=int), *indices)
        else:
            return (flat_idx // self.n_latent, flat_idx % self.n_latent)
            
    # def sample_entries(self, X, n_q=1):
    #     flat_idx = torch.randint(X.numel(), 
    #                                  (n_q,)
    #                                  )
    #     return (flat_idx // self.n_latent, flat_idx % self.n_latent)


    def max_error_q(self, X, func_name, n_calls=256, verbose=False):

        n_calls = 128 if X.dtype==torch.bfloat16 else n_calls

        n_entries = X.numel()
        y_stats   = {}

        pbar = tqdm(self.list_q, desc="Computing max error for index %") if verbose else self.list_q

        for q in pbar:

            n_q  = int(q*n_entries)
            y_max = torch.zeros(self.idx_samples)

            for i in range(self.idx_samples):

                idx = self.sample_entries(X, n_q)
                y   = self.compute_arch_diff(X, func_name, 
                                             idx,
                                             n_calls=n_calls)
                y_max[i] = torch.max(y).item()
            
            y_stats.update({f"{q}": y_max})

        return y_stats


    def stats_plots_q(self, n_calls, func_names=None, verbose=False):

        if isinstance(func_names, str):
            func_names = [func_names]
                          
        func_aux = self.func_names if not func_names else func_names

        for name in func_aux:

            _func_check_aux = self.functions[(name, self.DTYPES[1])]
            
            if isinstance(_func_check_aux, torch.Tensor):
                X_fp32 = self.X0.squeeze()
            elif isinstance(_func_check_aux, nn.Module):
                X_fp32 = self.X_img

            X_bf16 = X_fp32.clone().to(self.DTYPES[0])

            y_stats_bf16 = self.max_error_q(X_bf16, name, n_calls, verbose=verbose)
            y_stats_fp32 = self.max_error_q(X_fp32, name, n_calls, verbose=verbose)
            plot_bf16    = dict_to_plotting_data(y_stats_bf16)
            plot_fp32    = dict_to_plotting_data(y_stats_fp32)

            max_bf16 = self.baseline_err(X_bf16, name)
            max_fp32 = self.baseline_err(X_fp32, name)
            max_errs  = (max_bf16, max_fp32)

            fname=f"max_errors_{name}_q.png"
            q_error_distributions([plot_bf16, plot_fp32], max_errs, 
                                  fname=fname)

        
    

if __name__=="__main__":

    SEED       = 161
    model_name = "ResNet"

    args = parse_args()

    n_samples = args.n_samples

    LATENT_DIM   = 512
    N_CALLS_BF16 = 256
    N_CALLS      = 1024
    N_CALLS_Q    = 256
    Q_LIST       = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1]

    # Print CPU and GPU DATA
    print_sys_specs()
    
    # GROWTH OF DELTA
    delta_plots = DeltaNormPlots(n_latent=LATENT_DIM, seed=SEED)

    delta_plots.perturbation_size_plots(N_CALLS, N_CALLS_BF16)
    delta_plots.perturbation_size_plots(N_CALLS_Q, N_CALLS_BF16,
                                        qs_list=Q_LIST)


    # DTYPE - MODEL HISTOGRAMS
    funcname = f'{model_name}_clf'
    
    err_plots = ErrorPlots(model_name=model_name, seed=SEED)
    y_hist    = err_plots.error_distribution(N_CALLS, N_CALLS_BF16)

    err_plots.plot_distributions(y_hist)
    err_plots.plot_err_signals(y_hist, funcname)


    # CUMULATIVE ERROR DISTRIBUTIONS
    from_cache = False
    pkl_name   = "error_dist.pkl"

    acc_err_plots = ErrorPlots(model_name=model_name, 
                               seed=SEED, 
                               n_samples=n_samples)
                      
    if not from_cache:
        y_dist = acc_err_plots.error_distribution(N_CALLS, N_CALLS_BF16,
                                                  verbose=True,
                                                  func_names=funcname)
        with open(pkl_name, 'wb') as f:
                    pickle.dump(y_dist, f)
    else:
        print(f"Reading experiment results from {pkl_name}")
        with open(pkl_name, 'rb') as f:
            y_dist = pickle.load(f)
    
    acc_err_plots.stats_plots(y_dist, func_names=funcname)

    # MAX ERROR DISTRIBUTIONS FOR Q
    err_plots_q = qErrorPlots(model_name=model_name, 
                              seed=SEED,
                              list_q=Q_LIST, 
                              idx_samples=n_samples)

    err_plots_q.stats_plots_q(N_CALLS_Q, func_names=funcname)