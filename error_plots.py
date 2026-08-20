import pickle
import torch
import random
import copy
import itertools
import numpy as np
import torch.nn as nn

import torchvision.models as models

from psutil import *
from tqdm import tqdm
from torch.linalg import vector_norm

from utils.utils import product_err, cum_stats, tensor_to_plotting_inputs, dict_to_plotting_data, print_sys_specs
from utils.plotting_utils import *



class deltaNormPlots:

    def __init__(self, n_latent: int, seed:int):

        random.seed(seed)

        self.n_latent = n_latent

        self.X = torch.randn(self.n_latent, self.n_latent)
        
        self.INFTY = torch.tensor(torch.inf)
        
        
    def sample_entries(self, num_samples=1):
    
        flat_idx = torch.randint(self.X.numel(), 
                                 (num_samples,)
                                 )
    
        return (flat_idx // self.n_latent, flat_idx % self.n_latent)
    

    def perturbation_norm(self, q=None, n_calls=256):

        M = self.X
    
        if q is not None:
            _p  = int(q*M.numel())
            idx = self.sample_entries(_p)

        M_aux      = M.clone()
        M_bf16     = M.clone().to(dtype=torch.bfloat16)
        M_bf16_aux = M.clone().to(dtype=torch.bfloat16)

        delta_fp32    = torch.zeros(2, n_calls)
        delta_bf16 = torch.zeros(2, n_calls)

        for i in range(n_calls):

            if q is not None:
                M_aux[idx]      = torch.nextafter(M_aux[idx], self.INFTY)
                M_bf16_aux[idx] = torch.nextafter(M_bf16_aux[idx], self.INFTY)
            else:
                M_aux      = torch.nextafter(M_aux, self.INFTY)
                M_bf16_aux = torch.nextafter(M_bf16_aux, self.INFTY)

            delta_fp32[0, i] = vector_norm(M-M_aux, ord=np.inf).item()
            delta_fp32[1, i] = vector_norm(M-M_aux).item()
            delta_bf16[0, i] = vector_norm(M_bf16-M_bf16_aux, ord=np.inf).item()
            delta_bf16[1, i] = vector_norm(M_bf16-M_bf16_aux).item()

        return delta_fp32, delta_bf16


    def perturbation_size_plots(self, n_calls, qs_list=None):

        DEFAULT_FNAME = "perturbation_norm"

        if qs_list is None:

            fname = DEFAULT_FNAME + ".png"
            delta_f32, delta_bf16 = self.perturbation_norm(q=None,
                                          n_calls=n_calls)
            
        else:
            output_shape = (len(qs_list),
                            2,
                            n_calls)
            delta_f32  = torch.zeros(*output_shape)
            delta_bf16 = torch.zeros(*output_shape)

            for i, q in enumerate(qs_list):
                aux_f32, aux_bf16 = self.perturbation_norm(q=q, n_calls=n_calls)
                delta_f32[i, :]   = aux_f32
                delta_bf16[i, :]  = aux_bf16

            fname = DEFAULT_FNAME + "_q.png"

        delta_growth_plots(delta_f32, delta_bf16, 
                            qs_list=qs_list, 
                            fname=fname)
        print(f"Plot saved to {fname}")



class errorPlots:

    def __init__(self, model_name: str, seed:int, n_samples:int=1):
    
        random.seed(seed)
    
        self.INFTY  = torch.tensor(torch.inf)
        self.DTYPES = [torch.bfloat16, torch.float32]

        self.n_samples  = n_samples
        self.func_names = ["random",
                           f"{model_name}_clf",
                           model_name
                           ]
                
        if model_name.upper().startswith("EFF"):
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).eval()
            self.W     = torch.transpose(self.model.classifier[1].weight.data, 0, 1)
        else:
            self.model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1).eval()
            self.W     = torch.transpose(self.model.fc.weight.data, 0, 1)
            
        self.n_latent = self.W.shape[0]
        self.W0 = torch.randn(self.n_latent, self.n_latent)

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
            _y = y_diff.item()

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
            mat_cpu     = [None] + weights_cpu
            mat_gpu     = [None] + weights_gpu
            tensor_prod = True
              
        elif isinstance(func, nn.Module):
            tensor_prod = False

    
        Y  = torch.zeros(n_calls)

        for i in range(n_calls):

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

            Y[i] = _err

        return Y#.unsqueeze(0)

    def max_err(self, X, func_name, idx=None, n_calls=256):

        y_ = self.compute_arch_diff(X, func_name, indices=idx, 
                                    n_calls=n_calls)

        return torch.max(y_, dim=0).values.item()


    def error_distribution(self, n_calls, func_names=None, verbose=False):

        if isinstance(func_names, str):
            func_names = [func_names]
                  
        func_aux = self.func_names if not func_names else func_names

        shape = (len(func_aux),
                len(self.DTYPES),
                self.n_samples,
                n_calls
        )

        y_dist = torch.zeros(*shape)

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
                    Y  = self.compute_arch_diff(x, name, n_calls=n_calls)

                    if verbose and dtype==torch.bfloat16:
                        y_max = torch.max(Y).values().item
                        if y_max > 0:
                            print("Found something my guy :)")

                    y_dist[k, j, i, :] = Y.ravel()

        return y_dist
    

    def plot_distributions(self, y_dist):

        fname = "dtype_model_err.png"

        plot_error_histograms(y_dist,  self.func_names, self.DTYPES, fname=fname)


    def plot_err_signals(self, y_hist, funcname):

        idx = self.func_names.index(funcname)
        err_bf16 = y_hist[idx, 0, :]
        err_f32  = y_hist[idx, 1, :]
        
        fname = f"error_signals_{funcname}.png"
        
        iter_error((err_bf16, err_f32),
                        show_zero=False,
                        fname=fname)


    def stats_plots(self, y_dist:torch.Tensor, func_names):

        assert y.shape[0]==len(func_names), "Functions and samples do not match!!!"

        for y, name in list(zip(y_dist, func_names)):

            err_bf16 = y[0, :]
            err_fp32 = y[1, :]
            
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
            


class qErrorPlots(errorPlots):

    def __init__(self, model_name: str, seed:int, list_q, idx_samples:int=30):

        super().__init__(model_name, seed)

        self.list_q      = list_q
        self.idx_samples = idx_samples
        # random.seed(seed)
    
        # self.INFTY  = torch.tensor(torch.inf)
        # self.DTYPES = [torch.bfloat16, torch.float32]

        # self.n_samples  = n_samples
        # self.func_names = ["random",
        #                    f"{model_name}_clf",
        #                    model_name
        #                    ]
                
        # if model_name.upper().startswith("EFF"):
        #     self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).eval()
        #     self.W     = torch.transpose(self.model.classifier[1].weight.data, 0, 1)
        # else:
        #     self.model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1).eval()
        #     self.W     = torch.transpose(self.model.fc.weight.data, 0, 1)
            
        # self.n_latent = self.W.shape[0]
        # self.W0 = torch.randn(self.n_latent, self.n_latent)

        # self.init_func_dict()
        
        # # Experiment inputs
        # self.X0    = torch.randn(self.n_latent, self.n_latent)
        # self.X_img = torch.randn(1, 3, 224, 224)
    
    def sample_entries(self, X, n_q=1):
        
        flat_idx = torch.randint(X.numel(), 
                                     (n_q,)
                                     )
        
        return (flat_idx // self.n_latent, flat_idx % self.n_latent)


    def max_error_q(self, X, func_name, n_calls=256):

        n_entries = X.numel()
        y_stats   = {}

        for q in tqdm(self.list_q):

            n_q  = int(q*n_entries)
            y_max = torch.zeros(self.idx_samples)

            for i in range(self.idx_samples):

                idx      = self.sample_entries(X, n_q)
                print(idx)
                y_max[i] = self.max_err(X, func_name, idx, n_calls)
                # y   = self.compute_arch_diff(X, func_name, 
                #                              idx,
                #                              n_calls=n_calls)
            
            y_stats.update({f"{q}": y_max})

        return y_stats


    def stats_plots_q(self, n_calls, func_names=None):

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

            y_stats_bf16 = self.max_error_q(X_bf16, name, n_calls)
            y_stats_fp32 = self.max_error_q(X_fp32, name, n_calls)
            plot_bf16    = dict_to_plotting_data(y_stats_bf16)
            plot_fp32    = dict_to_plotting_data(y_stats_fp32)

            max_bf16 = self.max_err(X_bf16, name, n_calls=n_calls)
            max_fp32 = self.max_err(X_fp32, name, n_calls=n_calls)
            max_errs  = (max_bf16, max_fp32)

            fname=f"max_errors_{name}_q.png"
            q_error_distributions([plot_bf16, plot_fp32], max_errs, 
                                  fname=fname)

        
    

if __name__=="__main__":

    seed = 161
    model_name = "resnet"

    LATENT_DIM = 512
    N_CALLS    = 1024
    N_CALLS_Q  = 256
    Q_LIST     = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1]

    # Print CPU and GPU DATA
    print_sys_specs()

    # GROWTH OF DELTA
    delta_plots = deltaNormPlots(n_latent=LATENT_DIM, seed=seed)

    delta_plots.perturbation_size_plots(N_CALLS)
    delta_plots.perturbation_size_plots(N_CALLS_Q,  qs_list=Q_LIST)


    # DTYPE - MODEL HISTOGRAMS
    funcname = f'{model_name}_clf'
    
    err_plots = errorPlots(model_name=model_name, seed=seed)
    y_hist    = err_plots.error_distribution(N_CALLS)

    err_plots.plot_distributions(y_hist)
    err_plots.plot_err_signals(y_hist, funcname)


    # CUMULATIVE ERROR DISTRIBUTIONS
    from_cache = True
    pkl_name   = "error_dist.pkl"
        
    N_SAMPLES  = 30

    acc_err_plots = errorPlots(model_name=model_name, 
                               seed=seed, 
                               n_samples=N_SAMPLES)
                      
    if not from_cache:
        y_dist = acc_err_plots.error_distribution(N_CALLS, func_names=funcname)
        with open(pkl_name, 'wb') as f:
                    pickle.dump(y_dist, f)
    else:
        print(f"Reading experiment results from {pkl_name}")
        with open(pkl_name, 'rb') as f:
            y_dist = pickle.load(f)
    
    acc_err_plots.stats_plots(y_dist, func_names=funcname)

    # MAX ERROR DISTRIBUTIONS FOR Q
    err_plots_q = qErrorPlots(model_name=model_name, 
                              seed=seed,
                              list_q=Q_LIST, 
                              idx_samples=N_SAMPLES)

    err_plots_q.stats_plots_q(N_CALLS_Q, func_names=funcname)