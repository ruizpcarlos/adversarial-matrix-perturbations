import sys
import time
import itertools

import torch
import random
import numpy as np
from tqdm import tqdm

import torchvision.models as models

from adv_matrix import AdvPerturbation
from utils.utils import save_dict_to_pickle


class SimulatedAnnealingSearch(AdvPerturbation):

    def __init__(self,
                 input_matrix: torch.Tensor,
                 func,
                 c,
                 q,
                 func_gpu=None,
                 max_calls = 32):

        super().__init__(input_matrix, func, c, func_gpu, max_calls)

        self.q  = q
        self._q = int(q*input_matrix.numel())
        self.T0 = None


    def random_geneset(self):
        """Random genome via rejection sampling — better when total >> k."""
        chosen = set()
        while len(chosen) < self._q:
            chosen.add(random.randrange(self.total))
        return np.array(sorted(chosen), dtype=np.int64)


    def generate_new_sol(self, index):

        x_str = self.indices_to_geneset(*index)
        # x1 = self.recombine(x_str)
        x1, _ = self.recombine(x_str, self.random_geneset())
        x1 = self.mutate_geneset(x1)

        return self.geneset_to_indices(x1)


    def init_temp(self, target_prob=0.9, n_samples=100, tie_penalty=1e-8, verbose=False):
        """
        Calculates a initial temperature that will allow q% of 
        'worse' solutions to be accepted
        """

        obj_delta = np.zeros(n_samples)
        idx       = self._sample_entries(self._q)
        n_calls, err    = self.compute_max_err(idx)

        if verbose:
            pbar = tqdm(range(n_samples)) 
            pbar.set_description(f"Sampling {n_samples} moves to set init temperature")
        else:
            pbar = range(n_samples)

        for i in pbar:
            idx  = self.generate_new_sol(idx)

            _n_c, _err   = self.compute_max_err(idx)
            delta_err    = err-_err
            delta_calls  = n_calls - _n_c
            obj_delta[i] = delta_err - tie_penalty * delta_calls

            err     = _err
            n_calls = _n_c

        worse_moves = obj_delta[obj_delta<0]

        T0  = np.mean(worse_moves)/np.log(target_prob)

        self.T0 = T0
        # return T0


    def search(self, L0, *,
               alpha = 0.8,
               beta = 1.04, 
               tol = 1e-15,
               tie_penalty = 1e-8,
               early_stopping = True,
               verbose=False):

        if self.T0 is None:
            self.init_temp(verbose=verbose) # Initialize temperature

        T = self.T0

        self.compute_max_err.reset()
        self.ulp_calls = [0]

        k       = 0
        counter = 0
        L       = L0

        iter_idx             = self._sample_entries(self._q)
        iter_calls, iter_err = self.compute_max_err(iter_idx)

        # chains       = [L]
        temps        = [T]
        accept_probs = [0.9]
        Y            = [iter_err]
        # self.ulp_calls.append(self.compute_max_err.call_count)

        best_idx   = iter_idx
        best_err   = iter_err
        best_calls = iter_calls
        budget_exhausted = False

        if verbose:
            print(f"Starting search w/ T0 = {T:.3e}")

        while (T > tol and counter < 25
               and self.compute_max_err.call_count < self.c
               ):

            accept_prob = None

            for _ in range(L):

                if self.compute_max_err.call_count >= self.c:
                    budget_exhausted = True
                    break
                
                idx = self.generate_new_sol(iter_idx)
                n_calls, _err = self.compute_max_err(idx)
                _err = abs(_err)

                delta_err   = _err - iter_err
                delta_calls = n_calls - iter_calls
                effective_delta = delta_err - tie_penalty * delta_calls

                if effective_delta >= 0:
                    iter_err   = _err
                    iter_idx   = idx
                    iter_calls = n_calls
                else:
                    accept_prob = np.exp(effective_delta / T)
                    if random.uniform(0, 1) < accept_prob:
                        iter_err   = _err
                        iter_idx   = idx
                        iter_calls = n_calls

                if (_err > best_err
                    or (_err == best_err and n_calls < best_calls)
                    ):
                    best_idx   = idx
                    best_err   = _err
                    best_calls = n_calls

            self.ulp_calls.append(self.compute_max_err.call_count)
            Y.append(iter_err)
            accept_probs.append(accept_prob) 

            if early_stopping:
                if best_err > getattr(self, "_last_best_err", -np.inf):
                    counter = 0
                else:
                    counter += 1
                self._last_best_err = best_err

            k += 1
            T = alpha*T
            temps.append(T)

            L = int(L0*(beta**k))
            # chains.append(L)

            if verbose and k%10==0:
                msg = f"k = {k} ({counter}): max error = {best_err:.3e} , ulp_calls = {best_calls} -- temp={T:.3e}"
                if accept_prob is not None:
                    msg += f", p<{accept_prob:.3e}"
                print(msg)

            if budget_exhausted:
                break

        if verbose:
            print(f"Terminated in {len(Y)} iterations w/ error = {best_err:.4e}")

        Y = torch.Tensor(Y)#.unsqueeze(0)
        solution = (best_idx, best_calls)

        return Y, solution, temps, accept_probs 
    


if __name__ == "__main__":

    n_test = int(sys.argv[1])   # number of repeated runs per (pop_size, p) configuration
    data   = sys.argv[2]

    model_name = "ResNet"
    seed       = 420
    dtype      = torch.bfloat16 if data.upper().startswith("BF") else torch.float32

    random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # EXPERIMENT INPUTS
    # ------------------------------------------------------------------
    if model_name.upper().startswith("EFF"):
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).eval()
        W     = torch.transpose(model.classifier[1].weight.data, 0, 1).to(dtype)
    else:
        model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1).eval()
        W     = torch.transpose(model.fc.weight.data, 0, 1).to(dtype)
                    
    n_latent = W.shape[0]

    W0    = torch.randn(n_latent, n_latent,
                            dtype=dtype)  
    X     = torch.randn(n_test, n_latent, n_latent,
                            dtype=dtype)
    X_img = torch.randn(n_test, 1, 3, 224, 224,
                            dtype=dtype)

    MAX_CALLS      = 32 # if data.upper().startswith("BF") else 128
    c              = 1000  # total budget of calls to compute_max_err

    # ------------------------------------------------------------------
    # Grid search over param combinations and perturbation fraction
    # ------------------------------------------------------------------
    alphas   = [0.8, 0.85, 0.9]
    betas    = [1.04, 1.035, 1.03]
    params   = list(zip(alphas, betas))
    q_values = [0.05, 0.1]

    results = []
    fname   = f"sa_gridsearch_{model_name}_{data}_{seed}.pkl"

    targets      = []
    sa_instances = {}

    print(f"Computing target errors of the sample ({dtype})")
    for j, _x in enumerate(tqdm(X)):
        for q in q_values:
            targ_aux = SimulatedAnnealingSearch(
                                            input_matrix=_x, 
                                            func=W,
                                            c=c,
                                            q=q,
                                            max_calls=MAX_CALLS)
            targ_aux.init_temp(verbose=False)
            sa_instances.update({(j, q) : targ_aux})
        err = targ_aux.full_perturbation_err
        targets.append(err)

    for ab, q in itertools.product(params, q_values):
        print(f"Running test for params={ab}, q={q:.2f}")

        alpha, beta = ab

        run_errs      = []
        run_calls     = []
        run_times     = []
        run_err_ratio = []
        run_iters     = []

        for trial in range(n_test):
            # X_test     = X[trial]
            # target_err = targets[trial]

            target_err = targets[(trial, q)]

            print(f"{trial+1} - Full matrix perturbation = {target_err:.4e}")

            adv_sa = sa_instances[(trial, q)]

            # adv_sa = SimulatedAnnealingSearch(
            #                             input_matrix=X_test, 
            #                             func=W,
            #                             c=c,
            #                             q=q,
            #                             max_calls=MAX_CALLS)

            start_t = time.time()
            Y, sol, _, _ =  adv_sa.search(L0=5,
                                        alpha=alpha,
                                        beta=beta,
                                        early_stopping=True, 
                                        verbose=False)
            elapsed = time.time() - start_t

            best_err   = Y[-1].item()
            best_calls = sol[1]

            err_pct = abs(best_err)/target_err

            run_errs.append(abs(best_err))
            run_err_ratio.append(err_pct)          
            run_calls.append(adv_sa.ulp_calls[-1])
            run_times.append(elapsed)
            run_iters.append(Y.shape[0])

            print(# f"pop_size={pop_size:>4} | q={q:>5.2f} |"
                    f" trial={trial+1}/{n_test} | iters={Y.shape[0]} | "
                    f"best_err={best_err:.4e} ({100*err_pct:.2f}%) | "
                    f"ulp_calls={best_calls:>6} | "
                    f"budget spent = {adv_sa.ulp_calls[-1]} ({100*(adv_sa.ulp_calls[-1]/c):.2f}%)  in {elapsed:.2f}s")

        run_errs      = np.array(run_errs)
        run_err_ratio = np.array(run_err_ratio)
        success_pct   = (run_err_ratio >= 1).sum()/n_test

        results.append({
                "params":         ab,
                "q":              q,
                "n_test":         n_test,
                "target_err":     np.array(targets),
                # "mean_err":       run_errs.mean(),
                "std_err":        run_errs.std(),
                "success_pct":    success_pct,
                "mean_err_pct":   run_err_ratio.mean(),
                "err_dist":       run_errs,
                "mean_ulp_calls": float(np.mean(run_calls)),
                "mean_time_s":    float(np.mean(run_times)),
                "mean_iters":      float(np.mean(run_iters)),
            })

        save_dict_to_pickle(results, filename=fname)

        print(f"  -> mean err% ={run_err_ratio.mean():.4e} (std={run_errs.std():.4e}) "
                f"{(100*success_pct):.2f}% success over {n_test} trials\n")

    # ------------------------------------------------------------------
    # Best configuration (highest mean max error achieved)
    # ------------------------------------------------------------------
    best = max(results, key=lambda r: r["mean_err_pct"])
    print("Best configuration:")
    print(f"  params = {best['params']}, q = {best['q']} "
            f"-> {(100*success_pct):.2f}% success w/ mean_err = {best['mean_err_pct']:.4e}"
            f" (std = {best['std_err']:.4e}) "
            f"over {n_test} trials "
            f"(mean ulp_calls = {best['mean_ulp_calls']:.0f}, "
            f"mean time = {best['mean_time_s']:.2f}s)")
