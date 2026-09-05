import sys
import time
import pickle
import os

import random
import torch
import numpy as np
import torchvision.models as models

from scipy.stats import mannwhitneyu
from tqdm import tqdm

from adv_matrix import AdvPerturbation #,_nextafter
from simulated_annealing import SimulatedAnnealingSearch
from genetic_algorithm import AdversarialGeneticAlgorithm
from utils.utils import save_dict_to_pickle #plot_max, annealing_plot, track_evol_, pad_to_match

DRIVE_DIR = "/content/drive/MyDrive/exp_results"

# ----------------------------------------------------------------------
# CLI args
# ----------------------------------------------------------------------
n_samples = int(sys.argv[1])
data      = sys.argv[2]

SEED       = 161
model_name = "ResNet"

random.seed(SEED)
torch.manual_seed(SEED)

dtype = torch.bfloat16 if data.upper().startswith("BF") else torch.get_default_dtype()

# ----------------------------------------------------------------------
# Model / weight matrix setup
# ----------------------------------------------------------------------
if model_name.upper().startswith("EFF"):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).eval()
    W = torch.transpose(model.classifier[1].weight.data, 0, 1).to(dtype)
else:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    W = torch.transpose(model.fc.weight.data, 0, 1).to(dtype)

n_latent = W.shape[0]
n_input  = n_latent

func     = W
func_gpu = func.to('cuda')

MAX_CALLS = 32
C = 1_000

q = 0.1

# ----------------------------------------------------------------------
# Algorithm configurations
# ----------------------------------------------------------------------
L0            = 5
sa_params     = list(zip([0.8, 0.85], [1.04, 1.035]))  # (alpha, beta)
ga_pop_sizes  = [50, 100]

results_file =  f"results_{model_name}_{data}_{SEED}.pkl"
results_file_drive = os.path.join(DRIVE_DIR,
                                  results_file)

# Directory where per-sample raw tensors get streamed to disk, instead of
# being held in memory for the whole run.
TENSOR_DIR = f"tensor_data_{model_name}_{data}_{SEED}"
TENSOR_DIR = os.path.join(DRIVE_DIR, TENSOR_DIR)
# os.makedirs(TENSOR_DIR, exist_ok=True)

# results[algorithm_name] -> list of per-sample records (lightweight: no
# raw tensors, just scalars/paths).
results = {}

# y_hist used to be an in-memory torch.cat accumulator. Now we just track
# which per-sample files (already written by save_tensor for RANDOM) make
# it up, and periodically flush that manifest to disk instead of holding
# growing tensor data in RAM.
Y_HIST_MANIFEST_FILE = f"y_hist_manifest_{model_name}_{data}_{SEED}.pkl"
Y_HIST_MANIFEST_FILE = os.path.join(DRIVE_DIR, Y_HIST_MANIFEST_FILE)
y_hist_paths = []


def save_y_hist_manifest():
    with open(Y_HIST_MANIFEST_FILE, "wb") as f:
        pickle.dump(y_hist_paths, f)


def save_tensor(algorithm_name: str, sample_idx: int, tensor: torch.Tensor) -> str:
    """Write a sample's raw output tensor to its own file and return the path."""
    fname = f"{algorithm_name}_sample{sample_idx}.pt"
    fpath = os.path.join(TENSOR_DIR, fname)
    torch.save(tensor, fpath)
    return fpath


def record(algorithm_name:str, 
           sample_idx:int, 
           elapsed:float,
           target:float,
           results_data: torch.Tensor, 
           final_result:float, 
           calls:list,
           budget:int):
    data_path = save_tensor(algorithm_name, sample_idx, results_data)
    results.setdefault(algorithm_name, []).append({
        "sample":        sample_idx,
        "time_sec":      elapsed,
        "data_path":     data_path,
        "max_err":       final_result,
        "max_err_pct" :  final_result/target,
        "calls":         calls,
        "budget_calls":  budget,
    })


# def save_results():
#     with open(results_file, "wb") as f:
#         pickle.dump(results, f)


# ----------------------------------------------------------------------
# Main sampling loop
# ----------------------------------------------------------------------
for sample_idx in tqdm(range(n_samples), desc="Sampling matrices"):

    X = torch.randn(n_input, n_latent, dtype=dtype)

    # ---------------- RANDOM (benchmark) ----------------
    algorithm_name = "RANDOM"
    benchmark  = AdvPerturbation(X, func,
                                c=C,
                                func_gpu=func_gpu,
                                max_calls=MAX_CALLS)
    target_err = benchmark.full_perturbation_err

    print(f"Running benchmark")
    start_t = time.time()
    y, idx  = benchmark.random_perturbation()
    elapsed = time.time() - start_t

    max_err, _ = torch.max(torch.abs(y), dim=1)
    max_err = max_err.item()
    budget = C

    record(algorithm_name, sample_idx, elapsed, target_err,
           y, max_err,
           [], budget)

    # y is already written to disk by record()/save_tensor above; just
    # track its path so the full history can be reconstructed later
    # without keeping every sample's tensor in memory.
    y_hist_paths.append(results[algorithm_name][-1]["data_path"])
    save_y_hist_manifest()

    # ---------------- SIMULATED ANNEALING ----------------
    for alpha, beta in sa_params:
        algorithm_name = f"SA_{int(100 * alpha)}"
        sim_anneal = SimulatedAnnealingSearch(X, 
                                              func,
                                              func_gpu=func_gpu,
                                              c=C, 
                                              q=q, 
                                              max_calls=MAX_CALLS)

        
        print(f"Running {algorithm_name}")
        start_t = time.time()
        y, _, _, _ = sim_anneal.search(L0, 
                                       alpha=alpha, 
                                       beta=beta,
                                       early_stopping=True)
        elapsed = time.time() - start_t

        max_err = y[-1].item()
        budget  = sim_anneal.ulp_calls[-1]

        record(algorithm_name, sample_idx, elapsed, target_err,
               y, max_err,
               sim_anneal.ulp_calls, 
               budget)
        

    # ---------------- GENETIC ALGORITHM ----------------
    for pop_size in ga_pop_sizes:
        algorithm_name = f"GA_{pop_size}"
        gen_algorithm = AdversarialGeneticAlgorithm(
            X, func,
            func_gpu=func_gpu,
            q=q,
            c=C,
            max_calls=MAX_CALLS,
            pop_size=pop_size,
        )

        print(f"Running {algorithm_name}")
        start_t = time.time()
        gen_algorithm.search()
        elapsed = time.time() - start_t

        _, max_err = gen_algorithm.fitness[-1][0]
        budget    = gen_algorithm.ulp_calls[-1]
        ulp_calls = list(gen_algorithm.ulp_calls)  # copy for clarity; release() only clears population/fitness/history
        y = gen_algorithm.track_max()

        record(algorithm_name, sample_idx, elapsed, target_err,
               y, max_err, 
               ulp_calls, budget)

        # Free this instance's population/fitness history and GPU weight
        # copy now, rather than waiting on refcounting/cyclic GC to get to
        # it before the next pop_size's instance is created.
        gen_algorithm.release()
        del gen_algorithm

    # Save after every sample so partial progress isn't lost on a crash.
    save_dict_to_pickle(results, results_file)
    save_dict_to_pickle(results, results_file_drive)

print(f"Saved results for {n_samples} samples across "
      f"{len(results)} algorithms to {results_file}")

############################################
##         MANN-WHITNEY U TEST
############################################
# Compares each algorithm's max_err distribution (across the n_samples runs)
# against the RANDOM benchmark, testing whether it tends to find larger
# perturbation errors (alternative='greater').

benchmark_name = "RANDOM"
benchmark_err   = [r["max_err"]      for r in results[benchmark_name]]
benchmark_calls = [r["budget_calls"] for r in results[benchmark_name]]
mean_benchmark_calls = np.mean(benchmark_calls)

mwu_results = {}

for algorithm_name, records in results.items():
    if algorithm_name == benchmark_name:
        continue

    alg_err   = [r["max_err"]      for r in records]
    alg_calls = [r["budget_calls"] for r in records]

    stat, pv = mannwhitneyu(alg_err, benchmark_err, alternative="greater")
    calls_as_pct = 100 * (np.mean(alg_calls) / mean_benchmark_calls - 1)

    mwu_results[algorithm_name] = {
        "u_statistic":  stat,
        "p_value":      pv,
        "calls_vs_random_pct": calls_as_pct,
    }

    print(f"{algorithm_name}: p-value = {pv:.3e} -- ULP calls = {calls_as_pct:.2f}%")

results["_mann_whitney_vs_random"] = mwu_results

save_dict_to_pickle(results, results_file)
save_dict_to_pickle(results, results_file_drive)