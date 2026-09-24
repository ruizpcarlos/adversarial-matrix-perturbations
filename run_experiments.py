import time
import pickle
import os
import copy
import argparse

import random
import torch    
from tqdm import tqdm

from adv_matrix import AdvPerturbation #,_nextafter
from simulated_annealing import SimulatedAnnealingSearch
from genetic_algorithm import AdversarialGeneticAlgorithm
from utils.utils import save_dict_to_pickle, load_model_and_weights, adjusted_pvals
from utils.cli import add_common_args, DTYPES

DRIVE_DIR = "/content/drive/MyDrive/exp_results"

# ----------------------------------------------------------------------
# CLI args
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
add_common_args(parser, dtype=True, weighted=True, matmul=True)
args = parser.parse_args()

SEED       = 161
model_name = "ResNet"

n_samples = args.n_samples
dtype     = DTYPES[args.dtype]
WEIGHTED  = args.weighted
MATMUL    = args.matmul

# Tag used in output names so weighted / unweighted / fp32 / bf16 runs never overwrite each other
DTYPE_TAG = "bf16" if dtype == torch.bfloat16 else "fp32"
RUN_TAG   = f"{model_name}" + ("_W" if MATMUL else "")  + f"_{DTYPE_TAG}_{SEED}" + ("_weighted" if WEIGHTED else "")

random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# Model / weight matrix setup
# ----------------------------------------------------------------------
model, W = load_model_and_weights(model_name, dtype=dtype)

n_latent = W.shape[0]
n_input  = n_latent

func     = W if MATMUL else model
func_gpu = func.to('cuda') if MATMUL else copy.deepcopy(func).eval().to("cuda")

MAX_CALLS = 32
C = 1_000

q = 0.1

# ----------------------------------------------------------------------
# Algorithm configurations
# ----------------------------------------------------------------------
L0            = 5
sa_params     = list(zip([0.8, 0.85], [1.04, 1.035]))  # (alpha, beta)
ga_pop_sizes  = [50, 100]

results_file =  f"results_{RUN_TAG}.pkl"
results_file_drive = os.path.join(DRIVE_DIR,
                                  results_file)

# Directory where per-sample raw tensors get streamed to disk, instead of
# being held in memory for the whole run.
TENSOR_DIR = f"tensor_data_{RUN_TAG}"
TENSOR_DIR = os.path.join(DRIVE_DIR, TENSOR_DIR)
# os.makedirs(TENSOR_DIR, exist_ok=True)

# results[algorithm_name] -> list of per-sample records (lightweight: no
# raw tensors, just scalars/paths).
results = {}

# y_hist used to be an in-memory torch.cat accumulator. Now we just track
# which per-sample files (already written by save_tensor for RANDOM) make
# it up, and periodically flush that manifest to disk instead of holding
# growing tensor data in RAM.
Y_HIST_MANIFEST_FILE = f"y_hist_manifest_{RUN_TAG}.pkl"
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
                                q = q,
                                func_gpu=func_gpu,
                                max_calls=MAX_CALLS,
                                budget_calls=C, 
                                weighted_sampling=True) # WEIGHTED arg is used for SimAnneal/GenAlgos
    target_err = benchmark.baseline_err

    print(f"Running benchmark")
    start_t = time.time()
    y, _  = benchmark.random_perturbation(early_stopping=True)
    elapsed = time.time() - start_t

    max_err, _ = torch.max(y, dim=1)
    max_err = max_err.item()
    budget = y.shape[1] / MAX_CALLS

    record(algorithm_name, sample_idx, elapsed, target_err,
           y, max_err,
           [], budget)

    y_hist_paths.append(results[algorithm_name][-1]["data_path"])
    save_y_hist_manifest()

    # ---------------- RANDOM (benchmark) ----------------
    algorithm_name = "RANDOM_W"
    print(f"Running weighted benchmark")
    start_t = time.time()
    y, _  = benchmark.random_perturbation(weighted=True,
                                            early_stopping=True)
    elapsed = time.time() - start_t
    
    max_err, _ = torch.max(y, dim=1)
    max_err = max_err.item()
    budget = y.shape[1] / MAX_CALLS
    
    record(algorithm_name, sample_idx, elapsed, target_err,
            y, max_err,
            [], budget)
    
    y_hist_paths.append(results[algorithm_name][-1]["data_path"])
    save_y_hist_manifest()

    # ---------------- SIMULATED ANNEALING ----------------
    for alpha, beta in sa_params:
        algorithm_name = f"SA_{int(100 * alpha)}"
        sim_anneal = SimulatedAnnealingSearch(X, 
                                            func,
                                            q=q,
                                            func_gpu=func_gpu, 
                                            max_calls=MAX_CALLS,
                                            budget_calls=C, 
                                            weighted_sampling=WEIGHTED)

        
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
            q=q,
            func_gpu=func_gpu,
            max_calls=MAX_CALLS,
            budget_calls=C,
            pop_size=pop_size, 
            weighted_sampling=WEIGHTED
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
##         WILCOXON PAIRED TEST
############################################
# Compares each algorithm's max_err distribution (across the n_samples runs)
# against the RANDOM benchmark, testing whether it tends to find larger
# perturbation errors (alternative='greater').

benchmark_name = "RANDOM"
# benchmark_err   = [r["max_err"]      for r in results[benchmark_name]]
# benchmark_calls = [r["budget_calls"] for r in results[benchmark_name]]
# mean_benchmark_calls = np.mean(benchmark_calls)

stat_results = adjusted_pvals(results, 
                              baseline_key="RANDOM", 
                              alternative="greater")

# for algorithm_name, records in results.items():
#     if algorithm_name == benchmark_name:
#         continue

#     alg_err   = [r["max_err"]      for r in records]
#     alg_calls = [r["budget_calls"] for r in records]

#     stat, pv = wilcoxon(alg_err, benchmark_err, alternative="greater")
#     # calls_as_pct = 100 * (np.mean(alg_calls) / mean_benchmark_calls - 1)

#     stat_results[algorithm_name] = {
#         "u_statistic":  stat,
#         "p_value":      pv,
#     }
for algorithm_name, pvals in stat_results.items():
    print(f"{algorithm_name}: adj p-value = {pvals['p_adj']:.3e}")

results["_wilcoxon_vs_random"] = stat_results

save_dict_to_pickle(results, results_file)
save_dict_to_pickle(results, results_file_drive)