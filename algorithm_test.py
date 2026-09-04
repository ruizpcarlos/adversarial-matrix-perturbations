import sys
import time
import json

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
weights  = W

MAX_CALLS = 32
C = 1_000

q = 0.1

# ----------------------------------------------------------------------
# Algorithm configurations
# ----------------------------------------------------------------------
L0            = 5
sa_params     = list(zip([0.8, 0.85], [1.04, 1.035]))  # (alpha, beta)
ga_pop_sizes  = [50, 100]

results_file = f"results_{model_name}_{data}_{SEED}.pkl"

# results[algorithm_name] -> list of per-sample records
results = {}
y_hist = torch.tensor([])

def record(algorithm_name:str, 
           sample_idx:int, 
           elapsed:float,
           target:float,
           results_data: np.array, 
           final_result:float, 
           calls:list,
           budget:int):
    results.setdefault(algorithm_name, []).append({
        "sample":        sample_idx,
        "time_sec":      elapsed,
        "data":          results_data, 
        "max_err":       final_result,
        "max_err_pct" :  final_result/target,
        "calls":         calls,
        "budget_calls":  budget,
    })


# def save_results():
#     with open(results_file, "w") as f:
#         json.dump(results, f, indent=2)


# ----------------------------------------------------------------------
# Main sampling loop
# ----------------------------------------------------------------------
for sample_idx in tqdm(range(n_samples), desc="Sampling matrices"):

    X = torch.randn(n_input, n_latent, dtype=dtype)

    # ---------------- RANDOM (benchmark) ----------------
    algorithm_name = "RANDOM"
    benchmark  = AdvPerturbation(X, weights,
                                c=C,
                                max_calls=MAX_CALLS)
    target_err = benchmark.full_perturbation_err

    print(f"Running benchmark")
    start_t = time.time()
    y, idx  = benchmark.random_perturbation()
    elapsed = time.time() - start_t

    max_err, _ = torch.max(torch.abs(y), dim=1)
    max_err = max_err.item()
    budget = C

    y_hist  = torch.cat([y_hist, y])
        
    record(algorithm_name, sample_idx, elapsed, target_err,
           y, max_err,
           [], budget)

    # ---------------- SIMULATED ANNEALING ----------------
    for alpha, beta in sa_params:
        algorithm_name = f"SA_{int(100 * alpha)}"
        sim_anneal = SimulatedAnnealingSearch(X, 
                                              weights,
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
            X, weights,
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
        budget = gen_algorithm.ulp_calls[-1]
        y = gen_algorithm.track_max()

        record(algorithm_name, sample_idx, elapsed, target_err,
               y, max_err, 
               gen_algorithm.ulp_calls, budget)

    # Save after every sample so partial progress isn't lost on a crash.
    save_dict_to_pickle(results, results_file)

torch.save(y_hist, "err_dist.pt")

print(f"Saved results for {n_samples} samples across "
      f"{len(results)} algorithms to {results_file}")