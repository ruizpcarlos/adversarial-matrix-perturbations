import random
import torch
import sys
import pickle
import numpy as np
import torchvision.models as models

from scipy.stats import mannwhitneyu
from tqdm import tqdm

from adv_matrix import AdvPerturbation, _nextafter
from simulated_annealing import SimulatedAnnealingSearch
from evol_algorithms import AdversarialGeneticAlgorithm
from utils import save_dict_to_pickle #plot_max, annealing_plot, track_evol_, pad_to_match

_model    = sys.argv[1]
_data     = sys.argv[2]
N_samples = int(sys.argv[3])
seed      = int(sys.argv[4])
random.seed(seed)

dtype    = torch.bfloat16 if _data.startswith("bfloat") else torch.get_default_dtype()
type_aux = f"{dtype}".split('.')[1]

if _model.upper().startswith("EFF"):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # model.eval()
    W = torch.transpose(model.classifier[1].weight.data, 0, 1).to(dtype)
else:
    model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
    # model.eval()
    W = torch.transpose(model.fc.weight.data, 0, 1).to(dtype)

N_INPUT  = 1
SCALE    = -2
_entries = 10.0**SCALE
n_latent = W.shape[0]
M        = _entries*torch.randn(N_INPUT, n_latent, dtype=dtype)
weights  = W

p = 150
MAX_ITER = 256
print(f"MAX ULP CALLS = {p*MAX_ITER}")

results      = {}
alg_calls    = {}
max_trackers = {}

##############################
#     SIMULATED ANNEALING
##############################
sim_anneal =  SimulatedAnnealingSearch(M, weights,
                                       p,
                                       max_iter=MAX_ITER)
# Optimization params
L = 5
c = 1e-6
cooling_rates = [0.8, 0.85, 0.9]

for rate in cooling_rates:
    key_str = f"SA_{int(100*rate)}"

    results[key_str] = []
    _calls           = []

    print(f"Running {key_str}")
    for _ in tqdm(range(N_samples)):
        y, _, aux_calls = sim_anneal.search(c, L, cooling_r = rate)
        n_calls         = aux_calls[-1]

        results[key_str].append(
                                y.max().item()
                                )
        _calls.append(n_calls)

    alg_calls[key_str] = np.mean(_calls)

res_filename   = f"results_{type_aux}_{seed}.pkl"
calls_filename = f"alg_calls_{type_aux}_{seed}.pkl"

save_dict_to_pickle(results, res_filename)
save_dict_to_pickle(alg_calls, calls_filename)

########################################################
###         GENETIC ALGORITHMS
########################################################
mating_pct = 0.2
pop_sizes  = [25, 50, 76]

for pop_ in pop_sizes:

    key_str = f"GA_{pop_}"

    results[key_str] = []
    _calls = []

    print(f"Running {key_str}")
    for _ in tqdm(range(N_samples)):

        gen_algorithm = AdversarialGeneticAlgorithm(M, weights, p,
                                            max_iter=MAX_ITER,
                                            pop_size=pop_,
                                            mating_pct=mating_pct
                                            )

        gen_algorithm.search()

        results[key_str].append(
                                gen_algorithm.fitness[-1][0][1]
                                )

        _calls.append(n_calls)

    alg_calls[key_str] = np.mean(_calls)

save_dict_to_pickle(results, res_filename)
save_dict_to_pickle(alg_calls, calls_filename)

########################################################
###        RANDOM PERTURBATION (BENCHMARK)
########################################################
key_str = f"RANDOM"

results[key_str] = []
_calls           = [] 
y_hist           = torch.Tensor([])

print(f"Running {key_str}")
benchmark = AdvPerturbation(M, weights, p, max_iter=MAX_ITER)
for _ in tqdm(range(N_samples)):

    _nextafter.reset()
    y, idx = benchmark.random_perturbation()
    y_hist = torch.cat([y_hist, y], dim=1)

    max_err, _ = torch.max(torch.abs(y), dim=1)
    max_err = max_err.item()

    results[key_str].append(max_err)
    _calls.append(_nextafter.call_count)

alg_calls[key_str] = np.mean(_calls)

# Save results
save_dict_to_pickle(results, res_filename)
save_dict_to_pickle(alg_calls, calls_filename)

############################################
##         MANN-WHITNEY U TEST
############################################
with open(res_filename, 'rb') as f:
    results = pickle.load(f)

algorithms = list(results.keys())[:-1]
benchmark  = results['RANDOM']

for a in algorithms:
    _, pv = mannwhitneyu(results[a], benchmark,
                         # method='exact',
                         alternative='greater')
    
    calls_as_pct = 100*(alg_calls[a]/alg_calls['RANDOM']-1)
    
    print(f"{a}: p-value = {pv:.3e} -- ULP calls = {calls_as_pct:.2f}%")


