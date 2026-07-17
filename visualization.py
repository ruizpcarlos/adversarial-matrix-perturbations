import random
import torch
import sys
import torchvision.models as models

from adv_matrix import AdvPerturbation, _nextafter
from simulated_annealing import SimulatedAnnealingSearch
from evol_algorithms import AdversarialGeneticAlgorithm
from utils.utils import plot_max, annealing_plot, track_evol_, pad_to_match

_model    = sys.argv[1]
_data     = sys.argv[2]
seed      = int(sys.argv[3])

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

n_input  = 1
n_latent = W.shape[0]
scale    = -1
_entries = 10.0**scale
M        = _entries*torch.randn(n_input, n_latent, dtype=dtype)
weights  = W

p = 150
max_calls = 256
print(f"MAX ULP CALLS = {p*max_calls}")

max_trackers = {}

###############################################
###            SIMULATED ANNEALING
###############################################
sim_anneal =  SimulatedAnnealingSearch(M, weights, p,
                                 max_calls=max_calls)

L = 5
c = 1e-6
cooling_rates = [0.8, 0.85, 0.9]
sa_result     = {}

for rate in cooling_rates:

    key_str = f"SA_{int(100*rate)}"

    print(f"Running {key_str}")
    
    y, _, aux_calls = sim_anneal.search(c, L, 
                                        cooling_r = rate, 
                                        verbose=True)
    n_calls = aux_calls[-1]

    sa_result[key_str]    = track_evol_(y, aux_calls)
    max_trackers[key_str] = torch.cummax(sa_result[key_str],
                                         dim=1).values

y     = pad_to_match(list(sa_result.values()))
y_max = pad_to_match(list(max_trackers.values()))

annealing_plot(y, y_max, cooling_rates, n_latent, p)

########################################################
###         GENETIC ALGORITHMS
########################################################
mating_pct = 0.2
pop_sizes  = [25, 50, 74]

for _pop in pop_sizes:

    key_str = f"GA_{_pop}"

    print(f"Running {key_str}")
    
    gen_algorithm = AdversarialGeneticAlgorithm(M, weights, p,
                                            max_calls=max_calls,
                                            pop_size=_pop,
                                            mating_pct=mating_pct
                                            )

    gen_algorithm.search(verbose=True)

    max_trackers[key_str] = gen_algorithm.track_max()

########################################################
###        RANDOM PERTURBATION (BENCHMARK)
########################################################

key_str = f"RANDOM"
# y_hist = torch.Tensor([])

benchmark = AdvPerturbation(M, weights, p, max_calls=max_calls)

print(f"Running {key_str} perturbation")

_nextafter.reset()
y, idx = benchmark.random_perturbation(verbose=True)

# y_hist = torch.cat([y_hist, y], dim=1)

max_err, n_calls = torch.max(torch.abs(y), dim=1)
max_err = max_err.item()
n_calls = n_calls.item()

max_trackers[key_str] = torch.cummax(torch.flatten(y),
                                      dim=0).values.unsqueeze(0)


###########################################
###         PLOT
###########################################
y_max = pad_to_match(
          list(max_trackers.values())
          )
labels = list(max_trackers.keys())
plot_max(y_max, y, labels, n_latent, p)
