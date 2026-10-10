import torch
import math
import numpy as np

#----------------------------------------------------------------
#       OBJECTIVE FUNCTIONS
#--------------------------------------------------------
def vector_distance(x: torch.Tensor, y: torch.Tensor,
                    ord: float = np.inf, batch: bool = False):
    diff = x - y
    if batch:
        # one error per sample: reduce over everything except dim 0
        return torch.linalg.vector_norm(diff.flatten(1), ord=ord, dim=1)  # (N,)
    
    return torch.linalg.vector_norm(diff, ord=ord)               # (1, )


#------------------------------------------------------------------
#       STATISTICS FOR WEIGHT ATTACK SCORING
#-----------------------------------------------------------------

def net_success_pct(e:torch.Tensor, e0:torch.Tensor) -> float:
    frac = (e > e0).float().mean() - (e < e0).float().mean() # Works because both means have the same denominator
    gain = ((e - e0).mean() / (e0.mean() + 1e-30))
    return frac.item() + 1e-3 * gain.item()
    
def cvar_low(e:torch.Tensor, e0:torch.Tensor, alpha:float=0.1) -> float:
    """
    Calculates the mean of the bottom alpha(default=10%) differences 
    """
    diff = e-e0
    k = max(1, math.ceil(alpha * diff.shape[0]))
    low_quantiles = torch.topk(diff, k, dim=0, largest=False).values
    return low_quantiles.mean().item()

def mean_log_ratio(e:torch.Tensor, e0:torch.Tensor, eps:float=1e-15) -> float:

    log_ratio = torch.log(e.clamp_min(eps)) - torch.log(e0.clamp_min(eps))
    return log_ratio.mean().item()


