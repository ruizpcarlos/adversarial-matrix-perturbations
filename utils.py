import torch
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F


def save_dict_to_pickle(data_dict: dict, filename):

    with open(filename, 'wb') as fp:
        pickle.dump(data_dict, fp, 
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"--- Saved results in {filename}")


def plot_max(Y_max, Y_hist, labels, n_latent, p, fname, show_zero=True):

    if Y_max.dim()>2:
        raise ValueError(f"Expected 2D tensor, got shape {Y_hist.shape}")

    Y_hist = Y_hist.squeeze().ravel()

    fname = "algorithm_perf" + fname + f"_{p}.png"

    X    = range(Y_max.shape[1])
    Y_np = Y_max.numpy()

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize= [15, 5],
                                   sharey=True,
                                   gridspec_kw={'width_ratios': [3, 1]}
                                   )

    ax1.set_title(f"Error vs no. of ULP calls\n {n_latent} latent dims -  perturbed {p} entries")
    lines = ax1.plot(X, Y_max.T, linestyle="--")
    for line, name in zip(lines, labels):
        line.set_label(name)
    ax1.legend()

    n_values = Y_hist.unique().numel()

    ax2.set_title(f"Error distribution \n {n_values} unique values")
    ax2.hist(Y_hist.ravel(), density=True, bins=n_values, orientation = "horizontal")

    if show_zero:
        ax1.axhline(y=0, linestyle = "--", color = "grey")
        ax2.axhline(y=0, linestyle = "--", color = "grey")

    plt.savefig(fname)

def annealing_plot(y, y_max, opt_params, n_latent, p, fname):

    m = y.shape[0]  # number of searches
    n = y.shape[1]

    assert len(opt_params) == m, "Number of tensors and params mismatch"

    colors = plt.cm.tab10.colors  # or tab20 if m > 10

    for i in range(m):
        c = colors[i % len(colors)]
        plt.plot(range(n), y_max[i], c=c, label=f"alpha={opt_params[i]}")
        plt.plot(range(n), y[i], c=c, linestyle="dotted")

    fname = "annealingplot_" + fname + f"_{p}.png"

    plt.title(f"Simmulated annealing on {n_latent} dims\n perturbed {p} entries")
    plt.legend()
    plt.savefig(fname)

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