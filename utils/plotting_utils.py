import torch
import hashlib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

from torch.linalg import vector_norm, multi_dot

from matplotlib.lines import Line2D


def plot_max(Y_max, Y_hist, labels, n_latent, p, fname=None, show_zero=True):

    if Y_max.dim()>2:
        raise ValueError(f"Expected 2D tensor, got shape {Y_hist.shape}")

    Y_hist = Y_hist.squeeze().ravel()

    
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

    n_values  = Y_hist.unique().numel()
    Y_hist_np = Y_hist.numpy()
    n_bins    = min(50, max(10, int(np.sqrt(Y_hist_np.size))))  # Sturges/sqrt-style heuristic

    # _weights = np.ones_like(Y_hist_np) / Y_hist_np.size # Corrects issues w bin width/density integration
    ax2.set_title(f"Error distribution \n {n_values} unique values")
    ax2.hist(Y_hist_np, density=False, bins=n_bins, orientation = "horizontal")

    if show_zero:
        ax1.axhline(y=0, linestyle = "--", color = "grey")
        ax2.axhline(y=0, linestyle = "--", color = "grey")

    if fname is not None:
        fname = "algorithm_perf" + fname + f"_{p}.png"
        plt.savefig(fname)


def annealing_plot(y, y_max, opt_params, n_latent, p, fname=None):

    m = y.shape[0]  # number of searches
    n = y.shape[1]

    assert len(opt_params) == m, "Number of tensors and params mismatch"

    colors = plt.cm.tab10.colors  # or tab20 if m > 10

    for i in range(m):
        c = colors[i % len(colors)]
        plt.plot(range(n), y_max[i], c=c, label=f"alpha={opt_params[i]}")
        plt.plot(range(n), y[i], c=c, linestyle="dotted")


    plt.title(f"Simmulated annealing on {n_latent} dims\n perturbed {p} entries")
    plt.legend()
    if fname:
        fname = "annealingplot_" + fname + f"_{p}.png"
        plt.savefig(fname)

def delta_growth_plots(delta_f32, delta_bf16, qs_list=None, log_scale=True, fname=None):
    """
    Plots growth of the perturbation for single and half precision.

    Parameters
    ----------
    delta_f32, delta_bf16 : torch.Tensor
        Shape (n_p, 2, n_calls) — [max-norm, 2-norm] deltas per p value.
    test_p : list[float], optional
        p values used to generate each curve. If None, treats input as a
        single unlabeled run (adds a leading dim, uses black for color).
    log_scale : bool
        If True, plots max-norm on a log-scaled left axis and 2-norm on
        a separate twin axis (right).

    Returns
    -------
    fig, (ax1, ax2) : the created figure and primary axes.
    """
    X_LABEL = "ULP calls"
    COLORS  = plt.cm.tab10.colors

    if qs_list is None:
        delta_f32  = delta_f32.unsqueeze(0)
        delta_bf16 = delta_bf16.unsqueeze(0)
        COLORS = ["black"]

    assert delta_f32.shape[2] == delta_bf16.shape[2], \
        "fp32 and bf16 deltas must have the same number of ULP calls"
    PLOT_X   = range(delta_f32.shape[2])
    iter_aux = range(delta_f32.shape[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # LEFT SUBPLOT: SINGLE PRECISION
    for k in iter_aux:
        y = delta_f32[k, :]
        c = COLORS[k % len(COLORS)]

        ax1.plot(PLOT_X, y[0, :], c=c)  # Max-norm
        # target = ax1_2n if log_scale else ax1
        ax1.plot(PLOT_X, y[1, :], c=c, linestyle="--")  # 2-norm

    ax1.set_xlabel(X_LABEL)
    ax1.set_title("Single Precision (fp32)")
    ax1.set_ylabel("Delta size")
    if log_scale:
        ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.3)

    # RIGHT SUBPLOT: HALF PRECISION
    for k in iter_aux:
        y = delta_bf16[k, :]
        c = COLORS[k % len(COLORS)]

        ax2.plot(PLOT_X, y[0, :], c=c)  # Max-norm
        ax2.plot(PLOT_X, y[1, :], c=c, linestyle="--")  # 2-norm

    ax2.set_xlabel(X_LABEL)
    ax2.set_title("Half Precision (bfloat16)")
    ax2.set_ylabel("Delta size")
    if log_scale:
        ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.3)

    # --- Legends ---
    if qs_list is not None:
        color_handles = [
            Line2D([0], [0], color=COLORS[k % len(COLORS)], lw=2, label=f"q = {int(100*p)}%")
            for k, p in enumerate(qs_list)
        ]
        leg1 = ax2.legend(handles=color_handles, title="Noisy indices (%)",
                       loc="upper left", bbox_to_anchor=(1.15, 1.0),
                       borderaxespad=0.)
        ax2.add_artist(leg1)

    style_handles = [
        Line2D([0], [0], color="black", lw=2, linestyle="-",  label="Max-norm (lefty-axis, log)" if log_scale else "Max-norm"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="2-norm (right y-axis)" if log_scale else "2-norm"),
    ]
    ax2.legend(handles=style_handles, title="Norm",
               loc="upper left", bbox_to_anchor=(1.15, 0.4),
               borderaxespad=0.)

    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    plt.show()

    # return fig, (ax1, ax2)


def iter_error(Ys, show_zero:bool=True, fname:str=None):
    """
    Ys: list/tuple of 2 tensors, one per row (each 2D: [n_curves, n_steps])
    show_zero: boolean, adds a horizontal line at y=0
    fname: provides a filename, if None the plot is not saved
    """
    if len(Ys) != 2:
        raise ValueError(f"Expected 2 tensors (one per row), got {len(Ys)}")

    for Y in Ys:
        if Y.dim() > 2:
            raise ValueError(f"Expected 2D tensor, got shape {Y.shape}")

    fig, axes = plt.subplots(nrows=2, ncols=2,
                                figsize=[18, 10],
                                sharey='row',
                                gridspec_kw={'width_ratios': [3, 1]})

    for row, Y in enumerate(Ys):
        ax1, ax2 = axes[row]

        X    = range(Y.shape[1])
        Y_np = Y.cpu().numpy()

        ax1.plot(X, Y_np.T, linestyle="--")

        n_values = Y.unique().numel()

        fp  = "Half" if row==0 else "Single"
        fp2 = "bf16" if row==0 else "fp32"

        ax1.set_title(f"Errors in {fp} Precision ({fp2})")
        ax1.set_xlabel("ULP Calls")
        ax1.set_ylabel("Error")
        ax2.set_title(f"Error distribution \n {n_values} unique values")
        ax2.hist(Y_np.ravel(), bins=n_values, orientation="horizontal")

    if show_zero:
        ax1.axhline(y=0, linestyle="--", color="grey")
        ax2.axhline(y=0, linestyle="--", color="grey")

    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
        print(f"Saved error signals to {fname}")

    plt.show()


def iter_error_distributions(Ys, fname=None):
    """
    Ys: list of dictionaries of numpy arrays for plotting
    """
    if len(Ys) != 2:
        raise ValueError(f"Expected 2 tensors (one per row), got {len(Ys)}")

    for Y in Ys:
        if len(Y) > 2:
            raise ValueError(f"What is this bullshit")

    fig, axes = plt.subplots(nrows=2, ncols=2,
                              figsize=[18, 10],
                              sharey='row',
                              gridspec_kw={'width_ratios': [3, 1]})

    for row, Y in enumerate(Ys):
        ax1, ax2 = axes[row]

        df = Y[0]
        y_dist = Y[1]

        fp  = "Half" if row==0 else "Single"
        fp2 = "bf16" if row==0 else "fp32"

        sns.stripplot(data=df, x="n_calls", y="error", ax=ax1)
        ax1.set_title(f"Accumulated Error Distributions ({fp2})")
        ax1.set_ylabel("Error")
        ax1.set_xlabel("ULP Calls")

        n_values = np.unique(y_dist).shape[0]
        bins = max(0, min(26, n_values))

        
        ax2.set_title(f"Final Error distribution \n {n_values} unique values")
        ax2.hist(y_dist, bins=bins, orientation="horizontal")

    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")

    plt.show()

def plot_error_histograms(y_dist, func_names, dtypes, fname=None):

        # assert len(functions)==len(func_names), "Check that the length of names and the functions match"

        n_rows = len(func_names)
        
        fig, axs = plt.subplots(nrows=n_rows, ncols=2,
                                figsize=(10, 4*n_rows))
        fig.suptitle("Error distributions")

        for k, func in enumerate(func_names):

            for j, dtype in enumerate(dtypes):

                Y_np  = y_dist[k, j, :].numpy().ravel()
                ax = axs[k, j]

                if k == 0:
                    ax.set_title(f"Dtype = {dtype}", fontsize=10)
                if j == 0:
                    ax.annotate(f"F = {func}", (0, 0.5),
                                xytext=(-30, 0),
                                textcoords="offset points",
                                xycoords="axes fraction",
                                ha="right", va="center",
                                rotation=90)

                n_values = len(np.unique(Y_np))
                bins     = n_values if n_values < 25 else int(n_values/2)

                ax.hist(Y_np, bins=bins, alpha=0.8, # label=f"{n_values} values"
                        )
                # ax.axvline(x=0, linestyle="--", color="red")

                ax.set(xlabel=f"{n_values} values")
                ax.tick_params(labelrotation=45)

        if fname is not None:
            fig.savefig(fname, bbox_inches="tight")
            print(f"Saved error distribution plots to {fname}")

        plt.tight_layout()
        plt.show()
        

def plot_cum_stats(Y, fname=None):

    labels     = ['max', 'q99', 'q75', 'median']
    linestyles = ['-', '--', '-.', ':']
    color      = 'black'

    x = range(Y[0].shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for i, ax in enumerate(axes):
        y_stats = Y[i]

        fp2 = "bf16" if i==1 else "fp32"

        ax.set_title(f"Cumulative Error Stats ({fp2})")
        ax.set_ylabel("Error")
        ax.set_xlabel("ULP Calls")

        for row, label, ls in zip(y_stats, labels, linestyles):
            ax.plot(x, row.numpy(), color=color, linestyle=ls, label=label)

    # Build explicit linestyle handles
    handles = [
        Line2D([0], [0], color=color, linestyle=ls, label=label)
        for label, ls in zip(labels, linestyles)
    ]

    fig.legend(handles=handles, loc='upper center', ncol=len(labels))
    if fname is not None:
        fig.savefig(fname, bbox_inches="tight")
    plt.show()