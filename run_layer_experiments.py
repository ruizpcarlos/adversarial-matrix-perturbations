"""
Compare RANDOM / RANDOM_W / Simulated Annealing / Genetic Algorithm on the
classifier-layer ULP-perturbation task, over many seeds.

Usage:
    python run_layer_experiments.py --n_seeds 20 [--weighted] [--early_stop]

features.pt may be either
    {"features": Tensor[N, D], "features_val": Tensor[M, D]}   or   (train, val)

Fairness choices
----------------
* Budget: every algorithm gets `--budget` fitness() calls (n_evals).
* Held-out evaluation: after each run the best solution is scored on a batch
  drawn from features_val with an RNG that depends ONLY on the seed, so every
  algorithm in the same seed is judged on exactly the same inputs. (final_score()
  would draw a different batch per algorithm, adding noise to the comparison.)
* Early stopping is OFF by default so all algorithms spend the same budget.
"""
import argparse
import gc
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from tqdm import tqdm

# --- adjust to wherever the three layer-search files live (they use relative imports) ---
from weight_attack.adv_layer import AdvLayerPerturbation
from weight_attack.simulated_annealing import SimulatedAnnealingLayerSearch
from weight_attack.genetic_algorithm import AdvLayerGeneticAlgorithm

from utils.utils import load_model_and_weights, adjusted_pvals
from utils.objective_func import net_success_pct, mean_log_ratio

DRIVE_DIR = "/content/drive/MyDrive/exp_results"

MAX_CALLS = 32
EVAL_SEED_OFFSET = 10_000
BUDGET = 2_000


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_features(feat_path, val_path):
    train = torch.load(feat_path, map_location="cpu")
    val = torch.load(val_path, map_location="cpu")
    # if isinstance(obj, dict):
    #     train, val = obj["features"], obj["features_val"]
    # else:
    #     train, val = obj
    return train.float(), val.float()


# ----------------------------------------------------------------------
# Algorithms: name -> (class, extra ctor kwargs, runner(inst) -> None)
# ----------------------------------------------------------------------
def build_algos(args):
    patience = 25 if args.early_stop else 10**9
    w = args.weighted

    def sa(alpha, beta):
        return lambda s: s.search(L0=5, alpha=alpha, beta=beta, patience=patience,
                                  test_batch_size=args.test_batch_size)

    def ga(_):
        return lambda s: s.search(early_stopping=args.early_stop)

    return {
        # random_search needs the sampling weights to exist for the weighted variant
        "RANDOM":   (AdvLayerPerturbation, dict(weighted_sampling=True),
                     lambda s: s.random_search(weighted=False, test_batch_size=args.test_batch_size)),
        "RANDOM_W": (AdvLayerPerturbation, dict(weighted_sampling=True),
                     lambda s: s.random_search(weighted=True, test_batch_size=args.test_batch_size)),
        # "SA_80":    (SimulatedAnnealingLayerSearch, dict(weighted_sampling=w), sa(0.80, 1.04)),
        "SA_85":    (SimulatedAnnealingLayerSearch, dict(weighted_sampling=w), sa(0.85, 1.035)),
        "GA_50":    (AdvLayerGeneticAlgorithm, dict(weighted_sampling=w, pop_size=50), ga(50)),
        # "GA_100":   (AdvLayerGeneticAlgorithm, dict(weighted_sampling=w, pop_size=100), ga(100)),
    }


def get_curve(inst):
    """(evals, held-out-C score of the incumbent) after each logged step."""
    if hasattr(inst, "best_C"):                        # GA: one entry per generation
        return np.array(inst.ulp_calls[1:]), np.array(inst.best_C)
    return np.array(inst.log["evals"]), np.array(inst.log["best_C"])   # SA / random


def train_fitness(inst):
    if hasattr(inst, "history"):                       # GA
        return float(inst.history[-1][1])
    return float(inst.log["best_B"][-1])               # SA / random


# ----------------------------------------------------------------------
# Common held-out evaluation (identical inputs for all algorithms of a seed)
# ----------------------------------------------------------------------
@torch.inference_mode()
def held_out_eval(inst, seed, size):
    idx, n_ulp = inst.solution

    saved = inst.batch_rng
    inst.batch_rng = np.random.default_rng(EVAL_SEED_OFFSET + seed)
    try:
        h_cpu, h_gpu, e0 = inst.set_batch("C", size)
    finally:
        inst.batch_rng = saved

    W_c = inst.apply_ulps(idx, n_ulp)
    y_c = F.linear(h_cpu, W_c, inst.b_cpu)
    y_g = F.linear(h_gpu, W_c.to("cuda"), inst.b_gpu).cpu()
    e = inst.objective_fn(y_c, y_g)

    return {
        "max_err":        net_success_pct(e, e0),        # key name expected by adjusted_pvals
        "frac_up":        (e > e0).float().mean().item(),
        "frac_down":      (e < e0).float().mean().item(),
        "mean_err_ratio": (e.mean() / (e0.mean() + 1e-30)).item(),
        "mean_log_ratio": mean_log_ratio(e, e0),
    }


# ----------------------------------------------------------------------
# Single run
# ----------------------------------------------------------------------
def run_one(name, spec, seed, model, model_name, feats, args):
    cls, ctor_kw, runner = spec
    torch.manual_seed(seed)

    inst = cls(model, model_name, q=args.q, max_calls=MAX_CALLS,
               budget_calls=BUDGET, seed=seed, batch_size=args.batch_size, **ctor_kw)
    inst.set_features(*feats)

    t0 = time.time()
    runner(inst)
    elapsed = time.time() - t0     # includes the per-step held-out logging, so only roughly comparable

    rec = held_out_eval(inst, seed, args.eval_size)
    evals, curve = get_curve(inst)
    rec.update(algo=name, seed=seed, time_sec=elapsed, n_evals=inst.n_evals,
               n_ulp=int(inst.solution[1]), train_fit=train_fitness(inst),
               curve_evals=evals, curve_C=curve)

    del inst
    gc.collect()
    torch.cuda.empty_cache()
    return rec


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def summarize(results, out_prefix):
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("curve")}
                       for r in results.values()])
    table = (df.groupby("algo")
               .agg(n=("seed", "count"),
                    score_mean=("max_err", "mean"), score_std=("max_err", "std"),
                    score_median=("max_err", "median"),
                    frac_up=("frac_up", "mean"), frac_down=("frac_down", "mean"),
                    log_ratio=("mean_log_ratio", "mean"),
                    n_evals=("n_evals", "mean"), time_s=("time_sec", "mean"))
               .sort_values("score_mean", ascending=False))
    print("\n=== Held-out results (mean over seeds) ===")
    print(table.to_string(float_format=lambda x: f"{x:.4g}"))
    table.to_csv(out_prefix + "_summary.csv")
    df.to_csv(out_prefix + "_runs.csv", index=False)

    # Paired tests vs RANDOM (Wilcoxon + BH-FDR), only over seeds present for every algo
    algos = sorted(df.algo.unique())
    common = sorted(set.intersection(*[set(df[df.algo == a].seed) for a in algos]))
    by_algo = {a: [results[(a, s)] for s in common] for a in algos}
    print(f"\n=== Paired Wilcoxon vs RANDOM (alternative='greater', {len(common)} seeds) ===")
    for a, r in adjusted_pvals(by_algo, benchmark_key="RANDOM", alternative="greater").items():
        print(f"{a:9s} median_diff={r['median_diff']:+.4f}  p={r['p_value']:.3g}  "
              f"p_adj={r['p_adj']:.3g}  {'*' if r['significant'] else ''}")

    # Head-to-head: best GA config vs best SA config (two-sided)
    def best(prefix):
        c = [a for a in algos if a.startswith(prefix)]
        return max(c, key=lambda a: np.mean([r["max_err"] for r in by_algo[a]])) if c else None
    ga, sa = best("GA"), best("SA")
    if ga and sa:
        d = np.array([g["max_err"] - s["max_err"] for g, s in zip(by_algo[ga], by_algo[sa])])
        p = wilcoxon(d, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
        print(f"\n{ga} vs {sa}: median diff = {np.median(d):+.4f}, two-sided p = {p:.3g} "
              f"(configs picked by mean score, so slightly optimistic)")

    plot_curves(results, algos, common, out_prefix + "_curves.pdf")


def plot_curves(results, algos, seeds, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_e = int(max(r["curve_evals"].max() for r in results.values()))
    grid = np.arange(0, max_e + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for a in algos:
        Y = []
        for s in seeds:
            r = results[(a, s)]
            i = np.searchsorted(r["curve_evals"], grid, side="right") - 1   # step function
            Y.append(np.where(i >= 0, r["curve_C"][np.clip(i, 0, None)], np.nan))
        Y = np.array(Y)
        m, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
        ax.plot(grid, m, label=a)
        ax.fill_between(grid, m - sd, m + sd, alpha=0.12)
    ax.set_xlabel("fitness evaluations")
    ax.set_ylabel("score of incumbent on held-out batch C")
    ax.set_title(f"Anytime performance (mean ± std, {len(seeds)} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"Saved {fname}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_seeds", type=int, default=20)
    p.add_argument("--seed0", type=int, default=161)
    p.add_argument("--model_name", default="ResNet18")
    p.add_argument("--q", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256, help="size of batches A and B")
    p.add_argument("--test_batch_size", type=int, default=256, help="batch C used for logging curves")
    p.add_argument("--eval_size", type=int, default=1024, help="shared held-out eval batch")
    p.add_argument("--weighted", action=argparse.BooleanOptionalAction, default=True,
                   help="weighted sampling for SA/GA (RANDOM / RANDOM_W always run both ways)")
    p.add_argument("--early_stop", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--out", default="results_layer")
    args = p.parse_args()

    tag = f"{args.out}_{args.model_name}_{args.seed0}" + ("_weighted" if args.weighted else "")
    results_file = tag + ".pkl"
    results_file_drive = os.path.join(DRIVE_DIR,
                                       results_file)

    feat_path = f"features{args.model_name}.pt"
    val_path  = f"features{args.model_name}_val.pt"

    model, _ = load_model_and_weights(args.model_name)
    feats = load_features(feat_path, val_path)
    algos = build_algos(args)
    seeds = list(range(args.seed0, args.seed0 + args.n_seeds))

    results = {}
    if os.path.exists(results_file_drive):                       # resume
        with open(results_file_drive, "rb") as f:
            results = pickle.load(f)
        print(f"Resuming: {len(results)} runs already done")

    for seed in tqdm(seeds, desc="Seeds"):
        for name, spec in algos.items():
            if (name, seed) in results:
                continue
            rec = run_one(name, spec, seed, model, args.model_name, feats, args)
            results[(name, seed)] = rec
            tqdm.write(f"seed={seed} {name:9s} score={rec['max_err']:+.4f} "
                       f"up={rec['frac_up']:.3f} down={rec['frac_down']:.3f} "
                       f"evals={rec['n_evals']} n_ulp={rec['n_ulp']} t={rec['time_sec']:.1f}s")
            tmp = results_file + ".tmp"                    # atomic checkpoint
            with open(tmp, "wb") as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, results_file)

            with open(results_file_drive, "wb") as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            

    summarize(results, tag)


if __name__ == "__main__":
    main()
