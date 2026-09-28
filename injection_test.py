# %%
import copy
import time
import urllib.request
import tarfile
import os
import argparse
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.linalg import vector_norm

import torchvision.models as models
import torchvision.datasets
from torch.utils.data import DataLoader
# from huggingface_hub import login

from genetic_algorithm import AdversarialGeneticAlgorithm
from simulated_annealing import SimulatedAnnealingSearch

from utils.utils import load_model_and_weights, save_dict_to_pickle
from utils.output_recorder import LayerOutput, LayerOutputRecorder
from utils.injector import process_success #, LayerOutputInjector
from utils.cli import add_common_args


DRIVE_ROOT = "/content/drive"

if os.path.ismount(DRIVE_ROOT) and os.path.isdir(f"{DRIVE_ROOT}/MyDrive"):
    INJ_DIR = f"{DRIVE_ROOT}/MyDrive/exp_results/injection"
    print("Saving to Drive")
else:
    INJ_DIR = "/content/injection"
    print("WARNING: Drive not mounted, saving to local ephemeral disk")

os.makedirs(INJ_DIR, exist_ok=True)

WNID_TO_NAME = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

IMAGENETTE_TO_IMAGENET = {
    0: 0,     # tench
    1: 217,   # English springer
    2: 482,   # cassette player
    3: 491,   # chain saw
    4: 497,   # church
    5: 566,   # French horn
    6: 569,   # garbage truck
    7: 571,   # gas pump
    8: 574,   # golf ball
    9: 701,   # parachute
}


def load_imagenette(preprocess):
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    root = "imagenette2-320"

    if not os.path.exists(root):
        urllib.request.urlretrieve(url, "imagenette2-320.tgz")
        with tarfile.open("imagenette2-320.tgz") as f:
            f.extractall(".")

    dataset = torchvision.datasets.ImageFolder(f"{root}/val", transform=preprocess)
    return dataset


def get_images_and_labels(dataset, N_sample, generator):
    loader = DataLoader(dataset, 
                        shuffle=True,
                        batch_size=N_sample,    
                        generator=generator)
    X, y = next(iter(loader))
    return X, y


def sample_inputs(X: torch.Tensor, n_sample: int, generator):
    idx = torch.randperm(X.shape[0], generator=generator)[:n_sample]
    return X[idx], idx


@torch.inference_mode()
def successive_layer_injection(x, model_record, model_inject, record_device, inject_device):
    x_record, x_inject = x.to(record_device), x.to(inject_device)

    with LayerOutputRecorder(model_record) as recorder:
        y_record = model_record(x_record)

    records_inject = {
        rec.meta.layer_id: LayerOutput(rec.meta, rec.output.to(inject_device))
        for rec in recorder.activation_records
    }

    errors = []
    replaced_layers = {}

    def add_error(y):
        # stay on-device, no .item() / sync per step
        errors.append(
            vector_norm((y.to(record_device) - y_record).flatten(), ord=float("inf"))
        )

    y0, _ = process_success(0, model_inject, x_inject, {}, detailled_layer_records=False)
    add_error(y0)

    for layer_id in sorted(records_inject):
        replaced_layers[layer_id] = records_inject[layer_id]
        y_i, _ = process_success(
            layer_id, model_inject, x_inject, replaced_layers, detailled_layer_records=False
        )
        add_error(y_i)

    return torch.stack(errors).cpu()      # single sync per input


def run_ablation(inputs, model, platforms, out_path="ablation_errors.pt"):
    assert len(platforms) == 2
    record_device, inject_device = map(torch.device, platforms)

    # built ONCE, not per input
    model_record = copy.deepcopy(model).eval().to(record_device)
    model_inject = copy.deepcopy(model).eval().to(inject_device)

    all_errors = []
    for x in tqdm(inputs, desc="inputs"):
        all_errors.append(
            successive_layer_injection(x, model_record, model_inject, record_device, inject_device)
        )
        torch.save(torch.stack(all_errors), out_path)   # checkpoint

    return torch.stack(all_errors)          # shape (30, n_layers + 1)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    add_common_args(parser, include_seed=True)
    args = parser.parse_args()

    SEED       = args.seed
    DATASET_N  = 1_000
    _GENERATOR = torch.Generator().manual_seed(SEED)
    model_name = "ResNet18"
    
    weights    = models.ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    resnet18, _  = load_model_and_weights(model_name=model_name)
    resnet18_gpu = copy.deepcopy(resnet18).eval().to("cuda")

    dataset = load_imagenette(preprocess)
    X, y    = get_images_and_labels(dataset, DATASET_N, _GENERATOR)

    idx_to_name = {i: WNID_TO_NAME[wnid] for wnid, i in dataset.class_to_idx.items()}
    y_imagenet  = torch.tensor([IMAGENETTE_TO_IMAGENET[int(l)] for l in y])

    RUN_TAG = f"{model_name}_{SEED}"    
    # ------------------------------------------------------------------
    # SEARCH PARAMS
    # ------------------------------------------------------------------
    N_test       = args.n_samples
    max_calls    = 32 # if data.upper().startswith("BF") else 128
    budget       = 1_000  # total budget of calls to compute_max_err
    perturb_frac = 0.1

    # --------------------------------------------------
    #       SAMPLE A BATCH
    # --------------------------------------------------
    X_sub, idx = sample_inputs(X, N_test, _GENERATOR)
    inputs     = [x.unsqueeze(0) for x in X_sub]

    fname   = os.path.join(INJ_DIR, f"non_perturbed_{RUN_TAG}.pt")

    if os.path.exists(fname):
        print("Reading ablation results from cache")
        y_clean = torch.load(fname)
    else:
        print("Running ablation test for non-perturbed images")
        y_clean = run_ablation(inputs, resnet18, ['cpu', 'cuda'], out_path=fname)

    adv_fname  = os.path.join(INJ_DIR, f"adv_inputs_{SEED}")
    adv_pkl = adv_fname + ".pkl"
    # from_cache = os.path.exists(adv_fname + ".pkl")

    adv_inputs = {}
    if os.path.exists(adv_pkl):
        try:
            with open(adv_pkl, "rb") as f:
                adv_inputs = pickle.load(f)
            print(f"Loaded {len(adv_inputs)} cached adversarial inputs")
        except (EOFError, pickle.UnpicklingError):
            print("Cache corrupted, starting fresh")

    iter_aux   = list(zip(idx, inputs))
    todo = [(i, x) for i, x in iter_aux if i not in adv_inputs]
    
    if todo:
        for i, x in tqdm(todo, desc="Generating adversarial perturbations"):    
            ga = AdversarialGeneticAlgorithm(
                            input_matrix = x,
                            func         = resnet18,
                            q            = perturb_frac,
                            max_calls         = max_calls,
                            budget_calls      = budget,
                            weighted_sampling = True,
                            pop_size          = 50
                        )
            # start   = time.time()
            ga.search(early_stopping=True)
            # elapsed = time.time() - start

            # print(f"Elapsed time = {elapsed:.2f}s")
            # print(f"max err = {ga.history[-1][1]:.4e}"
            #     f"({100*(ga.history[-1][1]/target):.2f}% of baseline)")

            X_adv = ga.compute_adv_input()
            adv_inputs.update({i: X_adv})
            adv_inputs_list = list(adv_inputs.values())

            ga.release()

            save_dict_to_pickle(adv_inputs, adv_pkl)
            torch.save(torch.stack(adv_inputs_list), adv_fname + ".pt")
    
    fname   = os.path.join(INJ_DIR, f"perturbed_{RUN_TAG}.pt")
    if os.path.exists(fname):
        print("Reading ablation results from cache")
        y_adv = torch.load(fname)
    else:
        print("Running ablation test for adversarial inputs")
        y_adv = run_ablation(adv_inputs_list, resnet18, ['cpu', 'cuda'], out_path=fname)
