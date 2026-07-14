import argparse
import hashlib
import io
import json
import os
import zipfile
from dataclasses import asdict, dataclass
from functools import partial
from itertools import product
from typing import Any, List, Tuple

import torch
from utils.utils import hash_tensor

PLATFORMS_ABBR = [
    "a40",
    "a100",
    "a100-mig40",
    "rtx6000",
    "h100",
    # "rtx3090", # == a40
]

PLATFORMS = [
    "A40",
    "A100",
    "A100-mig40",
    "RTX6000",
    "H100",
    # "RTX 3090", # == a40
]


@dataclass
class LayerOutputMeta:
    layer_id: int
    layer_name: str
    module_type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    input_hash: str
    output_hash: str
    filename: str


@dataclass
class LayerOutput:
    meta: LayerOutputMeta
    output: torch.Tensor


class LayerOutputRecorder:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layer_counter = 0
        self.register_fns: List[Any] = []
        self.activation_records: List[LayerOutput] = []

    def record_hook(self, layer_name: str):
        def hook(module, input_args, output_args):
            self.layer_counter += 1

            input_tensor = input_args[0]
            assert len(input_args) == 1

            if isinstance(module, torch.nn.modules.activation.MultiheadAttention):
                output, none = output_args
                assert none is None
            else:
                output = output_args

            input_hash = hash_tensor(input_tensor).hex()
            output_hash = hash_tensor(output).hex()

            self.activation_records.append(
                LayerOutput(
                    LayerOutputMeta(
                        self.layer_counter,
                        layer_name,
                        type(module).__name__,
                        tuple(input_tensor.shape),
                        tuple(output.shape),
                        input_hash,
                        output_hash,
                        output_hash + ".pt",
                    ),
                    torch.clone(output).detach(),
                )
            )

        return hook

    def __enter__(self):
        self.layer_counter = 0
        self.activation_records.clear()

        # Register hooks recursively on *all* submodules
        for layer_name, module in self.model.named_modules():
            # Skip the top-level module (usually not helpful)
            if layer_name == "":
                continue

            self.register_fns.append(
                module.register_forward_hook(self.record_hook(layer_name))
            )

        return self

    def __exit__(self, *args):
        for register_fn in self.register_fns:
            register_fn.remove()


def hash_file(filename):
    with open(filename, "rb") as f:
        content = f.read()
    return hashlib.sha256(content).digest()


def verify_hash_equal(filename1, filename2):
    assert hash_file(filename1) == hash_file(filename2)

"""
@torch.inference_mode()
def process_success(model_path: str, x_path: str, *, device, loader):
    model = torch.load(model_path, weights_only=True, map_location=device)
    model.eval()

    x = torch.load(x_path, weights_only=True, map_location=device)
    x_norm = loader.normalize(x)

    with LayerOutputRecorder(model) as record:
        y = model(x_norm)

    return (
        y,
        x_norm,
        record.activation_records,
    )


@torch.inference_mode()
def main():
    torch.manual_seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform_name",
        choices=["a40", "a100", "a100-mig40", "rtx6000", "h100"],
        type=str,
        required=True,
    )
    parser.add_argument("--input_basedir", type=str, required=True)
    parser.add_argument("--output_basedir", type=str, required=True)
    args = parser.parse_args()

    platform_name = args.platform_name
    input_basedir = args.input_basedir
    output_basedir = args.output_basedir

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    assert torch.cuda.is_available()

    loader = get_dataset_loader("imagenet", "data")

    for platform_i, platform_j in product(PLATFORMS_ABBR, PLATFORMS_ABBR):
        platform_combination = f"{platform_i}-{platform_j}"
        logs_dir = os.path.join(input_basedir, platform_combination, "logs")

        if platform_i == platform_j:
            continue

        if not os.path.exists(logs_dir):
            print("SKIP", logs_dir)
            continue

        if platform_name not in [platform_i, platform_j]:
            continue

        for (
            x_index,
            rel_model_path,
            rel_x_path,
            success_type,
        ) in list_all_backdoored_models_from_log_dir(logs_dir):
            model_path = os.path.join(logs_dir, rel_model_path)
            x_path = os.path.join(logs_dir, rel_x_path)

            assert os.path.exists(model_path), model_path
            assert os.path.exists(x_path), x_path

            output_dir = os.path.join(
                output_basedir, platform_combination, f"Index-{x_index}"
            )
            output_model_path = os.path.join(
                output_dir, f"adv-model-{platform_name}.txt"
            )
            output_x_path = os.path.join(output_dir, f"x_fool-{platform_name}.txt")
            output_x_norm_path = os.path.join(output_dir, f"x_norm-{platform_name}.pt")
            output_x_norm_hash_path = os.path.join(
                output_dir, f"x_norm-hash-{platform_name}.txt"
            )
            output_y_path = os.path.join(output_dir, f"y_fool-{platform_name}.pt")
            output_y_txt_path = os.path.join(output_dir, f"y_fool-{platform_name}.txt")

            if (
                os.path.exists(output_x_path)
                and os.path.exists(output_x_norm_path)
                and os.path.exists(output_y_path)
                and os.path.exists(output_y_txt_path)
                and os.path.exists(output_x_norm_hash_path)
            ):
                print("SKIP", output_y_path)
                continue

            y, x_norm, activations_record = process_success(
                model_path,
                x_path,
                device=device,
                loader=loader,
            )

            print("Saving", output_dir, flush=True)
            os.makedirs(output_dir, exist_ok=True)

            activation_dir = os.path.join(output_dir, f"activations-{platform_name}")
            os.makedirs(activation_dir, exist_ok=True)

            with open(os.path.join(activation_dir, "activations-meta.json"), "w") as f:
                json.dump(
                    [asdict(record.meta) for record in activations_record], f, indent=4
                )

            with zipfile.ZipFile(
                os.path.join(activation_dir, "activations.zip"),
                "w",
                zipfile.ZIP_DEFLATED,
            ) as zipf:
                filenames = set()

                for layer_output in activations_record:
                    filename = layer_output.meta.filename

                    if filename in filenames:
                        continue

                    # Create a buffer to save the tensor
                    tensor_buffer = io.BytesIO()
                    torch.save(layer_output.output, tensor_buffer)
                    tensor_buffer.seek(0)  # Go back to the beginning of the buffer

                    # Tensor file name and save it to the zip
                    zipf.writestr(filename, tensor_buffer.read())
                    filenames.add(filename)

            with open(output_model_path, "w") as f:
                print(rel_model_path, file=f, end="")
            with open(output_x_path, "w") as f:
                print(rel_x_path, file=f, end="")

            torch.save(x_norm, output_x_norm_path)
            torch.save(y, output_y_path)

            with open(output_x_norm_hash_path, "w") as f:
                print(hash_tensor(x_norm).hex(), file=f)

            with open(output_y_txt_path, "w") as f:
                print(torch.argmax(y).item(), file=f)

    print("DONE")


if __name__ == "__main__":
    torch.serialization.add_safe_globals(
        [
            EfficientNet,
            set,
            Conv2dNormActivation,
            FusedMBConv,
            MBConv,
            StochasticDepth,
            SqueezeExcitation,
            BasicBlock,
            nn.AvgPool2d,
            nn.AdaptiveAvgPool2d,
            nn.BatchNorm2d,
            nn.Conv2d,
            nn.Dropout,
            nn.GELU,
            nn.LayerNorm,
            nn.Linear,
            nn.MaxPool2d,
            nn.MultiheadAttention,
            nn.ReLU,
            nn.Sequential,
            nn.SiLU,
            nn.Sigmoid,
            ResNet,
            NonDynamicallyQuantizableLinear,
            VisionTransformer,
            Encoder,
            EncoderBlock,
            MLPBlock,
            partial,
        ]
    )

    main()
 """