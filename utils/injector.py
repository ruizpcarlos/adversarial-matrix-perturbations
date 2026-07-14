import copy
from dataclasses import asdict, dataclass
from functools import partial
from itertools import product
from typing import Dict, Generator, List, Tuple

import torch

from utils.utils import hash_tensor

B2B = False  # execute on B and inject from B (instead of from A)
REPLACE_RELU = False  # replace 1st relu with 2nd relu output (should change result)
ADD_NOISE_TO_RELU = False  # add noise to one layer (should change result)

ID_FIRST_LAYER = 1


def implies(a: bool, b: bool):
    return not a or b


assert implies(REPLACE_RELU, B2B)
assert implies(ADD_NOISE_TO_RELU, REPLACE_RELU)

LAYER_4_1_RELU_1 = 68
LAYER_4_1_CONV2 = 69
LAYER_4_1_RELU_2 = 71

REPLACEMENT_KEY = 0xDEADBEEF

# PLATFORMS_ABBR = [
#     "a40",
#     "a100",
#     "a100-mig40",
#     "rtx6000",
#     "h100",
#     # "rtx3090", # == a40
# ]

# PLATFORMS = [
#     "A40",
#     "A100",
#     "A100-mig40",
#     "RTX6000",
#     "H100",
#     # "RTX 3090", # == a40
# ]


# BasicBlock::forward
def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)  # LAYER_4_1_RELU_1

    out = self.conv2(out)  # LAYER_4_1_CONV2
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)  # LAYER_4_1_RELU_2

    return out


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


@dataclass
class LayerRecord:
    layer_id: int
    layer_name: str
    module_type: str

    input_shapes: List[Tuple[int, ...]]
    output_shape: Tuple[int, ...]

    origin_input_ids: List[int]
    origin_output_id: int
    in_place: bool

    original_input_hash: str
    original_output_hash: str
    replaced_output_hash: str
    has_replaced_output: bool


class LayerOutputInjector:
    def __init__(
        self,
        model: torch.nn.Module,
        replaced_layer_ids: Dict[int, LayerOutput],
        *,
        detailled_layer_records: bool,
    ):
        self.model = model
        self.replaced_layer_ids = replaced_layer_ids
        self.detailled_layer_records = detailled_layer_records

        self.register_fns = []
        self.layer_id = 0

        self.layer_records = []

    def replace_hook(self, layer_name: str):
        def hook(module, input_args, output_args):
            self.layer_id += 1

            if isinstance(output_args, tuple):
                original_output = output_args[0]
            else:
                original_output = output_args

            # --- Original input/output ---
            assert isinstance(input_args, tuple), f"{layer_name}: {type(input_args)}"
            assert [torch.is_tensor(input_arg) for input_arg in input_args]

            assert torch.is_tensor(
                original_output
            ), f"{layer_name}: {type(original_output)}"

            input_shapes = [tuple(input_arg.shape) for input_arg in input_args]
            output_shape = tuple(original_output.shape)

            original_input_hash = hash_tensor(*input_args)
            original_output_hash = hash_tensor(original_output)

            has_replaced_output = True
            replacement = None

            try:
                replacement_layer_output = self.replaced_layer_ids[self.layer_id]
            except KeyError:
                has_replaced_output = False
                replacement = original_output
            else:
                assert layer_name == replacement_layer_output.meta.layer_name

                if not REPLACE_RELU:
                    assert self.layer_id == replacement_layer_output.meta.layer_id
                else:
                    if self.layer_id != LAYER_4_1_RELU_1:
                        assert self.layer_id == replacement_layer_output.meta.layer_id

                if REPLACE_RELU:
                    if self.layer_id == LAYER_4_1_RELU_1:  # layer4.1.relu
                        replacement_layer_output = self.replaced_layer_ids[
                            REPLACEMENT_KEY
                        ]

                replacement_orig = copy.deepcopy(replacement_layer_output.output)

                if REPLACE_RELU and ADD_NOISE_TO_RELU:
                    if self.layer_id == LAYER_4_1_RELU_1:
                        replacement_orig += 1

                if layer_name == "classifier.0":
                    replacement = replacement_orig.reshape(1, 1280)
                else:
                    replacement = replacement_orig

            replaced_output_hash = hash_tensor(replacement)

            assert (
                replacement.shape == output_shape
            ), f"{replacement.shape} == {output_shape}"

            # --- Record layer info (ALWAYS executed) ---
            if self.detailled_layer_records:
                self.layer_records.append(
                    LayerRecord(
                        layer_id=self.layer_id,
                        layer_name=layer_name,
                        module_type=type(module).__name__,
                        input_shapes=input_shapes,
                        output_shape=output_shape,
                        origin_input_ids=[id(input_arg) for input_arg in input_args],
                        origin_output_id=id(original_output),
                        in_place=id(input_args[0]) == id(original_output),
                        original_input_hash=original_input_hash.hex(),
                        original_output_hash=original_output_hash.hex(),
                        replaced_output_hash=replaced_output_hash.hex(),
                        has_replaced_output=has_replaced_output,
                    )
                )

            # --- Return ---
            if isinstance(module, torch.nn.modules.activation.MultiheadAttention):
                return replacement, None
            else:
                return replacement

        return hook

    def __enter__(self):
        self.layer_id = 0

        # Register hooks recursively on *all* submodules
        for layer_name, module in self.model.named_modules():
            # Skip the top-level module (usually not helpful)
            if layer_name == "":
                continue

            self.register_fns.append(
                module.register_forward_hook(self.replace_hook(layer_name))
            )

        return self

    def __exit__(self, *args):
        for register_fn in self.register_fns:
            register_fn.remove()


@torch.inference_mode()
def process_success(
    layer_id: int,
    model: torch.nn.Module,
    x_norm: torch.Tensor,
    replaced_layer_ids: Dict[int, LayerOutput],
    *,
    detailled_layer_records: bool,
):
    with LayerOutputInjector(
        model, replaced_layer_ids, detailled_layer_records=detailled_layer_records
    ) as injector:
        y: torch.Tensor = model(x_norm)

    values, indices = torch.topk(y[0], k=5, largest=True, sorted=True)
    top5_results = [
        {"class": int(idx), "value": float(val)} for idx, val in zip(indices, values)
    ]

    return y, {
        "replaced_layer": layer_id,
        "top5": top5_results,
        "records": [asdict(record) for record in injector.layer_records],
    }

""" 
def get_index_dirs(
    backdoor_basedir: str, input_basedir: str, platform_name: str, other_platform: str
) -> Generator[
    Tuple[str, str, str, str, str, List[LayerOutputMeta], str, str], None, None
]:
    if B2B:
        print("CAREFUL: You are using B2B")

    for platform_i, platform_j in product(PLATFORMS_ABBR, PLATFORMS_ABBR):
        platform_combination = f"{platform_i}-{platform_j}"
        platform_dir = os.path.join(input_basedir, platform_combination)

        if set([other_platform, platform_name]) != set([platform_i, platform_j]):
            continue

        # The execution happens on platform_name and we inject the recorded
        # activations from other_platform
        if platform_i == platform_name:
            other_platform = platform_j
        elif platform_j == platform_name:
            other_platform = platform_i
        else:
            assert False

        if B2B:
            other_platform = platform_name

        if not os.path.exists(platform_dir):
            continue

        for index_subdir in sorted(os.listdir(platform_dir)):
            index_dir = os.path.join(platform_dir, index_subdir)

            with open(os.path.join(index_dir, f"adv-model-{platform_name}.txt")) as f:
                rel_model_path = f.readline().strip()

            with open(os.path.join(index_dir, f"x_fool-{platform_name}.txt")) as f:
                rel_x_path = f.readline().strip()

            y_A_path = os.path.join(index_dir, f"y_fool-{other_platform}.pt")
            y_B_path = os.path.join(index_dir, f"y_fool-{platform_name}.pt")

            activations_subdir = f"activations-{other_platform}"
            activations_dir = os.path.join(index_dir, activations_subdir)

            with open(os.path.join(activations_dir, "activations-meta.json")) as f:
                activations_meta = json.load(f)
                activation_records = [
                    LayerOutputMeta(**obj) for obj in activations_meta
                ]

            data_dir = os.path.join(
                backdoor_basedir,
                platform_combination,
                "logs",
            )
            model_path = os.path.join(data_dir, rel_model_path)
            x_path = os.path.join(data_dir, rel_x_path)

            yield (
                model_path,
                x_path,
                platform_combination,
                index_subdir,
                activations_dir,
                activation_records,
                y_A_path,
                y_B_path,
            )
 """