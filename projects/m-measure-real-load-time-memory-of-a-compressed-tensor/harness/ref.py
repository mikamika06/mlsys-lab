import numpy as np

CONFIGS = [
    {"name": "int4_group128", "bits": 4, "group_size": 128, "numel": 1024 * 1024, "sparsity": 0.0},
    {"name": "sparse_int8", "bits": 8, "group_size": 64, "numel": 2048 * 2048, "sparsity": 0.5},
    {"name": "fp16_dense", "bits": 16, "group_size": 0, "numel": 512 * 512, "sparsity": 0.0},
]


def parse_metadata(config):
    bits = config["bits"]
    numel = config["numel"]
    sparsity = config["sparsity"]
    effective_numel = int(numel * (1.0 - sparsity))
    theoretical_bytes = int(np.ceil(effective_numel * bits / 8.0))
    return {
        "name": config["name"],
        "effective_numel": effective_numel,
        "theoretical_bytes": theoretical_bytes,
    }


def simulate_load_memory(config):
    meta = parse_metadata(config)
    overhead_factor = 1.15
    if config["sparsity"] > 0.0:
        overhead_factor = 1.30
    return int(meta["theoretical_bytes"] * overhead_factor)
