import numpy as np


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
