import numpy as np


MODELS = [
    {"eos_token_id": 2, "stop_strings": ["<|endoftext|>", "###"]},
    {"eos_token_id": [2, 32000], "generation_config": {"stop_strings": ["<|im_end|>"]}},
    {"generation_config": {"eos_token_id": 1}}
]


def extract_stop_sequences(config):
    stops = set()
    if "eos_token_id" in config:
        eos = config["eos_token_id"]
        if isinstance(eos, list):
            stops.update(eos)
        else:
            stops.add(eos)
    if "stop_strings" in config:
        for s in config["stop_strings"]:
            stops.add(s)
    if "generation_config" in config:
        g = config["generation_config"]
        if "eos_token_id" in g:
            eos = g["eos_token_id"]
            if isinstance(eos, list):
                stops.update(eos)
            else:
                stops.add(eos)
        if "stop_strings" in g:
            for s in g["stop_strings"]:
                stops.add(s)
    return sorted(list(stops), key=lambda x: str(x))


def apply_adapter_and_forward(base_weights, adapter_weights, x, alpha=1.0):
    w_base = np.array(base_weights, dtype=float)
    a, b = adapter_weights
    w_adapted = w_base + alpha * (np.array(a, dtype=float) @ np.array(b, dtype=float))
    x_arr = np.array(x, dtype=float)
    return x_arr @ w_adapted
