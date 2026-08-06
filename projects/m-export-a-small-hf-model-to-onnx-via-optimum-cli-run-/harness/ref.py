import numpy as np

CONFIGS = [
    {"model_name": "tiny-bert", "hidden_size": 64, "num_layers": 2, "vocab_size": 256},
    {"model_name": "tiny-gpt", "hidden_size": 128, "num_layers": 4, "vocab_size": 512},
]


def compute_output(spec, inputs):
    x = inputs.astype(np.float32)
    return x * 1.05 + 0.01
