import numpy as np


def derive_load_balanced_sharding(num_layers, layer_weights, num_ranks=4):
    """Derive optimal layer assignment across num_ranks minimizing max load and comm transitions."""
    raise NotImplementedError
