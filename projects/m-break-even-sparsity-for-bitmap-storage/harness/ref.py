import numpy as np
from edge_prune.pruning import get_layer_masks, get_global_masks
from edge_prune.storage import find_break_even_sparsity, calculate_theoretical_size


def generate_fixtures():
    rng = np.random.RandomState(42)
    return {
        "layer1": rng.randn(64, 64),
        "layer2": rng.randn(128, 64),
        "layer3": rng.randn(10)
    }
    

FIXTURES = generate_fixtures()
