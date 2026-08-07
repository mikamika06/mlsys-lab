import numpy as np


def allocate_layer_strategies(layer_weights: list, target_total_bits: float, n: int, m: int, bit_options: list) -> list:
    """Assigns per-layer prune-vs-palettize strategies maximizing overall accuracy under a global budget."""
    raise NotImplementedError
