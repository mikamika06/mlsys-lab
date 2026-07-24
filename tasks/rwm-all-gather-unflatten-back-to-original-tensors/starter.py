import numpy as np


def unflatten_all_gathered(shards: list[np.ndarray], shapes: list[tuple[int, ...]]) -> list[np.ndarray]:
    """Reconstruct original parameter tensors from gathered shards."""
    raise NotImplementedError
