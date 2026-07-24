import numpy as np
from typing import Set

def classify_removable_layers(bis: np.ndarray, threshold: float) -> Set[int]:
    """Return the set of layer indices whose Batch Importance is below the threshold."""
    return set(np.where(bis < threshold)[0])
