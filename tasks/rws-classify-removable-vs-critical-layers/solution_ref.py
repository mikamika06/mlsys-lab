from typing import Set

def classify_removable_layers(bis: list[float], threshold: float) -> Set[int]:
    """Return the set of layer indices whose Batch Importance is below the threshold."""
    result = set()
    for i in range(len(bis)):
        if bis[i] < threshold:
            result.add(i)
    return result
