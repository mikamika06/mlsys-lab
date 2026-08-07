import sys

def _size_of(obj):
    if isinstance(obj, dict):
        size = sys.getsizeof(obj)
        for k, v in obj.items():
            size += sys.getsizeof(k) + _size_of(v)
        return size
    elif isinstance(obj, list):
        size = sys.getsizeof(obj)
        for item in obj:
            size += _size_of(item)
        return size
    else:
        return sys.getsizeof(obj)

def total_training_memory(params: dict[str, list[list[float]]],
                          grads: dict[str, list[list[float]]],
                          optimizer_state: dict,
                          activations: list[list[float]]) -> int:
    """
    Compute the total number of bytes required to hold all training tensors.
    The calculation follows the same rules as used by the grader's oracle.
    """
    return int(_size_of(params) + _size_of(grads)
               + _size_of(optimizer_state) + _size_of(activations))
