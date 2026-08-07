import sys
import math

def _oracle_size_of(obj):
    if isinstance(obj, dict):
        size = sys.getsizeof(obj)
        for k, v in obj.items():
            size += sys.getsizeof(k) + _oracle_size_of(v)
        return size
    elif isinstance(obj, list):
        size = sys.getsizeof(obj)
        for item in obj:
            size += _oracle_size_of(item)
        return size
    else:
        return sys.getsizeof(obj)

def grade(sol, fx) -> dict:
    params = {"w": [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]}
    grads = {"w": [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]}
    optimizer_state = {"momentum": [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]}
    activations = [[float(i) for i in range(12)]]

    oracle_val = int(_oracle_size_of(params) + _oracle_size_of(grads) +
                     _oracle_size_of(optimizer_state) + _oracle_size_of(activations))

    candidate_val = sol.total_training_memory(params, grads, optimizer_state, activations)

    if candidate_val == 0:
        ratio = 0.0
    else:
        ratio = float(oracle_val) / float(candidate_val)

    passed = math.isclose(ratio, 1.0, abs_tol=1e-9)

    return {
        "size_ratio": ratio,
        "passed": passed
    }
