from typing import Callable, List, Dict, Any
import numpy as np
from tolgate.tolerance import evaluate_gate


def bisect_divergence(
    eager_steps: List[Callable[[np.ndarray], np.ndarray]],
    compiled_steps: List[Callable[[np.ndarray], np.ndarray]],
    initial_input: np.ndarray,
    dtype_str: str,
    k_terms: int
) -> int:
    """Find the index of the first step where compiled output diverges from eager output."""
    low = 0
    high = len(eager_steps) - 1
    divergent_idx = -1

    while low <= high:
        mid = (low + high) // 2

        x_eager = initial_input.copy()
        for i in range(mid + 1):
            x_eager = eager_steps[i](x_eager)

        x_comp = initial_input.copy()
        for i in range(mid + 1):
            x_comp = compiled_steps[i](x_comp)

        gate = evaluate_gate(x_comp, x_eager, dtype_str, k_terms)
        if not gate["passed"]:
            divergent_idx = mid
            high = mid - 1
        else:
            low = mid + 1

    return divergent_idx
