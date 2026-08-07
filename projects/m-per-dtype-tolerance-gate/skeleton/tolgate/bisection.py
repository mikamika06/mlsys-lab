from typing import Callable, List, Dict, Any
import numpy as np


def bisect_divergence(
    eager_steps: List[Callable[[np.ndarray], np.ndarray]],
    compiled_steps: List[Callable[[np.ndarray], np.ndarray]],
    initial_input: np.ndarray,
    dtype_str: str,
    k_terms: int
) -> int:
    """Find the index of the first step where compiled output diverges from eager output."""
    raise NotImplementedError
