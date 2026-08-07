from typing import Any, Callable, Dict, List, Tuple

def run_parameter_sweep(
    handler_factory: Callable[[int, float], Callable],
    request_sequence: List[Tuple[float, Any]],
    max_batch_sizes: List[int],
    timeouts: List[float]
) -> List[Dict[str, Any]]:
    """Sweep max_batch_size x batch_wait_timeout_s and collect metrics."""
    raise NotImplementedError
