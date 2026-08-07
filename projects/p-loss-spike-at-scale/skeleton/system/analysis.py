import numpy as np

def find_first_spike(losses: list[float]) -> int:
    raise NotImplementedError()

def simulate_data_invariant(total_samples: int, num_workers: int) -> float:
    raise NotImplementedError()

def check_determinism(reduce_fn, tensors: list[np.ndarray]) -> float:
    raise NotImplementedError()
