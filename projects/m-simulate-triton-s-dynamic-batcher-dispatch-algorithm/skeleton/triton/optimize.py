from triton.simulate import simulate
from triton.metrics import calculate_metrics

def optimize_delay(arrivals: list[int], max_batch_size: int, preferred_batch_sizes: list[int], delays_to_try: list[int], throughput_floor: float, compute_fn):
    raise NotImplementedError
