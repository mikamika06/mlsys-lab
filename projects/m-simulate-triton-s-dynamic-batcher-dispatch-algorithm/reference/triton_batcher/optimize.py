from triton_batcher.simulate import simulate
from triton_batcher.metrics import measure_metrics

def optimize_config(arrivals: list[int], max_batch_size: int, preferred_candidates: list[list[int]], delay_candidates: list[int], throughput_floor: float, compute_us_fn) -> dict:
    best_p99 = float('inf')
    best_config = None

    for pref in preferred_candidates:
        for delay in delay_candidates:
            disps = simulate(arrivals, max_batch_size, pref, delay, compute_us_fn)
            metrics = measure_metrics(arrivals, disps, compute_us_fn)

            if metrics["throughput_req_sec"] >= throughput_floor:
                if metrics["p99_queue_delay_us"] < best_p99:
                    best_p99 = metrics["p99_queue_delay_us"]
                    best_config = {"preferred": pref, "delay_us": delay}

    return best_config
