import ref
import numpy as np

def check(workdir):
    from headprune.greedy import compute_removal_order
    from headprune.latency import measure_latency
    rng = np.random.default_rng(42)
    mat = rng.uniform(size=(4, 8))
    want_order = ref.compute_removal_order(mat)
    got_order = compute_removal_order(mat)
    order_match = 1.0 if list(got_order) == list(want_order) else 0.0

    base_lat = 100.0
    total_h = 32
    removed = 8
    want_lat = ref.measure_latency(base_lat, removed, total_h)
    got_lat = measure_latency(base_lat, removed, total_h)
    latency_valid = 1.0 if abs(got_lat - want_lat) < 1e-5 else 0.0

    return {"order_match": order_match, "latency_valid": latency_valid}
