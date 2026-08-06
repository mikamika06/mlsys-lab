import numpy as np
from batchopt.profile import compute_curves


def recommend_batches(profile):
    bs = np.array(profile["batch_sizes"], dtype=int)
    lat, tp = compute_curves(profile)
    lat_opt_idx = int(np.argmin(lat))
    tp_opt_idx = int(np.argmax(tp))
    return {
        "latency_optimal_batch": int(bs[lat_opt_idx]),
        "throughput_optimal_batch": int(bs[tp_opt_idx]),
    }
