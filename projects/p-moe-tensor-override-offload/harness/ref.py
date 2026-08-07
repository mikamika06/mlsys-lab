import numpy as np
from moe_offload.offload import MoEOffloader

def get_sample_data():
    np.random.seed(42)
    sizes = [100, 200, 150, 250, 300, 100]
    traces = [
        [0, 1, 2],
        [0, 1],
        [0, 1, 2, 3],
        [0, 1],
        [0, 1, 2]
    ]
    base_latency = [10.0, 15.0, 12.0, 20.0, 25.0, 10.0]
    ref_out = np.array([1.0, 2.0, 3.0])
    cand_out = np.array([1.0, 2.0, 3.000001])
    return sizes, traces, base_latency, ref_out, cand_out
