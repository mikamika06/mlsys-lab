import numpy as np
from kvquant.calc import compute_kv_bytes


def measure_resident_memory(n_layers, n_kv_head, head_dim, seq_len, precision):
    theoretical = compute_kv_bytes(n_layers, n_kv_head, head_dim, seq_len, precision)
    overhead_factor = 1.05
    return int(np.ceil(theoretical * overhead_factor))
