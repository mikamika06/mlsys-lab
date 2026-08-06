import numpy as np


def compute_memory_tradeoff(seq_len, batch_size, checkpoint_layers):
    base_mem = float(seq_len * seq_len * batch_size * 4)
    if checkpoint_layers > 0:
        saved_mem = base_mem * (1.0 - 0.5 / float(checkpoint_layers))
        recompute_overhead = float(checkpoint_layers) * 12.5
    else:
        saved_mem = base_mem
        recompute_overhead = 0.0
    return {"saved_memory": float(saved_mem), "overhead": float(recompute_overhead)}
