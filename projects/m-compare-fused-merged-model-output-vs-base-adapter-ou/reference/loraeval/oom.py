import numpy as np


def tune_training_batch(vram_limit_mb, base_mb, rank, seq_len, target_tokens):
    best_batch = 1
    for b in range(1, 128):
        mem = base_mb + b * seq_len * rank * 0.002 + b * seq_len * 0.01
        if mem <= vram_limit_mb:
            best_batch = b
    iters = int(np.ceil(target_tokens / (best_batch * seq_len)))
    return {"batch_size": int(best_batch), "iterations": int(iters)}
