import numpy as np


def mixed_precision_step(master_weight, grad_fp16, lr):
    master32 = np.asarray(master_weight, dtype=np.float32)
    grad32 = np.asarray(grad_fp16, dtype=np.float32)
    updated_master = master32 - np.float32(lr) * grad32
    return updated_master, updated_master.astype(np.float16)
