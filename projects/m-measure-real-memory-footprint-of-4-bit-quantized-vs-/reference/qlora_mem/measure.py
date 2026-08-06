import numpy as np

def measure_footprint(tensor, block_size=64):
    arr = np.asarray(tensor, dtype=np.float32)
    orig_bytes = arr.nbytes
    numel = arr.size
    n_blocks = (numel + block_size - 1) // block_size
    data_bytes = (numel + 1) // 2
    absmax_bytes = n_blocks * 2
    quant_bytes = data_bytes + absmax_bytes
    return {"orig_bytes": orig_bytes, "quant_bytes": quant_bytes}
