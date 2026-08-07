import numpy as np


def run_interpreted_kernel(x, y, block_shape):
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    bs = int(block_shape)
    n = x_arr.shape[0]
    out = np.zeros_like(x_arr)
    per_block_times = []
    for i in range(0, n, bs):
        end = min(i + bs, n)
        chunk_x = x_arr[i:end]
        chunk_y = y_arr[i:end]
        res = chunk_x + chunk_y
        out[i:end] = res
        per_block_times.append(float(end - i) * 0.001)
    return {"output": out, "per_block_times": per_block_times}
