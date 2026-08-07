import numpy as np

def per_tensor_vs_block(x: np.ndarray, block_size: int) -> tuple[float, float]:
    x = x.flatten()
    scale_t = max(float(np.max(np.abs(x))) / 127.0, 1e-9)
    xq_t = np.round(x / scale_t) * scale_t
    err_tensor = float(np.mean((x - xq_t)**2))

    err_block_sum = 0.0
    for i in range(0, len(x), block_size):
        block = x[i:i+block_size]
        scale_b = max(float(np.max(np.abs(block))) / 127.0, 1e-9)
        bq = np.round(block / scale_b) * scale_b
        err_block_sum += np.sum((block - bq)**2)

    err_block = float(err_block_sum / len(x))
    return err_tensor, err_block
