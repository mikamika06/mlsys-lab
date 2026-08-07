import numpy as np


def analyze_scaling_overflow(x: np.ndarray, block_size: int = 32) -> dict:
    """Analyze dynamic range preservation under per-tensor vs per-block scaling."""
    abs_x = np.abs(x)
    tensor_max = np.max(abs_x)
    tensor_scale = tensor_max / 127.0 if tensor_max > 0 else 1.0
    q_tensor = np.clip(np.round(x / tensor_scale), -128, 127)
    deq_tensor = q_tensor * tensor_scale
    tensor_sq_err = float(np.mean((x - deq_tensor) ** 2))

    shape = x.shape
    x_flat = x.reshape(-1, block_size)
    block_max = np.max(np.abs(x_flat), axis=-1, keepdims=True)
    block_scale = np.where(block_max > 0, block_max / 127.0, 1.0)
    q_block = np.clip(np.round(x_flat / block_scale), -128, 127)
    deq_block = (q_block * block_scale).reshape(shape)
    block_sq_err = float(np.mean((x - deq_block) ** 2))

    zero_cnt_tensor = int(np.sum((q_tensor == 0) & (x != 0)))
    zero_cnt_block = int(np.sum((q_block == 0) & (x_flat != 0)))

    return {
        "tensor_mse": tensor_sq_err,
        "block_mse": block_sq_err,
        "tensor_underflow_count": zero_cnt_tensor,
        "block_underflow_count": zero_cnt_block,
        "mse_ratio": tensor_sq_err / (block_sq_err + 1e-12)
    }
