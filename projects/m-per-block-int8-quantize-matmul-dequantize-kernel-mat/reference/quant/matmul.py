import numpy as np


def per_block_int8_quant_matmul(
    A: np.ndarray, B: np.ndarray, block_size: int = 32
) -> np.ndarray:
    """Compute A @ B using per-block INT8 quantization along inner dimension K."""
    M, K = A.shape
    K_B, N = B.shape
    assert K == K_B, "Inner dimensions must match"
    assert K % block_size == 0, "K dimension must be divisible by block_size"

    num_blocks = K // block_size

    A_reshaped = A.reshape(M, num_blocks, block_size)
    A_max = np.max(np.abs(A_reshaped), axis=-1, keepdims=True)
    scale_A = np.where(A_max > 0, A_max / 127.0, 1.0)
    q_A = np.clip(np.round(A_reshaped / scale_A), -128, 127).astype(np.int8)

    B_t = B.T
    B_reshaped = B_t.reshape(N, num_blocks, block_size)
    B_max = np.max(np.abs(B_reshaped), axis=-1, keepdims=True)
    scale_B = np.where(B_max > 0, B_max / 127.0, 1.0)
    q_B = np.clip(np.round(B_reshaped / scale_B), -128, 127).astype(np.int8)

    C = np.zeros((M, N), dtype=np.float32)

    for b in range(num_blocks):
        sub_A = q_A[:, b, :].astype(np.float32)
        sub_B = q_B[:, b, :].astype(np.float32)

        block_acc = np.matmul(sub_A, sub_B.T)

        s_a = scale_A[:, b, :]
        s_b = scale_B[:, b, :]
        Combined_scale = np.matmul(s_a, s_b.T)

        C += block_acc * Combined_scale

    return C
