import numpy as np


def compute_block_mask(offsets: np.ndarray, N: int) -> np.ndarray:
    return offsets < N


def process_data(x: np.ndarray, N: int, BLOCK_SIZE: int = 64) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    padded_len = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    buf_in = np.zeros(padded_len, dtype=np.float32)
    buf_in[:N] = x_arr[:N]
    buf_out = np.zeros(padded_len, dtype=np.float32)

    num_blocks = padded_len // BLOCK_SIZE
    for pid in range(num_blocks):
        offsets = pid * BLOCK_SIZE + np.arange(BLOCK_SIZE, dtype=np.int32)
        mask = compute_block_mask(offsets, N)
        in_vals = np.where(mask, buf_in[offsets], 0.0)
        out_vals = in_vals * 2.5 + 1.0
        buf_out[offsets] = np.where(mask, out_vals, buf_out[offsets])

    return buf_out[:N]


def detect_corrupted_indices(N: int, BLOCK_SIZE: int = 64) -> np.ndarray:
    padded_len = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    all_indices = np.arange(padded_len, dtype=np.int32)
    return all_indices[all_indices >= N]


def run_boundary_sweep(start_size: int, end_size: int, BLOCK_SIZE: int = 64) -> dict:
    results = {}
    for n in range(start_size, end_size + 1):
        x = np.arange(n, dtype=np.float32) + 1.0
        kernel_out = process_data(x, n, BLOCK_SIZE)
        ref_out = x * 2.5 + 1.0
        results[n] = bool(np.allclose(kernel_out, ref_out))
    return results
