import numpy as np

def emulated_triton_add(a: np.ndarray,
                        b: np.ndarray,
                        block_size: int) -> np.ndarray:
    """Emulate a Triton vector-add kernel with block processing and boundary masking."""
    N = a.shape[0]
    n_blocks = (N + block_size - 1) // block_size
    output = np.empty(N, dtype=a.dtype)

    for pid in range(n_blocks):
        offset = pid * block_size
        width = min(block_size, N - offset)

        # Boundary mask: True for valid lanes, False for out-of-range
        mask = np.arange(block_size) < width

        # Zero-padded block tiles — avoids slicing past array end
        a_padded = np.zeros(block_size, dtype=a.dtype)
        b_padded = np.zeros(block_size, dtype=b.dtype)
        a_padded[:width] = a[offset:offset + width]
        b_padded[:width] = b[offset:offset + width]

        # Apply mask (zero-fill for invalid lanes)
        a_tile = np.where(mask, a_padded, 0.0)
        b_tile = np.where(mask, b_padded, 0.0)

        # Compute in the block
        c_tile = a_tile + b_tile

        # Store only the valid portion back to output
        output[offset:offset + width] = c_tile[:width]

    return output
