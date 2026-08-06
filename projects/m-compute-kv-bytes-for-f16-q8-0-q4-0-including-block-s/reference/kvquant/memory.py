import numpy as np
from kvquant.calc import compute_kv_bytes


def measure_kv_footprint(
    num_tokens: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: str,
) -> dict:
    """Measure allocated memory footprint for KV cache buffers."""
    theoretical = compute_kv_bytes(
        num_tokens, num_layers, num_kv_heads, head_dim, dtype
    )
    total_elements = num_tokens * 2 * num_layers * num_kv_heads * head_dim
    num_blocks = total_elements // 32

    if dtype == "f16":
        buf = np.zeros(total_elements, dtype=np.float16)
        allocated = int(buf.nbytes)
    elif dtype == "q8_0":
        scales = np.zeros(num_blocks, dtype=np.float16)
        quants = np.zeros((num_blocks, 32), dtype=np.int8)
        allocated = int(scales.nbytes + quants.nbytes)
    elif dtype == "q4_0":
        scales = np.zeros(num_blocks, dtype=np.float16)
        quants = np.zeros((num_blocks, 16), dtype=np.uint8)
        allocated = int(scales.nbytes + quants.nbytes)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return {
        "theoretical_bytes": theoretical,
        "allocated_bytes": allocated,
    }
