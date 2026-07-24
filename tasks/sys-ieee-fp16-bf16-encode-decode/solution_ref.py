import numpy as np

def encode_fp32_to_fp16_and_bf16(arr: np.ndarray):
    """Return the IEEE‑754 bit patterns for FP16 and BF16 representations of a float32 array."""
    arr = np.asarray(arr, dtype=np.float32)
    fp16_bits = arr.astype(np.float16).view(np.uint16)
    bf16_bits = (arr.view(np.uint32) >> 16).astype(np.uint16)
    return fp16_bits, bf16_bits
