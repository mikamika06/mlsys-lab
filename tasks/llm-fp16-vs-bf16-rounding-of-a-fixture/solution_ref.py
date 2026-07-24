import numpy as np

def compare_rounding(x: np.ndarray):
    """
    Return two arrays:
      - fp16_arr: x rounded to FP16 and cast back to float32.
      - bf16_arr: x rounded to BF16 by truncating the lower 16 bits of each
        32‑bit word, then interpreted as a new float32 value.

    Parameters
    ----------
    x : np.ndarray
        One‑dimensional array of dtype float32.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (fp16_arr, bf16_arr), both dtype float32.
    """
    # FP16 rounding via NumPy cast
    fp16_arr = x.astype(np.float16).astype(np.float32)

    # BF16 emulation: keep upper 16 bits of the uint32 representation
    bits = x.view(np.uint32)
    bf16_bits = (bits >> 16) & 0xFFFF
    bf16_arr = (bf16_bits.astype(np.uint32) << 16).view(np.float32)

    return fp16_arr, bf16_arr
