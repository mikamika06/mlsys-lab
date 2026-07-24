import numpy as np

FP4_MAX = 6.0
FP8_MAX = 448.0

E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def nvfp4_two_level_quantize(w: np.ndarray, block_size: int = 16):
    """NVFP4 two-level scaling: a single per-tensor fp32 scale, an E4M3
    per-block scale for every `block_size`-element block, and E2M1 (4-bit
    float) elements.

    Returns (global_scale, block_scales_e4m3, codes, dequantized):
      global_scale: python float, tensor_amax / (6 * 448).
      block_scales_e4m3: float64 array, shape (len(w) // block_size,), each
        entry snapped to the nearest non-negative E4M3 representable value.
      codes: float64 array, same shape as `w`, each entry one of the signed
        E2M1 values {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
      dequantized: float64 array, same shape as `w`,
        codes * (block_scales_e4m3 * global_scale) broadcast per block.
    """
    raise NotImplementedError('your code here')
