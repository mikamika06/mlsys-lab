import numpy as np
from typing import Tuple

def size_ratio_fp16_quantized(
    fp16_weights: np.ndarray,
    int8_codes: np.ndarray,
    int8_scales: np.ndarray,
    int4_codes: np.ndarray,
    int4_scales: np.ndarray
) -> Tuple[float, float]:
    """
    Compute the memory footprint ratio of quantised representations
    (INT8 and INT4 packed) relative to an FP16 baseline.

    Parameters
    ----------
    fp16_weights : np.ndarray
        Original weights stored as 16‑bit floats.
    int8_codes : np.ndarray
        Quantised integer codes with the same spatial shape, dtype = np.int8.
    int8_scales : np.ndarray
        Scale array (per channel or per matrix).
    int4_codes : np.ndarray
        Packed 4‑bit codes stored in uint8; two codes per element.
    int4_scales : np.ndarray
        Scale array analogous to the INT8 scales.

    Returns
    -------
    Tuple[float, float]
        The first entry is the ratio for INT8,
        the second entry is the ratio for INT4.
    """
    fp_bytes = fp16_weights.nbytes

    int8_total = int8_codes.nbytes + int8_scales.nbytes
    int4_total = int4_codes.nbytes + int4_scales.nbytes

    return float(fp_bytes / int8_total), float(fp_bytes / int4_total)
