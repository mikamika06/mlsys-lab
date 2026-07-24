import numpy as np

def unpack_dequant_qint4(packed: np.ndarray, scale: float) -> np.ndarray:
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    vals = np.concatenate([lo, hi], axis=0).astype(np.int16)
    vals = np.where(vals > 7, vals - 16, vals)
    return scale * vals.astype(np.float32)
