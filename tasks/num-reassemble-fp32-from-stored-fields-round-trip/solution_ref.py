import numpy as np

def reassemble_fp32(signs: np.ndarray,
                    exps: np.ndarray,
                    mantissas: np.ndarray) -> np.ndarray:
    bits = (signs.astype(np.uint32) << 31) | \
           (exps.astype(np.uint32) << 23) | \
           mantissas.astype(np.uint32)
    return bits.view(np.float32)
