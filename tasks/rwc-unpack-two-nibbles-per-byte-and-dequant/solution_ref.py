import numpy as np


def unpack_nf4(packed: np.ndarray, absmax: float) -> tuple[np.ndarray, np.ndarray]:
    levels = np.array(
        [
            -1.0000, -0.6962, -0.5251, -0.3949,
            -0.2844, -0.1848, -0.0911, 0.0,
             0.0796,  0.1609,  0.2461,  0.3379,
             0.4407,  0.5626,  0.7229, 1.0000,
        ],
        dtype=np.float64,
    )

    packed = np.asarray(packed, dtype=np.uint8)
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[0::2] = packed >> np.uint8(4)
    codes[1::2] = packed & np.uint8(15)

    weights = levels[codes.astype(np.int64)] * float(absmax)
    return codes, weights
