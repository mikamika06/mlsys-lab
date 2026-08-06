import numpy as np

def unpack_dequant_qint4(packed: np.ndarray, scale: float) -> np.ndarray:
    res = []
    for x in packed:
        v = int(x) & 0xF
        if v > 7:
            v -= 16
        res.append(v)
    for x in packed:
        v = (int(x) >> 4) & 0xF
        if v > 7:
            v -= 16
        res.append(v)
    return scale * np.array(res, dtype=np.float32)
