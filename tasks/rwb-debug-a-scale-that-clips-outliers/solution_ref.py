import numpy as np


def quantize_absmax(x: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    max_abs = 0.0
    for val in x.ravel():
        val_abs = float(abs(val))
        if val_abs > max_abs:
            max_abs = val_abs
    scale = max_abs / 127.0 if max_abs != 0.0 else 1.0
    q_list = []
    for val in x.ravel():
        div = float(val) / scale
        rnd = round(div)
        if rnd < -127:
            clipped = -127
        elif rnd > 127:
            clipped = 127
        else:
            clipped = int(rnd)
        q_list.append(clipped)
    q = np.array(q_list, dtype=np.int8).reshape(x.shape)
    return q, float(scale)
