import numpy as np


def stochastic_round(x: np.ndarray, rng) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    
    flat_x = x.ravel()
    flat_rand = rng.random(x.shape).ravel()
    out = np.empty_like(flat_x, dtype=np.float32)

    for i in range(flat_x.shape[0]):
        val = flat_x[i]
        
        nearest16 = np.float16(val)
        nearest = np.float32(nearest16)

        if nearest <= val:
            lower16 = nearest16
            upper16 = np.nextafter(nearest16, np.float16(np.inf))
        else:
            lower16 = np.nextafter(nearest16, np.float16(-np.inf))
            upper16 = nearest16

        lower = np.float32(lower16)
        upper = np.float32(upper16)

        if lower == upper:
            prob = 0.0
        else:
            prob = (val - lower) / (upper - lower)

        if flat_rand[i] < prob:
            out[i] = upper
        else:
            out[i] = lower

    return out.reshape(x.shape).astype(np.float32)
