import numpy as np


def signed_zero_profile():
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.array([
            1.0 / np.float64(-0.0),
            np.float64(-0.0) + np.float64(0.0),
            np.float64(0.0) + np.float64(-0.0),
            np.copysign(np.float64(0.0), np.float64(-1.0)),
            np.copysign(np.float64(-0.0), np.float64(1.0)),
            np.copysign(np.float64(5.0), np.float64(-0.0)),
        ], dtype=np.float64)
    return [int(x) for x in np.signbit(values)]
