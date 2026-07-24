import numpy as np


def unpack_q4k_scales_mins(packed: np.ndarray):
    """Unpack the 12-byte Q4_K scales/mins blob into 8 scales + 8 mins.

    Faithful port of ggml's ``get_scale_min_k4``:

        if (j < 4) {
            d = q[j] & 63;
            m = q[j + 4] & 63;
        } else {
            d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
            m = (q[j+4] >>  4) | ((q[j  ] >> 6) << 4);
        }

    Parameters
    ----------
    packed : np.ndarray, shape (12,), uint8
        The packed scales/mins blob.

    Returns
    -------
    (scales, mins) : tuple[np.ndarray, np.ndarray]
        Each shape (8,), dtype uint8, values in [0, 63].
    """
    q = np.asarray(packed, dtype=np.uint8)
    scales = np.zeros(8, dtype=np.uint8)
    mins = np.zeros(8, dtype=np.uint8)
    for j in range(8):
        if j < 4:
            d = int(q[j] & 63)
            m = int(q[j + 4] & 63)
        else:
            d = int((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4))
            m = int((q[j + 4] >> 4) | ((q[j] >> 6) << 4))
        scales[j] = d
        mins[j] = m
    return scales, mins
