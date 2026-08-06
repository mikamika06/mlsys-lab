import numpy as np


def compare_2_4_masks(weights, logits):
    weights = np.asarray(weights, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)

    num_rows = weights.shape[0]
    num_cols = weights.shape[1]

    soft_mask = [[0.0] * num_cols for _ in range(num_rows)]
    mag_mask = [[0.0] * num_cols for _ in range(num_rows)]

    for i in range(num_rows):
        soft_indices = sorted(range(num_cols), key=lambda j: (-logits[i, j], j))[:2]
        mag_indices = sorted(range(num_cols), key=lambda j: (-abs(weights[i, j]), j))[:2]

        for j in soft_indices:
            soft_mask[i][j] = 1.0
        for j in mag_indices:
            mag_mask[i][j] = 1.0

    soft_retained_list = []
    magnitude_retained_list = []
    soft_error_list = []
    magnitude_error_list = []

    soft_total = 0.0
    magnitude_total = 0.0

    for i in range(num_rows):
        s_ret = 0.0
        m_ret = 0.0
        s_err = 0.0
        m_err = 0.0

        for j in range(num_cols):
            w = weights[i, j]
            abs_w = abs(w)
            s_m = soft_mask[i][j]
            m_m = mag_mask[i][j]

            s_ret += abs_w * s_m
            m_ret += abs_w * m_m

            diff_s = w - (w * s_m)
            s_err += diff_s * diff_s

            diff_m = w - (w * m_m)
            m_err += diff_m * diff_m

        soft_retained_list.append(s_ret)
        magnitude_retained_list.append(m_ret)
        soft_error_list.append(s_err)
        magnitude_error_list.append(m_err)

        soft_total += s_err
        magnitude_total += m_err

    if soft_total < magnitude_total:
        better = "soft"
    elif magnitude_total < soft_total:
        better = "magnitude"
    else:
        better = "tie"

    return {
        "soft_retained": np.asarray(soft_retained_list, dtype=np.float64),
        "magnitude_retained": np.asarray(magnitude_retained_list, dtype=np.float64),
        "soft_error": np.asarray(soft_error_list, dtype=np.float64),
        "magnitude_error": np.asarray(magnitude_error_list, dtype=np.float64),
        "better": better,
    }
