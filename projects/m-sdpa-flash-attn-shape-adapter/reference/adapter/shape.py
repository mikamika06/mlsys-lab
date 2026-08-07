import numpy as np


def sdpa_to_flash(q, k, v):
    q_f = np.transpose(q, (0, 2, 1, 3))
    k_f = np.transpose(k, (0, 2, 1, 3))
    v_f = np.transpose(v, (0, 2, 1, 3))
    return q_f, k_f, v_f


def flash_to_sdpa(out):
    return np.transpose(out, (0, 2, 1, 3))
