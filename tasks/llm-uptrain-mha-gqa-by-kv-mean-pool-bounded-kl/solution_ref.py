import numpy as np


def uptrain_mha_to_gqa(q, k, v, groups):
    b, h, tk, d = k.shape
    per_group = h // groups
    k_gqa = np.zeros((b, groups, tk, d), dtype=k.dtype)
    v_gqa = np.zeros((b, groups, tk, d), dtype=v.dtype)
    for bi in range(b):
        for gi in range(groups):
            for ti in range(tk):
                for di in range(d):
                    k_sum = 0.0
                    v_sum = 0.0
                    for p in range(per_group):
                        hi = gi * per_group + p
                        k_sum += k[bi, hi, ti, di]
                        v_sum += v[bi, hi, ti, di]
                    k_gqa[bi, gi, ti, di] = k_sum / per_group
                    v_gqa[bi, gi, ti, di] = v_sum / per_group
    return k_gqa, v_gqa
