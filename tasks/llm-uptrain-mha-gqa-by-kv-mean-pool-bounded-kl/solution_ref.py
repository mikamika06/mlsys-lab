import numpy as np


def uptrain_mha_to_gqa(q, k, v, groups):
    b, h, tk, d = k.shape
    per_group = h // groups
    k_gqa = k.reshape(b, groups, per_group, tk, d).mean(axis=2)
    v_gqa = v.reshape(b, groups, per_group, tk, d).mean(axis=2)
    return k_gqa, v_gqa
