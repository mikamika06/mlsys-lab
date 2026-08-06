import numpy as np


def quantize_kv_group_affine(kv, kv_bits, kv_group_size):
    x = np.asarray(kv, dtype=np.float64)
    shape = x.shape
    leading_shape = shape[:-1]
    last_dim = shape[-1]
    groups = last_dim // kv_group_size

    g_shape = leading_shape + (groups, kv_group_size)
    scale_shape = leading_shape + (groups, 1)
    zero_shape = leading_shape + (groups, 1)

    scale_arr = np.empty(scale_shape, dtype=np.float64)
    zero_arr = np.empty(zero_shape, dtype=np.float64)
    q_arr = np.empty(g_shape, dtype=np.int32)

    qmax = (1 << kv_bits) - 1
    g = x.reshape(g_shape)

    for idx in np.ndindex(g_shape[:-1]):
        group_slice = g[idx]
        mn = group_slice[0]
        mx = group_slice[0]
        for val in group_slice[1:]:
            if val < mn:
                mn = val
            if val > mx:
                mx = val

        sc = (mx - mn) / qmax
        if sc == 0.0:
            sc = 1.0
        scale_arr[idx] = sc

        z = round(-mn / sc)
        if z < 0:
            z = 0
        elif z > qmax:
            z = qmax
        zero_arr[idx] = z

        for i_elem in range(kv_group_size):
            val = group_slice[i_elem]
            q_val = round(val / sc + z)
            if q_val < 0:
                q_val = 0
            elif q_val > qmax:
                q_val = qmax
            q_arr[idx + (i_elem,)] = int(q_val)

    return q_arr, scale_arr, zero_arr
