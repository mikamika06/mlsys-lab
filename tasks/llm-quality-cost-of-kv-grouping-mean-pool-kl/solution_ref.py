import numpy as np
import math


def _softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    shape = x.shape
    ndim = x.ndim
    if axis < 0:
        axis = ndim + axis
    
    out = np.empty_like(x)
    other_dims = [dim for dim in range(ndim) if dim != axis]

    def iterate_other(other_dim_pos, current_indices_dict):
        if other_dim_pos == len(other_dims):
            full_idx = []
            for d_idx in range(ndim):
                if d_idx == axis:
                    full_idx.append(slice(None))
                else:
                    full_idx.append(current_indices_dict[d_idx])
            full_idx_tuple = tuple(full_idx)
            
            slice_1d = x[full_idx_tuple]
            n_elem = shape[axis]
            max_val = slice_1d[0]
            for i in range(1, n_elem):
                val = slice_1d[i]
                if val > max_val:
                    max_val = val
            
            sum_val = 0.0
            out_slice = np.empty_like(slice_1d)
            for i in range(n_elem):
                e_val = math.exp(slice_1d[i] - max_val)
                out_slice[i] = e_val
                sum_val += e_val
            for i in range(n_elem):
                out_slice[i] /= sum_val
            
            out[full_idx_tuple] = out_slice
            return

        d_idx = other_dims[other_dim_pos]
        for i in range(shape[d_idx]):
            current_indices_dict[d_idx] = i
            iterate_other(other_dim_pos + 1, current_indices_dict)

    iterate_other(0, {})
    return out


def kv_grouping_mean_pool_kl(q, k, group_size):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    b, h, tq, d = q.shape
    tk = k.shape[2]
    sqrt_d = math.sqrt(d)

    full_logits = np.empty((b, h, tq, tk), dtype=np.float64)
    for bi in range(b):
        for hi in range(h):
            for i in range(tq):
                for j in range(tk):
                    acc = 0.0
                    for di in range(d):
                        acc += q[bi, hi, i, di] * k[bi, hi, j, di]
                    full_logits[bi, hi, i, j] = acc / sqrt_d

    full = _softmax(full_logits, axis=-1)

    groups = h // group_size
    pooled = np.empty((b, groups, tk, d), dtype=np.float64)
    for bi in range(b):
        for g in range(groups):
            for j in range(tk):
                for di in range(d):
                    acc = 0.0
                    for gs in range(group_size):
                        h_idx = g * group_size + gs
                        acc += k[bi, h_idx, j, di]
                    pooled[bi, g, j, di] = acc / group_size

    grouped = np.empty_like(full)
    for head in range(h):
        q_head = q[:, head]
        pooled_group = pooled[:, head // group_size]
        logits = np.empty((b, tq, tk), dtype=np.float64)
        for bi in range(b):
            for i in range(tq):
                for j in range(tk):
                    acc = 0.0
                    for di in range(d):
                        acc += q_head[bi, i, di] * pooled_group[bi, j, di]
                    logits[bi, i, j] = acc / sqrt_d
        grouped[:, head] = _softmax(logits, axis=-1)

    kl = np.empty((b, h, tq), dtype=np.float64)
    for bi in range(b):
        for hi in range(h):
            for i in range(tq):
                acc = 0.0
                for j in range(tk):
                    f_val = full[bi, hi, i, j]
                    g_val = grouped[bi, hi, i, j]
                    acc += f_val * (math.log(f_val + 1e-12) - math.log(g_val + 1e-12))
                kl[bi, hi, i] = acc

    total_sum = 0.0
    count = 0
    for bi in range(b):
        for hi in range(h):
            for i in range(tq):
                total_sum += kl[bi, hi, i]
                count += 1
    return float(total_sum / count)
