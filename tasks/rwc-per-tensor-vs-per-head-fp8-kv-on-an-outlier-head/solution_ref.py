import math
import numpy as np


def _quant_dequant(x, axis=None):
    x = np.asarray(x, dtype=np.float64)
    if axis is None:
        max_val = 0.0
        it = np.nditer(x, flags=["multi_index"])
        while not it.finished:
            val = abs(it[0])
            if val > max_val:
                max_val = val
            it.iternext()
        scale = max_val / 127.0
        if scale == 0:
            out = np.zeros(x.shape, dtype=np.float64)
            return out
        out = np.empty(x.shape, dtype=np.float64)
        it_in = np.nditer(x, flags=["multi_index"])
        while not it_in.finished:
            idx = it_in.multi_index
            val = x[idx] / scale
            rounded = round(val)
            if rounded < -127.0:
                clipped = -127.0
            elif rounded > 127.0:
                clipped = 127.0
            else:
                clipped = rounded
            out[idx] = clipped * scale
            it_in.iternext()
        return out
    else:
        shape = x.shape
        if isinstance(axis, int):
            axes = (axis,)
        else:
            axes = axis
        reduced_shape = list(shape)
        for ax in axes:
            reduced_shape[ax] = 1
        max_arr = np.zeros(reduced_shape, dtype=np.float64)
        it = np.nditer(x, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            val = abs(x[idx])
            red_idx = list(idx)
            for ax in axes:
                red_idx[ax] = 0
            red_idx = tuple(red_idx)
            if val > max_arr[red_idx]:
                max_arr[red_idx] = val
            it.iternext()
        scale = max_arr / 127.0
        scale_checked = np.empty(reduced_shape, dtype=np.float64)
        it_s = np.nditer(scale, flags=["multi_index"])
        while not it_s.finished:
            idx = it_s.multi_index
            s = scale[idx]
            scale_checked[idx] = 1.0 if s == 0 else s
            it_s.iternext()
        out = np.empty(shape, dtype=np.float64)
        it_in = np.nditer(x, flags=["multi_index"])
        while not it_in.finished:
            idx = it_in.multi_index
            red_idx = list(idx)
            for ax in axes:
                red_idx[ax] = 0
            red_idx = tuple(red_idx)
            s = scale_checked[red_idx]
            val = x[idx] / s
            rounded = round(val)
            if rounded < -127.0:
                clipped = -127.0
            elif rounded > 127.0:
                clipped = 127.0
            else:
                clipped = rounded
            out[idx] = clipped * s
            it_in.iternext()
        return out


def _attention(Q, K, V):
    batch_size, seq_q, dim = Q.shape
    _, seq_k, _ = K.shape
    logits = np.empty((batch_size, seq_q, seq_k), dtype=np.float64)
    scale = math.sqrt(dim)
    for b in range(batch_size):
        for i in range(seq_q):
            for j in range(seq_k):
                acc = 0.0
                for d in range(dim):
                    acc += Q[b, i, d] * K[b, j, d]
                logits[b, i, j] = acc / scale
    for b in range(batch_size):
        for i in range(seq_q):
            max_val = logits[b, i, 0]
            for j in range(1, seq_k):
                if logits[b, i, j] > max_val:
                    max_val = logits[b, i, j]
            for j in range(seq_k):
                logits[b, i, j] -= max_val
    weights = np.empty((batch_size, seq_q, seq_k), dtype=np.float64)
    for b in range(batch_size):
        for i in range(seq_q):
            for j in range(seq_k):
                weights[b, i, j] = math.exp(logits[b, i, j])
    for b in range(batch_size):
        for i in range(seq_q):
            s = 0.0
            for j in range(seq_k):
                s += weights[b, i, j]
            for j in range(seq_k):
                weights[b, i, j] /= s
    v_dim = V.shape[-1]
    out = np.empty((batch_size, seq_q, v_dim), dtype=np.float64)
    for b in range(batch_size):
        for i in range(seq_q):
            for d in range(v_dim):
                acc = 0.0
                for j in range(seq_k):
                    acc += weights[b, i, j] * V[b, j, d]
                out[b, i, d] = acc
    return out


def choose_kv_fp8_scheme(Q, K, V):
    ref = _attention(Q, K, V)

    tensor_out = _attention(Q, _quant_dequant(K), _quant_dequant(V))
    diff_tensor = tensor_out - ref
    norm_sq_tensor = 0.0
    it = np.nditer(diff_tensor)
    while not it.finished:
        val = it[0]
        norm_sq_tensor += val * val
        it.iternext()
    norm_tensor = math.sqrt(norm_sq_tensor)

    norm_ref_sq = 0.0
    it_ref = np.nditer(ref)
    while not it_ref.finished:
        val = it_ref[0]
        norm_ref_sq += val * val
        it_ref.iternext()
    norm_ref = math.sqrt(norm_ref_sq)

    tensor_error = norm_tensor / (norm_ref + 1e-12)

    head_out = _attention(
        Q,
        _quant_dequant(K, axis=(1, 2)),
        _quant_dequant(V, axis=(1, 2)),
    )
    diff_head = head_out - ref
    norm_sq_head = 0.0
    it = np.nditer(diff_head)
    while not it.finished:
        val = it[0]
        norm_sq_head += val * val
        it.iternext()
    norm_head = math.sqrt(norm_sq_head)

    head_error = norm_head / (norm_ref + 1e-12)

    scheme = "per_head" if head_error < tensor_error else "per_tensor"
    return float(tensor_error), float(head_error), scheme
