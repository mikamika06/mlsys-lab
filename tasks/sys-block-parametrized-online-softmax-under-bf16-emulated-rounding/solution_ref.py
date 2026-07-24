import numpy as np


def _bf16_round(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> 16) & 1
    rounded = u + np.uint32(0x7FFF) + lsb
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32).astype(np.float64)


def tiled_online_softmax(x, B):
    x = np.asarray(x, dtype=np.float64)
    x = _bf16_round(x)

    m = np.float64(-np.inf)
    l = np.float64(0.0)

    for start in range(0, len(x), B):
        block = _bf16_round(x[start:start + B])
        block_max = _bf16_round(np.max(block))
        new_m = _bf16_round(max(m, block_max))

        if np.isneginf(m):
            old_term = 0.0
        else:
            old_term = _bf16_round(np.exp(_bf16_round(m - new_m)))

        l = _bf16_round(_bf16_round(l * old_term) + _bf16_round(
            np.sum(_bf16_round(np.exp(_bf16_round(block - new_m))))
        ))
        m = new_m

    out = np.empty(len(x), dtype=np.float64)
    for i in range(len(x)):
        out[i] = _bf16_round(
            _bf16_round(np.exp(_bf16_round(x[i] - m))) / _bf16_round(l)
        )
    return out
