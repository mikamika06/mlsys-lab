import math
import numpy as np


def ring_step(state, Q, K_block, V_block, scale):
    Q = np.asarray(Q, dtype=np.float64)
    K_block = np.asarray(K_block, dtype=np.float64)
    V_block = np.asarray(V_block, dtype=np.float64)

    n = Q.shape[0]
    dk = K_block.shape[1]
    dv = V_block.shape[1]
    m_block = K_block.shape[0]

    if state is None:
        m = np.full(n, -float("inf"), dtype=np.float64)
        l = np.zeros(n, dtype=np.float64)
        o = np.zeros((n, dv), dtype=np.float64)
    else:
        m, l, o = state

    scores = np.empty((n, m_block), dtype=np.float64)
    for i in range(n):
        for j in range(m_block):
            dot = 0.0
            for d in range(dk):
                dot += Q[i, d] * K_block[j, d]
            scores[i, j] = dot * scale

    block_max = np.empty(n, dtype=np.float64)
    for i in range(n):
        mx = -float("inf")
        for j in range(m_block):
            val = scores[i, j]
            if val > mx:
                mx = val
        block_max[i] = mx

    new_m = np.empty(n, dtype=np.float64)
    for i in range(n):
        val_m = m[i]
        val_bm = block_max[i]
        if val_bm > val_m:
            new_m[i] = val_bm
        else:
            new_m[i] = val_m

    carry = np.empty(n, dtype=np.float64)
    for i in range(n):
        val_m = m[i]
        if not math.isfinite(val_m):
            carry[i] = 0.0
        else:
            carry[i] = math.exp(val_m - new_m[i])

    block_exp = np.empty((n, m_block), dtype=np.float64)
    for i in range(n):
        nm = new_m[i]
        for j in range(m_block):
            block_exp[i, j] = math.exp(scores[i, j] - nm)

    new_l = np.empty(n, dtype=np.float64)
    for i in range(n):
        sum_exp = 0.0
        for j in range(m_block):
            sum_exp += block_exp[i, j]
        new_l[i] = carry[i] * l[i] + sum_exp

    new_o = np.empty((n, dv), dtype=np.float64)
    for i in range(n):
        c = carry[i]
        for col in range(dv):
            s = 0.0
            for j in range(m_block):
                s += block_exp[i, j] * V_block[j, col]
            new_o[i, col] = c * o[i, col] + s

    return new_m, new_l, new_o


def ring_output(state) -> np.ndarray:
    _, l, o = state
    n = o.shape[0]
    dv = o.shape[1]
    res = np.empty((n, dv), dtype=np.float64)
    for i in range(n):
        inv_l = 1.0 / l[i]
        for col in range(dv):
            res[i, col] = o[i, col] * inv_l
    return res
