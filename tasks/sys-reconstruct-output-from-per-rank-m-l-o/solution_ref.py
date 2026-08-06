import math
import numpy as np


def reconstruct_output(states):
    m = np.stack([np.asarray(s[0], dtype=np.float64) for s in states], axis=0)
    l = np.stack([np.asarray(s[1], dtype=np.float64) for s in states], axis=0)
    o = np.stack([np.asarray(s[2], dtype=np.float64) for s in states], axis=0)

    S, num_queries = m.shape
    d_v = o.shape[2]

    global_m = np.zeros(num_queries, dtype=np.float64)
    for q_idx in range(num_queries):
        max_val = m[0, q_idx]
        for s_idx in range(1, S):
            if m[s_idx, q_idx] > max_val:
                max_val = m[s_idx, q_idx]
        global_m[q_idx] = max_val

    scale = np.zeros((S, num_queries), dtype=np.float64)
    for s_idx in range(S):
        for q_idx in range(num_queries):
            scale[s_idx, q_idx] = math.exp(m[s_idx, q_idx] - global_m[q_idx])

    contributions = np.zeros((S, num_queries), dtype=np.float64)
    for s_idx in range(S):
        for q_idx in range(num_queries):
            contributions[s_idx, q_idx] = scale[s_idx, q_idx] * l[s_idx, q_idx]

    denom = np.zeros(num_queries, dtype=np.float64)
    for q_idx in range(num_queries):
        acc = 0.0
        for s_idx in range(S):
            acc += contributions[s_idx, q_idx]
        denom[q_idx] = acc

    output = np.zeros((num_queries, d_v), dtype=np.float64)
    for q_idx in range(num_queries):
        for d_idx in range(d_v):
            acc = 0.0
            for s_idx in range(S):
                acc += scale[s_idx, q_idx] * o[s_idx, q_idx, d_idx]
            output[q_idx, d_idx] = acc / denom[q_idx]

    global_lse = np.zeros(num_queries, dtype=np.float64)
    for q_idx in range(num_queries):
        global_lse[q_idx] = global_m[q_idx] + math.log(denom[q_idx])

    rank_mass = np.zeros((S, num_queries), dtype=np.float64)
    for s_idx in range(S):
        for q_idx in range(num_queries):
            rank_mass[s_idx, q_idx] = contributions[s_idx, q_idx] / denom[q_idx]

    return output, global_lse, rank_mass
