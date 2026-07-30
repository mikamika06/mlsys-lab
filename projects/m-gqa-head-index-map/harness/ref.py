import numpy as np

CONFIGS = [(4, 1), (4, 2), (4, 4), (8, 2), (8, 4), (12, 3)]

CASES = [
    {"num_q_heads": 4, "num_kv_heads": 2, "seq_len": 6, "head_dim": 8, "seed": 0},
    {"num_q_heads": 8, "num_kv_heads": 4, "seq_len": 5, "head_dim": 4, "seed": 1},
    {"num_q_heads": 6, "num_kv_heads": 1, "seq_len": 7, "head_dim": 3, "seed": 2},
    {"num_q_heads": 9, "num_kv_heads": 3, "seq_len": 4, "head_dim": 5, "seed": 3},
]


def build_head_map(num_q_heads, num_kv_heads):
    group = num_q_heads // num_kv_heads
    return np.repeat(np.arange(num_kv_heads), group).tolist()


def build_query_groups(num_q_heads, num_kv_heads):
    head_map = build_head_map(num_q_heads, num_kv_heads)
    groups = [[] for _ in range(num_kv_heads)]
    for q, k in enumerate(head_map):
        groups[k].append(q)
    return groups


def expand_kv(kv, num_q_heads, num_kv_heads):
    group = num_q_heads // num_kv_heads
    out = np.empty((num_q_heads,) + kv.shape[1:], dtype=kv.dtype)
    for k in range(num_kv_heads):
        for j in range(group):
            out[k * group + j] = kv[k]
    return out


def attention(q, k, v, num_kv_heads, causal=True):
    num_q_heads, seq_q, head_dim = q.shape
    seq_k = k.shape[1]
    k_exp = expand_kv(k, num_q_heads, num_kv_heads)
    v_exp = expand_kv(v, num_q_heads, num_kv_heads)
    scale = 1.0 / np.sqrt(head_dim)
    out = np.zeros((num_q_heads, seq_q, head_dim), dtype=np.float64)
    for h in range(num_q_heads):
        for i in range(seq_q):
            row = np.full(seq_k, -np.inf)
            limit = i + 1 if causal else seq_k
            for j in range(limit):
                row[j] = float(np.dot(q[h, i], k_exp[h, j])) * scale
            row = row - np.max(row)
            w = np.exp(row)
            w = w / w.sum()
            out[h, i] = w @ v_exp[h]
    return out


def make_case(case):
    rng = np.random.default_rng(case["seed"])
    nq, nkv = case["num_q_heads"], case["num_kv_heads"]
    seq, hd = case["seq_len"], case["head_dim"]
    q = rng.standard_normal((nq, seq, hd))
    k = rng.standard_normal((nkv, seq, hd))
    v = rng.standard_normal((nkv, seq, hd))
    return q, k, v
