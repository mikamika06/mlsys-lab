import numpy as np
from rotary.fusion import fused_rotary_attention


def sweep_num_splits(q, k, v, cos, sin, split_candidates):
    results = {}
    ref_out, _, _ = fused_rotary_attention(q, k, v, cos, sin, num_splits=1)
    for s in split_candidates:
        out, _, _ = fused_rotary_attention(q, k, v, cos, sin, num_splits=s)
        err = np.max(np.abs(out - ref_out))
        results[s] = {"max_err": float(err), "latency_score": float(1.0 + 0.05 * s)}
    best_split = min(split_candidates, key=lambda x: results[x]["latency_score"])
    return results, best_split


def decode_latency_curve(seq_lens, head_dim):
    curve = []
    for sl in seq_lens:
        simulated_lat = float(sl * head_dim * 0.001 + 0.1)
        curve.append({"seq_len": int(sl), "latency_ms": simulated_lat})
    return curve
