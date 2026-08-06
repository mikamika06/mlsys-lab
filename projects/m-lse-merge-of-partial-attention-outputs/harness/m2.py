import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from ringattn.simulator import run_ring_attention
    from ringattn.analysis import compute_causal_load_imbalance

    rng = np.random.RandomState(2026)
    world_size = 4
    seq_len = 8
    head_dim = 16

    q_blocks = [rng.randn(seq_len, head_dim) for _ in range(world_size)]
    k_blocks = [rng.randn(seq_len, head_dim) for _ in range(world_size)]
    v_blocks = [rng.randn(seq_len, head_dim) for _ in range(world_size)]

    got_outs = run_ring_attention(q_blocks, k_blocks, v_blocks, is_causal=False)

    q_full = np.concatenate(q_blocks, axis=0)
    k_full = np.concatenate(k_blocks, axis=0)
    v_full = np.concatenate(v_blocks, axis=0)

    scores = np.matmul(q_full, k_full.T) / np.sqrt(head_dim)
    exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    want_full = np.matmul(attn, v_full)

    got_full = np.concatenate(got_outs, axis=0)
    ring_rel_err = float(np.max(np.abs(got_full - want_full) / (np.abs(want_full) + 1e-9)))

    want_analysis = ref.ref_compute_causal_load_imbalance(world_size=4, num_blocks=2)
    got_analysis = compute_causal_load_imbalance(world_size=4, num_blocks=2)

    load_match = 1 if (
        got_analysis.get("rank_work") == want_analysis["rank_work"] and
        abs(got_analysis.get("imbalance_ratio", 0) - want_analysis["imbalance_ratio"]) < 1e-5
    ) else 0

    return {
        "ring_rel_err": ring_rel_err,
        "load_ratio_match": float(load_match)
    }
