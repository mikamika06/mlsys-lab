import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    res = {
        "mask_correct": 0.0,
        "attention_no_leakage": 0.0,
        "attention_matches_unpacked": 0.0,
    }

    try:
        from seqpack.pack import pack_sequences, measure_attention_leakage
        from seqpack.attention import create_block_diagonal_mask, compute_packed_attention
    except Exception:
        return res

    seq_ids = np.array([0, 0, 0, 1, 1, -1], dtype=np.int64)
    expected_mask = np.array([
        [True, False, False, False, False, False],
        [True, True, False, False, False, False],
        [True, True, True, False, False, False],
        [False, False, False, True, False, False],
        [False, False, False, True, True, False],
        [False, False, False, False, False, False],
    ], dtype=bool)

    try:
        m = create_block_diagonal_mask(seq_ids)
        if np.array_equal(m, expected_mask):
            res["mask_correct"] = 1.0
    except Exception:
        pass

    rng = np.random.RandomState(123)
    L, D = len(seq_ids), 8
    Q = rng.randn(L, D)
    K = rng.randn(L, D)
    V = rng.randn(L, D)

    try:
        out, weights = compute_packed_attention(Q, K, V, seq_ids)
        leak = measure_attention_leakage(weights, seq_ids)
        if abs(leak) < 1e-6:
            res["attention_no_leakage"] = 1.0

        Q1, K1, V1 = Q[3:5], K[3:5], V[3:5]
        seq1_ids = np.array([0, 0], dtype=np.int64)
        out1_ref, _ = compute_packed_attention(Q1, K1, V1, seq1_ids)

        if np.allclose(out[3:5], out1_ref, atol=1e-5):
            res["attention_matches_unpacked"] = 1.0
    except Exception:
        pass

    return res
