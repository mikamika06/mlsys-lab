import numpy as np
import ref


def check(workdir):
    from ringattn.ring import ring_attention

    rng = np.random.default_rng(42)
    b, seq_total, h, d = 1, 16, 2, 8
    world_size = 4

    q_data = rng.standard_normal((b, seq_total, h, d)).astype(np.float32)
    k_data = rng.standard_normal((b, seq_total, h, d)).astype(np.float32)
    v_data = rng.standard_normal((b, seq_total, h, d)).astype(np.float32)

    chunk_size = seq_total // world_size
    q_chunks = [q_data[:, i*chunk_size:(i+1)*chunk_size] for i in range(world_size)]
    k_chunks = [k_data[:, i*chunk_size:(i+1)*chunk_size] for i in range(world_size)]
    v_chunks = [v_data[:, i*chunk_size:(i+1)*chunk_size] for i in range(world_size)]

    ref_outs = ref.reference_ring_attention(q_chunks, k_chunks, v_chunks)
    got_outs = ring_attention(q_chunks, k_chunks, v_chunks)

    max_err = 0.0
    for r_out, g_out in zip(ref_outs, got_outs):
        err = np.max(np.abs(r_out - g_out))
        if err > max_err:
            max_err = err

    return {"max_abs_err": float(max_err)}
