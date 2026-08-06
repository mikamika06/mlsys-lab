def reconstruct_call_sequence(M, N, K, mb, nb, kb):
    """Reconstruct the sequence of BRGeMM microkernel calls for a tiled GEMM."""
    calls = []
    for m in range(0, M, mb):
        m_len = min(mb, M - m)
        for n in range(0, N, nb):
            n_len = min(nb, N - n)
            batch_size = (K + kb - 1) // kb
            a_offsets = []
            b_offsets = []
            for k_idx in range(batch_size):
                k_start = k_idx * kb
                a_off = m * K + k_start
                b_off = k_start * N + n
                a_offsets.append(a_off)
                b_offsets.append(b_off)
            c_off = m * N + n
            calls.append({
                "m_start": m,
                "n_start": n,
                "m_len": m_len,
                "n_len": n_len,
                "batch_size": batch_size,
                "a_offsets": a_offsets,
                "b_offsets": b_offsets,
                "c_offset": c_off,
            })
    return calls
