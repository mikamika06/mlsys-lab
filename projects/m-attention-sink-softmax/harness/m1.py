import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from attnsink.sink_softmax import attention_sink_softmax

    test_cases = [
        (128, 16, 16, 0, 16, 42),
        (256, 32, 32, 4, 32, 101),
        (512, 32, 64, 8, 64, 2024),
        (100, 16, 16, 4, 10, 7),
    ]

    max_err = 0.0
    for seq_len, d_k, d_v, sink_size, window_size, seed in test_cases:
        Q, K, V = ref.make_inputs(seq_len, d_k, d_v, seed)
        want_out, want_lse = ref.attention_sink_softmax(Q, K, V, sink_size, window_size)

        try:
            got_out, got_lse = attention_sink_softmax(Q, K, V, sink_size, window_size)
        except Exception as e:
            return {"rel_err": 1.0, "_note": f"Exception raised: {type(e).__name__}: {e}"}

        err_out = float(np.linalg.norm(got_out - want_out) / (np.linalg.norm(want_out) + 1e-12))
        err_lse = float(np.linalg.norm(got_lse - want_lse) / (np.linalg.norm(want_lse) + 1e-12))
        max_err = max(max_err, err_out, err_lse)

    return {"rel_err": max_err}
