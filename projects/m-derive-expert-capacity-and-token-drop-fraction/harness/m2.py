import numpy as np


def check(workdir):
    from moe.dispatch import grouped_gemm_dispatch, naive_expert_loop

    out = {"dispatch_matched": 0.0}
    np.random.seed(123)
    tokens = np.random.randn(16, 8).astype(np.float32)
    topk_indices = np.random.randint(0, 4, size=(16, 2))
    topk_weights = np.random.rand(16, 2).astype(np.float32)
    expert_weights = np.random.randn(4, 8, 8).astype(np.float32)
    capacity = 10

    try:
        out1 = grouped_gemm_dispatch(tokens, topk_indices, topk_weights, expert_weights, capacity)
        out2 = naive_expert_loop(tokens, topk_indices, topk_weights, expert_weights, capacity)
        if out1 is not None and out2 is not None and np.allclose(out1, out2, rtol=1e-5, atol=1e-5):
            out["dispatch_matched"] = 1.0
    except Exception as e:
        out["_note"] = str(e)
    return out
