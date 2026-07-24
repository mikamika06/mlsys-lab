import numpy as np

def _oracle(layers, ctx_len, d_model, num_heads, groups, dtype):
    # Use NumPy arrays to compute sizes exactly
    d_head = d_model // num_heads

    k_mha = np.empty((layers, ctx_len, num_heads, d_head), dtype=dtype)
    v_mha = np.empty_like(k_mha)
    size_mha = (k_mha.nbytes + v_mha.nbytes)

    k_gqa = np.empty((layers, ctx_len, groups, d_head), dtype=dtype)
    v_gqa = np.empty_like(k_gqa)
    size_gqa = (k_gqa.nbytes + v_gqa.nbytes)

    k_mqa = np.empty((layers, ctx_len, d_head), dtype=dtype)
    v_mqa = np.empty_like(k_mqa)
    size_mqa = (k_mqa.nbytes + v_mqa.nbytes)

    return {"mha": int(size_mha), "gqa": int(size_gqa), "mqa": int(size_mqa)}

def grade(sol, fx) -> dict:
    # Fixed test parameters
    layers = 4
    ctx_len = 2048
    d_model = 4096
    num_heads = 32
    groups = 8
    dtype = np.float16

    ref = _oracle(layers, ctx_len, d_model, num_heads, groups, dtype)

    try:
        got = sol.kv_cache_bytes(
            layers=layers,
            ctx_len=ctx_len,
            d_model=d_model,
            num_heads=num_heads,
            groups=groups,
            dtype=dtype
        )
    except Exception:
        return {"gqa_mha_ratio": 0.0, "mqa_mha_ratio": 0.0}

    # Compute ratios from solution and reference
    gqa_ratio = got["gqa"] / ref["mha"]
    mqa_ratio = got["mqa"] / ref["mha"]

    expected_gqa_ratio = ref["gqa"] / ref["mha"]
    expected_mqa_ratio = ref["mqa"] / ref["mha"]

    eps = 1e-9

    def rel_err(a, b):
        return abs(a - b) / max(abs(b), 1e-12)

    gqa_metric = 1.0 if rel_err(gqa_ratio, expected_gqa_ratio) <= eps else 0.0
    mqa_metric = 1.0 if rel_err(mqa_ratio, expected_mqa_ratio) <= eps else 0.0

    return {"gqa_mha_ratio": gqa_metric, "mqa_mha_ratio": mqa_metric}
