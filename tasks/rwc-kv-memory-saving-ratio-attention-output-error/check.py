import numpy as np

def _oracle(kv_fp16, kv_fp8, query):
    # ratio of bytes
    ratio = kv_fp16.nbytes / kv_fp8.nbytes

    # dequantise fp8 to float32 in [-1, 1]
    kv_fp8_flt = kv_fp8.astype(np.float32) / 127.0

    K_fp16, V_fp16 = kv_fp16[0], kv_fp16[1]
    K_fp8 , V_fp8  = kv_fp8_flt[0], kv_fp8_flt[1]

    d = K_fp16.shape[-1]
    scale = np.sqrt(d)

    # FP16 attention
    scores_fp16 = query @ K_fp16.T / scale
    exp_fp16 = np.exp(scores_fp16 - np.max(scores_fp16, axis=-1, keepdims=True))
    softmax_fp16 = exp_fp16 / np.sum(exp_fp16, axis=-1, keepdims=True)
    out_fp16 = softmax_fp16 @ V_fp16

    # FP8 attention
    scores_fp8 = query @ K_fp8.T / scale
    exp_fp8 = np.exp(scores_fp8 - np.max(scores_fp8, axis=-1, keepdims=True))
    softmax_fp8 = exp_fp8 / np.sum(exp_fp8, axis=-1, keepdims=True)
    out_fp8 = softmax_fp8 @ V_fp8

    error = np.max(np.abs(out_fp16 - out_fp8))

    return ratio, error


def grade(sol, fx) -> dict:
    # deterministic random data
    rng = np.random.default_rng(0)

    cases = []
    for _ in range(3):
        seq_len = rng.integers(2, 5)
        dim = rng.integers(4, 8)
        kv_fp16 = rng.standard_normal((2, seq_len, dim)).astype(np.float32)
        kv_fp8 = np.round(kv_fp16 * 127).clip(-127, 127).astype(np.uint8)
        query = rng.standard_normal((seq_len, dim)).astype(np.float32)
        cases.append((kv_fp16, kv_fp8, query))

    ok_ratio = 1.0
    rel_errs = []

    for kv_fp16, kv_fp8, query in cases:
        try:
            ratio_s, err_s = sol.kv_memory_saving_ratio_and_attention_error(
                kv_fp16, kv_fp8, query)
        except Exception:
            return {"size_ratio": 0.0, "attention_output_error_rel": 1.0}

        oracle_ratio, oracle_err = _oracle(kv_fp16, kv_fp8, query)

        if not np.isclose(ratio_s, oracle_ratio, rtol=1e-12, atol=0):
            ok_ratio = 0.0

        # relative error between student's error and oracle's error
        rel_err = abs(err_s - oracle_err) / (oracle_err + 1e-12)
        rel_errs.append(rel_err)

    return {
        "size_ratio": ok_ratio,
        "attention_output_error_rel": max(rel_errs)
    }
