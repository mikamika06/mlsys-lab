import numpy as np
import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)
    from kvfp8.attn import compute_attention_error_by_position
    from kvfp8.quant import dequantize_fp8_per_head, quantize_fp8_per_head

    rng = np.random.defaultrng if hasattr(np.random, "defaultrng") else np.random.default_rng
    gen = rng(12345)

    seq_len, num_heads, head_dim = 64, 8, 64
    x = gen.normal(loc=0.0, scale=2.5, size=(seq_len, num_heads, head_dim))

    q_user, scale_user = quantize_fp8_per_head(x)
    deq_user = dequantize_fp8_per_head(q_user, scale_user)

    q_ref, scale_ref = ref.ref_quantize_fp8_per_head(x)
    deq_ref = ref.ref_dequantize_fp8_per_head(q_ref, scale_ref)

    quant_err = float(np.linalg.norm(deq_ref - deq_user) / (np.linalg.norm(deq_ref) + 1e-12))

    q = gen.normal(0, 1, size=(seq_len, num_heads, head_dim))
    k = gen.normal(0, 1, size=(seq_len, num_heads, head_dim))
    v = gen.normal(0, 1, size=(seq_len, num_heads, head_dim))

    err_user = compute_attention_error_by_position(q, k, v)
    err_ref = ref.ref_compute_attention_error_by_position(q, k, v)

    attn_err = float(np.linalg.norm(err_ref - err_user) / (np.linalg.norm(err_ref) + 1e-12))

    return {"quant_rel_err": quant_err, "attn_rel_err": attn_err}
