import numpy as np
import ref


def check(workdir):
    from fa_tradeoff.attention import blockwise_attention
    q, k, v, mask = ref.generate_inputs()
    mask_full = np.zeros_like(mask)
    try:
        out_nan = blockwise_attention(q, k, v, mask_full)
        has_nan = 1.0 if not np.any(np.isnan(out_nan)) else 0.0
    except Exception:
        has_nan = 0.0

    out_got = blockwise_attention(q, k, v, mask)
    out_ref = ref.reference_blockwise_attention(q, k, v, mask)
    match = 1.0 if np.allclose(out_got, out_ref, atol=1e-5) else 0.0

    return {"no_nan": has_nan, "blockwise_match": match}
