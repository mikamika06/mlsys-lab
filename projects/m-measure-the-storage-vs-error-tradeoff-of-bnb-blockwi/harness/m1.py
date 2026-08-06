import ref
import numpy as np


def check(workdir):
    from bnbquant.quantize import blockwise_quantize, blockwise_dequantize

    out = {"quant_match": 0.0}
    ok = 0
    for t in ref.TENSORS:
        for bs in ref.BLOCK_SIZES:
            for b in ref.BITS_LIST:
                ref_q, ref_s, ref_orig_len = ref.blockwise_quantize(t, bs, b)
                ref_dq = ref.blockwise_dequantize(ref_q, ref_s, bs, ref_orig_len, b)

                try:
                    got_q, got_s, got_orig_len = blockwise_quantize(t, bs, b)
                    got_dq = blockwise_dequantize(got_q, got_s, bs, got_orig_len, b)
                except Exception:
                    continue

                if np.allclose(ref_dq, got_dq, atol=1e-5):
                    ok += 1
    total = len(ref.TENSORS) * len(ref.BLOCK_SIZES) * len(ref.BITS_LIST)
    if ok == total:
        out["quant_match"] = 1.0
    return out
