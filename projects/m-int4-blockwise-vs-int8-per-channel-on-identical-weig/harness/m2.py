import numpy as np
import ref


def check(workdir):
    from quant.core import (
        dequantize_int4_blockwise,
        dequantize_int8_per_channel,
        quantize_int4_blockwise,
        quantize_int8_per_channel,
    )
    from quant.metrics import compute_bit_size, compute_mse

    out = {"mse_matches": 0.0, "compression_ratio_matches": 0.0}

    mse_ok = True
    ratio_ok = True

    for w in ref.TEST_WEIGHTS:
        ref_q8, ref_s8 = ref.quantize_int8_per_channel(w)
        ref_deq8 = ref.dequantize_int8_per_channel(ref_q8, ref_s8)
        ref_mse8 = ref.compute_mse(w, ref_deq8)

        got_q8, got_s8 = quantize_int8_per_channel(w)
        got_deq8 = dequantize_int8_per_channel(got_q8, got_s8)
        got_mse8 = compute_mse(w, got_deq8)

        ref_q4, ref_s4 = ref.quantize_int4_blockwise(w, block_size=32)
        ref_deq4 = ref.dequantize_int4_blockwise(ref_q4, ref_s4, block_size=32)
        ref_mse4 = ref.compute_mse(w, ref_deq4)

        got_q4, got_s4 = quantize_int4_blockwise(w, block_size=32)
        got_deq4 = dequantize_int4_blockwise(got_q4, got_s4, block_size=32)
        got_mse4 = compute_mse(w, got_deq4)

        if not np.isclose(got_mse8, ref_mse8, rtol=1e-3) or not np.isclose(got_mse4, ref_mse4, rtol=1e-3):
            mse_ok = False

        b8_ref = ref.compute_bit_size(w.shape, "int8_per_channel")
        b4_ref = ref.compute_bit_size(w.shape, "int4_blockwise", block_size=32)

        b8_got = compute_bit_size(w.shape, "int8_per_channel")
        b4_got = compute_bit_size(w.shape, "int4_blockwise", block_size=32)

        if b8_ref != b8_got or b4_ref != b4_got:
            ratio_ok = False

    out["mse_matches"] = 1.0 if mse_ok else 0.0
    out["compression_ratio_matches"] = 1.0 if ratio_ok else 0.0
    return out
