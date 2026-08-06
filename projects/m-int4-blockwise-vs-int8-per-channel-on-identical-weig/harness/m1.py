import numpy as np
import ref


def check(workdir):
    from quant.core import (
        dequantize_int4_blockwise,
        dequantize_int8_per_channel,
        quantize_int4_blockwise,
        quantize_int8_per_channel,
    )

    out = {"quantization_matches": 0.0}
    ok = 0

    for i, w in enumerate(ref.TEST_WEIGHTS):
        ref_q8, ref_s8 = ref.quantize_int8_per_channel(w)
        ref_deq8 = ref.dequantize_int8_per_channel(ref_q8, ref_s8)

        got_q8, got_s8 = quantize_int8_per_channel(w)
        got_deq8 = dequantize_int8_per_channel(got_q8, got_s8)

        ref_q4, ref_s4 = ref.quantize_int4_blockwise(w, block_size=32)
        ref_deq4 = ref.dequantize_int4_blockwise(ref_q4, ref_s4, block_size=32)

        got_q4, got_s4 = quantize_int4_blockwise(w, block_size=32)
        got_deq4 = dequantize_int4_blockwise(got_q4, got_s4, block_size=32)

        if (
            np.allclose(got_q8, ref_q8, atol=1)
            and np.allclose(got_s8, ref_s8, rtol=1e-3)
            and np.allclose(got_deq8, ref_deq8, atol=1e-2)
            and np.allclose(got_q4, ref_q4, atol=1)
            and np.allclose(got_s4, ref_s4, rtol=1e-3)
            and np.allclose(got_deq4, ref_deq4, atol=1e-2)
        ):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"test weight set {i} mismatch"

    out["quantization_matches"] = float(ok)
    return out
