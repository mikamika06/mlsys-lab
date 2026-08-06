import numpy as np
import ref


def check(workdir):
    from fp8kv.quant import quantize_e4m3_per_tensor, dequantize_e4m3_per_tensor
    from fp8kv.compare import compare_formats_on_kv_dump, compute_mse

    out = {"scale_exact_fraction": 0.0, "roundtrip_mse_match": 0.0}

    kv_dumps = ref.generate_kv_dumps()

    import sys
    import os
    sys.path.insert(0, os.path.join(workdir, "reference"))
    import fp8kv.quant as ref_quant
    import fp8kv.compare as ref_compare

    matched_scales = 0
    matched_mses = 0
    total = len(kv_dumps)

    for idx, dump in enumerate(kv_dumps):
        ref_q, ref_scale = ref_quant.quantize_e4m3_per_tensor(dump)
        try:
            got_q, got_scale = quantize_e4m3_per_tensor(dump)
        except Exception as e:
            out["_note"] = f"quantize_e4m3_per_tensor raised on dump {idx}: {type(e).__name__}: {e}"
            return out

        if np.isclose(ref_scale, got_scale, rtol=1e-5):
            matched_scales += 1

        try:
            got_deq = dequantize_e4m3_per_tensor(got_q, got_scale)
            got_mse = compute_mse(dump, got_deq)
            ref_deq = ref_quant.dequantize_e4m3_per_tensor(ref_q, ref_scale)
            ref_mse = ref_compare.compute_mse(dump, ref_deq)
            if np.isclose(got_mse, ref_mse, rtol=1e-4):
                matched_mses += 1
        except Exception as e:
            out["_note"] = f"dequantize or compare raised on dump {idx}: {type(e).__name__}: {e}"
            return out

    out["scale_exact_fraction"] = float(matched_scales) / float(total)
    out["roundtrip_mse_match"] = float(matched_mses) / float(total)
    return out
