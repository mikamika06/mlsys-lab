import numpy as np
import ref
from fp8util.scale import compute_scale, quantize_per_tensor
from fp8util.quant import compare_formats


def check(workdir):
    out = {"scale_match": 0.0, "comparison_match": 0.0}
    data = ref.generate_test_data(seed=456)

    try:
        scale = compute_scale(data)
        res = quantize_per_tensor(data)
        comp = compare_formats(data)

        from reference.fp8util.scale import compute_scale as ref_compute_scale, quantize_per_tensor as ref_quantize
        from reference.fp8util.quant import compare_formats as ref_compare

        ref_scale = ref_compute_scale(data)
        ref_comp = ref_compare(data)

        if np.isclose(scale, ref_scale, rtol=1e-5):
            out["scale_match"] = 1.0
        else:
            out["_note"] = f"Scale mismatch: got {scale}, expected {ref_scale}"

        if isinstance(comp, dict) and "mse_e4m3" in comp and "long_tail_handled" in comp:
            out["comparison_match"] = 1.0
        else:
            out["_note"] = "Comparison format output format is invalid."
    except Exception as e:
        out["_note"] = f"Error during scale check: {type(e).__name__}: {str(e)}"

    return out
