import numpy as np
import ref


def check(workdir):
    from bf16num.fp16 import fp16_subnormal_mask, round_fp32_to_fp16
    from bf16num.ulp import compute_ulp, get_dtype_max

    data = ref.make_m2_dataset()

    out = {
        "subnormals_matched": 0.0,
        "ulp_matched": 0.0,
        "max_val_matched": 0.0,
    }

    try:
        want_sub = ref.fp16_subnormal_mask(data)
        got_sub = fp16_subnormal_mask(data)

        want_flushed = ref.round_fp32_to_fp16(data, flush_subnormals=True)
        got_flushed = round_fp32_to_fp16(data, flush_subnormals=True)

        sub_ok = np.array_equal(want_sub, got_sub) and np.array_equal(
            (np.isnan(want_flushed) & np.isnan(got_flushed)) | (want_flushed == got_flushed),
            np.ones(len(data), dtype=bool),
        )
        out["subnormals_matched"] = 1.0 if sub_ok else 0.0

        ulp_ok = True
        for dt in ["fp32", "fp16", "bf16"]:
            want_u = ref.compute_ulp(data, dt)
            got_u = compute_ulp(data, dt)
            if not np.allclose(want_u, got_u, rtol=1e-5, atol=1e-30):
                ulp_ok = False
                break
        out["ulp_matched"] = 1.0 if ulp_ok else 0.0

        max_ok = True
        for dt in ["fp32", "fp16", "bf16"]:
            want_m = ref.get_dtype_max(dt)
            got_m = get_dtype_max(dt)
            if not np.isclose(want_m, got_m, rtol=1e-6):
                max_ok = False
                break
        out["max_val_matched"] = 1.0 if max_ok else 0.0

    except Exception as e:
        out["_note"] = f"Execution error: {e}"

    return out
