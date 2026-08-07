import ref


def check(workdir):
    from quant.q4_0 import quantize_q4_0, dequantize_q4_0, max_abs_err

    out = {"max_abs_err_match": 0.0}
    try:
        t = ref.TENSORS[1]
        got_bytes = quantize_q4_0(t)
        got_recon = dequantize_q4_0(got_bytes, t.shape)
        got_err = max_abs_err(t, got_recon)
        ref_bytes = ref.quantize_q4_0(t)
        ref_recon = ref.dequantize_q4_0(ref_bytes, t.shape)
        ref_err = ref.max_abs_err(t, ref_recon)
        if abs(got_err - ref_err) < 1e-5:
            out["max_abs_err_match"] = 1.0
        else:
            out["_note"] = f"got error {got_err}, reference error {ref_err}"
    except Exception as e:
        out["_note"] = f"raised {type(e).__name__}: {e}"
    return out
