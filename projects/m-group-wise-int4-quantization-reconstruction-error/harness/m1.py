import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"quantization_matches": 0.0}
    try:
        from quant.group_quant import quantize_group_int4, dequantize_group_int4
    except Exception as e:
        out["_note"] = f"Failed to import quantize_group_int4: {type(e).__name__}: {e}"
        return out

    ok = True
    for item in ref.TEST_DATASETS:
        t = item["tensor"]
        gs = item["group_size"]
        asym = item["asymmetric"]

        ref_q, ref_s, ref_zp = ref.quantize_group_int4(t, gs, asymmetric=asym)
        ref_rec = ref.dequantize_group_int4(ref_q, ref_s, ref_zp, gs)

        try:
            got_q, got_s, got_zp = quantize_group_int4(t, gs, asymmetric=asym)
            got_rec = dequantize_group_int4(got_q, got_s, got_zp, gs)
        except Exception as e:
            out["_note"] = f"Function raised exception: {type(e).__name__}: {e}"
            return out

        if not np.array_equal(ref_q, got_q):
            ok = False
            out["_note"] = f"Quantized int4 array mismatch for group_size={gs}, asymmetric={asym}"
            break
        if not np.allclose(ref_s, got_s, rtol=1e-5, atol=1e-5):
            ok = False
            out["_note"] = f"Scale array mismatch for group_size={gs}"
            break
        if not np.allclose(ref_zp, got_zp, rtol=1e-5, atol=1e-5):
            ok = False
            out["_note"] = f"Zero point array mismatch for group_size={gs}"
            break
        if not np.allclose(ref_rec, got_rec, rtol=1e-4, atol=1e-4):
            ok = False
            out["_note"] = f"Dequantized reconstruction mismatch for group_size={gs}"
            break

    if ok:
        out["quantization_matches"] = 1.0

    return out
