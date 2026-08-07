import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {
        "quant_matched": 0.0,
        "dequant_matched": 0.0,
        "error_bounded": 0.0,
    }

    try:
        from kvquant.quant import quantize_q8_0, dequantize_q8_0, max_abs_error_bound
    except Exception as e:
        out["_note"] = f"Failed to import quant module: {e}"
        return out

    x = ref.generate_quant_test_data()

    try:
        qdict = quantize_q8_0(x, block_size=32)
        if "qdata" in qdict and "scales" in qdict:
            if qdict["qdata"].dtype == np.int8 and qdict["qdata"].shape == (8 * 128 // 32, 32):
                out["quant_matched"] = 1.0
            else:
                out["_note"] = f"qdata mismatch shape/type: {qdict['qdata'].shape}, {qdict['qdata'].dtype}"
        else:
            out["_note"] = "qdict missing required keys"
    except Exception as e:
        out["_note"] = f"quantize_q8_0 failed: {e}"
        return out

    try:
        rec = dequantize_q8_0(qdict)
        if rec.shape == x.shape and rec.dtype == np.float32:
            max_diff = np.max(np.abs(x - rec))
            if max_diff < 0.1:
                out["dequant_matched"] = 1.0
            else:
                out["_note"] = f"High dequant error: {max_diff}"
        else:
            out["_note"] = f"dequant shape/type mismatch: {rec.shape}, {rec.dtype}"
    except Exception as e:
        out["_note"] = f"dequantize_q8_0 failed: {e}"
        return out

    try:
        bound = max_abs_error_bound(x, block_size=32)
        max_err = np.max(np.abs(x - rec))
        if max_err <= bound + 1e-6:
            out["error_bounded"] = 1.0
        else:
            out["_note"] = f"Max error {max_err} exceeds bound {bound}"
    except Exception as e:
        out["_note"] = f"max_abs_error_bound failed: {e}"

    return out
