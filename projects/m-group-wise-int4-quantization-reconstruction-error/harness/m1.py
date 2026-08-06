import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant.group_int4 import compute_reconstruction_mse, quantize_dequantize_int4

    test_cases = ref.generate_test_cases()
    out = {"reconstruction_mse_matched": 0.0, "total_cases": float(len(test_cases))}
    matched = 0

    for i, case in enumerate(test_cases):
        x = case["x"]
        g = case["group_size"]
        want_res = ref.compute_reconstruction_mse(x, g)
        want_q, want_dq, want_s = ref.quantize_dequantize_int4(x, g)

        try:
            got_q, got_dq, got_s = quantize_dequantize_int4(x, g)
            got_res = compute_reconstruction_mse(x, g)

            q_ok = np.array_equal(got_q, want_q)
            dq_ok = np.allclose(got_dq, want_dq, atol=1e-5)
            s_ok = np.allclose(got_s, want_s, atol=1e-5)
            total_mse_ok = np.isclose(got_res["total_mse"], want_res["total_mse"], atol=1e-6)
            group_mse_ok = np.allclose(got_res["group_mse"], want_res["group_mse"], atol=1e-6)

            if q_ok and dq_ok and s_ok and total_mse_ok and group_mse_ok:
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"case {i} mismatch: q_ok={q_ok}, dq_ok={dq_ok}, mse_ok={total_mse_ok}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {i} raised exception: {type(e).__name__}: {str(e)[:100]}"

    out["reconstruction_mse_matched"] = float(matched)
    return out
