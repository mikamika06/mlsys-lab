import numpy as np
import ml_dtypes


def grade(sol, fx) -> dict:
    x = np.array(
        [
            -1000.0, -449.0, -448.0, -447.5, -20.0, -0.02,
            -0.001, -0.0, 0.0, 0.001, 0.02, 1.0,
            3.5, 100.0, 448.0, 449.0, 1000.0, np.nan
        ],
        dtype=np.float32,
    )

    oracle_codes = x.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)

    try:
        got_codes = np.asarray(sol.encode_fp8_e4m3fn(x), dtype=np.uint8)
    except Exception:
        return {"byte_exact_fraction": 0.0, "max_abs_err": float("inf")}

    byte_exact_fraction = float(np.mean(got_codes == oracle_codes))

    ref_decoded = x.astype(ml_dtypes.float8_e4m3fn).astype(np.float32)
    try:
        got_decoded = np.asarray(sol.decode_fp8_e4m3fn(oracle_codes), dtype=np.float32)
        finite = np.isfinite(ref_decoded) & np.isfinite(got_decoded)
        if np.any(finite):
            max_abs_err = float(np.max(np.abs(ref_decoded[finite] - got_decoded[finite])))
        else:
            max_abs_err = 0.0
    except Exception:
        max_abs_err = float("inf")

    return {
        "byte_exact_fraction": byte_exact_fraction,
        "max_abs_err": max_abs_err,
    }
