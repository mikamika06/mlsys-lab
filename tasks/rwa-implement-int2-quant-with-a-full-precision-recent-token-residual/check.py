import numpy as np

def _ref_quant(x: np.ndarray, R: int):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if R >= n:
        return np.empty((0,), dtype=np.uint8), x.astype(np.float16)
    seg = x[:n-R]
    min_val = seg.min()
    max_val = seg.max()
    if min_val == max_val:
        codes = np.zeros_like(seg, dtype=np.uint8)
    else:
        scale = 3.0 / (max_val - min_val)
        codes = np.clip(np.round((seg - min_val) * scale), 0, 3).astype(np.uint8)
    residuals = x[n-R:].astype(np.float16)
    return codes, residuals

def grade(sol, fx) -> dict:
    func = getattr(sol, "int2_quant_with_residual", None)
    if func is None or not callable(func):
        return {"codes_match": 0.0, "residual_len_ok": 0.0}
    rng = np.random.default_rng(12345)
    ok_codes = 1.0
    ok_res_len = 1.0
    for n in [5, 10, 20]:
        x = rng.uniform(-1000, 1000, size=n).astype(np.float32)
        R = rng.integers(0, n+1)
        try:
            codes, residuals = func(x, R)
        except Exception:
            return {"codes_match": 0.0, "residual_len_ok": 0.0}
        ref_codes, ref_residuals = _ref_quant(x, R)
        if not (isinstance(codes, np.ndarray) and isinstance(residuals, np.ndarray)):
            ok_codes = 0.0
            ok_res_len = 0.0
            break
        if codes.shape != ref_codes.shape or residuals.shape != ref_residuals.shape:
            ok_codes = 0.0
            ok_res_len = 0.0
            break
        if not np.array_equal(codes, ref_codes):
            ok_codes = 0.0
        if not np.array_equal(residuals.astype(np.float16), ref_residuals.astype(np.float16)):
            ok_res_len = 0.0
    return {"codes_match": ok_codes, "residual_len_ok": ok_res_len}
