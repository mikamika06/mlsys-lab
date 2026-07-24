import numpy as np

def _oracle(tensor):
    """Independent NumPy oracle for asymmetric uint8 calibration parameters."""
    t = np.asarray(tensor, dtype=np.float64)
    mn = float(t.min())
    mx = float(t.max())
    scale = (mx - mn) / 255.0
    if scale == 0.0:
        return 0.0, 0
    zp = int(np.clip(np.round(-mn / scale), 0, 255))
    return scale, zp

def grade(sol, fx) -> dict:
    cases = [
        np.array([-1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 2.0, 255.0]),
        np.array([0.0, 0.0, 0.0]),             # constant tensor
        np.array([-128.0, 127.0]),
        np.array([3.14, 2.71, 1.41, -2.0, 5.0]),
        np.array([0.5]),                         # single element (constant)
        np.array([-0.5, 0.5]),
        np.array([-100.0, -50.0, 0.0]),         # all non-positive → zp clamped high
        np.array([1000.0, 2000.0, 3000.0]),     # all positive → zp clamped low
        np.arange(0, 1000, dtype=np.float32),   # wider range
        np.random.RandomState(42).randn(200),   # random normal
    ]

    max_scale_err = 0.0
    zp_pass = True

    for tensor in cases:
        try:
            s_got, z_got = sol.calibration_params(tensor)
            s_got = float(s_got)
            z_got = int(z_got)
        except Exception:
            return {"scale_ok": 0.0, "zp_ok": 0.0}

        s_ref, z_ref = _oracle(tensor)

        # scale: relative error
        if s_ref == 0.0:
            if s_got != 0.0:
                max_scale_err = float("inf")
        else:
            err = abs(s_got - s_ref) / abs(s_ref)
            if err > max_scale_err:
                max_scale_err = err

        # zero_point: exact integer match
        if z_got != z_ref:
            zp_pass = False

    scale_ok = 1.0 if max_scale_err <= 1e-8 else 0.0
    zp_ok = 1.0 if zp_pass else 0.0

    return {"scale_ok": scale_ok, "zp_ok": zp_ok}
