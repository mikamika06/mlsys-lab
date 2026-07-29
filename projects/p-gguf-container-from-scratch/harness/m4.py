import numpy as np

import w


def check(workdir):
    out = {"data_matches": 0.0, "dtype_kept": 0.0, "missing_raises": 0.0}
    if w.gguf_or_none() is None:
        return w.needs_gguf(out)
    from gguf_lab import read_tensor_data

    tmp = w.tmpdir()
    p = w.fixture(tmp, rich=True)
    ref = {t["name"]: t["data"] for t in w.oracle_tensors(p)}
    worst = 0.0
    dtypes_ok = True
    for name, want in ref.items():
        got = np.asarray(read_tensor_data(p, name))
        if got.size != want.size:
            out["_note"] = f"{name}: {got.size} elements, expected {want.size}"
            return out
        worst = max(worst, float(np.max(np.abs(got.ravel().astype(np.float64)
                                               - want.ravel().astype(np.float64)))))
        dtypes_ok = dtypes_ok and got.dtype == want.dtype
    out["data_matches"] = 1.0 if worst == 0.0 else 0.0
    out["max_abs_err"] = worst
    out["dtype_kept"] = 1.0 if dtypes_ok else 0.0
    try:
        read_tensor_data(p, "nope.weight")
        out["missing_raises"] = 0.0
    except Exception:
        out["missing_raises"] = 1.0
    return out
