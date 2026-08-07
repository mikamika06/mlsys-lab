import numpy as np
import ref


def check(workdir):
    import compress.api as api

    m = {"type_ok": 0.0, "size_reduced": 0.0, "loss_measured": 0.0}
    arr = np.array([-0.5, 0.01, 0.05, 0.8], dtype=np.float32)

    try:
        out = api.sparsify(arr, threshold=0.1)
        if out.get("type") == "sparse":
            m["type_ok"] = 1.0
        if out.get("indices") is not None and len(out["indices"]) == 2 and len(out["data"]) == 2:
            m["size_reduced"] = 1.0

        dec = ref.decompress(out)
        mse = float(np.mean((arr - dec)**2))
        expected_mse = (0.01**2 + 0.05**2) / 4.0
        if np.isclose(mse, expected_mse, atol=1e-5):
            m["loss_measured"] = 1.0
    except Exception:
        pass

    return m
