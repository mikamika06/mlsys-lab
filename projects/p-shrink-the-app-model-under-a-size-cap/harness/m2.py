import numpy as np
import ref


def check(workdir):
    import compress.api as api

    m = {"type_ok": 0.0, "indices_ok": 0.0, "palette_ok": 0.0, "loss_measured": 0.0}
    arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    try:
        out = api.palettize(arr, k=3)
        if out.get("type") == "palette":
            m["type_ok"] = 1.0
        if out.get("indices") is not None and out["indices"].dtype == np.uint8 and out["indices"].tolist() == [0, 1, 2]:
            m["indices_ok"] = 1.0
        if out.get("palette") is not None and out["palette"].dtype == np.float32 and len(out["palette"]) == 3:
            m["palette_ok"] = 1.0

        rng = np.random.RandomState(1)
        test_arr = rng.randn(1000).astype(np.float32)
        out2 = api.palettize(test_arr, k=16)
        dec = ref.decompress(out2)
        mse = float(np.mean((test_arr - dec)**2))
        if mse < 0.05:
            m["loss_measured"] = 1.0
    except Exception:
        pass

    return m
