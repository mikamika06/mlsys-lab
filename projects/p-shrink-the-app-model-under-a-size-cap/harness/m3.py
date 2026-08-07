import numpy as np
import ref


def check(workdir):
    import compress.api as api

    m = {"type_ok": 0.0, "scale_zp_ok": 0.0, "loss_measured": 0.0}
    rng = np.random.RandomState(2)
    arr = rng.uniform(-2.0, 4.0, size=1000).astype(np.float32)

    try:
        out = api.quantize(arr)
        if out.get("type") == "uint8":
            m["type_ok"] = 1.0
        if isinstance(out.get("scale"), float) and isinstance(out.get("zp"), int):
            m["scale_zp_ok"] = 1.0

        dec = ref.decompress(out)
        mse = float(np.mean((arr - dec)**2))
        if mse < 0.001:
            m["loss_measured"] = 1.0
    except Exception:
        pass

    return m
