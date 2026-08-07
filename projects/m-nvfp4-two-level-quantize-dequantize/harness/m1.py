import numpy as np
import ref


def check(workdir):
    try:
        from quant.formats import round_e2m1, mxfp4
    except ImportError:
        return {"_note": "ImportError on quant.formats"}

    out = {"e2m1_match": 0.0, "mxfp4_match": 0.0}

    np.random.seed(42)
    test_vals = np.linspace(-7, 7, 100)
    if np.allclose(ref.round_e2m1(test_vals), round_e2m1(test_vals)):
        out["e2m1_match"] = 1.0

    x = np.random.randn(1024) * 10
    if np.allclose(ref.mxfp4(x), mxfp4(x)):
        out["mxfp4_match"] = 1.0

    return out
