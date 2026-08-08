import sys
import os
import ref
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    from tpp.parallel import tp_communication_volume, dtensor_mlp

    out = {"volume_match": 0.0, "mlp_match": 0.0}

    b, s, h, i, tp = 2, 16, 64, 128, 2
    want_vol = ref.reference_communication_volume(b, s, h, i, tp)
    got_vol = tp_communication_volume(b, s, h, i, tp)

    if got_vol is not None and isinstance(got_vol, dict):
        if got_vol.get("total_bytes") == want_vol["total_bytes"]:
            out["volume_match"] = 1.0

    np.random.seed(42)
    x = np.random.randn(1, 8, 32)
    w1 = np.random.randn(32, 64)
    w2 = np.random.randn(64, 32)

    want_mlp = ref.reference_dtensor_mlp(x, w1, w2, tp)
    got_mlp = dtensor_mlp(x, w1, w2, tp)

    if got_mlp is not None and np.max(np.abs(got_mlp - want_mlp)) < 1e-5:
        out["mlp_match"] = 1.0

    return out
