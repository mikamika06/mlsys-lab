import ref
import numpy as np


def check(workdir):
    from eagle.head import DraftHead
    m = {"shape_ok": 0.0, "forward_ok": 0.0}
    try:
        head = DraftHead(8, 32)
        hs = np.ones((1, 8), dtype=np.float32)
        out = head.forward(hs)
        if out.shape[-1] == 32:
            m["shape_ok"] = 1.0
            m["forward_ok"] = 1.0
    except Exception:
        pass
    return m
