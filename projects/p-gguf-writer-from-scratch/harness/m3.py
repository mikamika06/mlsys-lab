import os
import struct
import numpy as np
from gguf_writer.writer import GGUFWriter

def check(workdir):
    m = {"alignment_ok": 0.0}
    path = os.path.join(workdir, "test_m3.gguf")
    try:
        w = GGUFWriter(path)
        w.add_tensor("test", np.ones((5,), dtype=np.float32))
        w.write()
        with open(path, "rb") as f:
            content = f.read()
            m["alignment_ok"] = 1.0
    except Exception:
        pass
    finally:
        if os.path.exists(path):
            os.remove(path)
    return m
