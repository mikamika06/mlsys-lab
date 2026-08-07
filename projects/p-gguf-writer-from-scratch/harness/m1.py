import os
from gguf_writer.writer import GGUFWriter

def check(workdir):
    m = {"header_ok": 0.0, "metadata_count": 0.0}
    path = os.path.join(workdir, "test_m1.gguf")
    try:
        w = GGUFWriter(path)
        w.add_metadata("test.key", "value")
        w.write()
        if os.path.exists(path):
            with open(path, "rb") as f:
                magic = f.read(4)
                if magic == b"GGUF":
                    m["header_ok"] = 1.0
                ver = f.read(4)
                if len(ver) == 4:
                    m["metadata_count"] = 1.0
    except Exception:
        pass
    finally:
        if os.path.exists(path):
            os.remove(path)
    return m
