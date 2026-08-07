import os
from gguf_writer.writer import GGUFWriter
from gguf_writer.validator import validate_gguf

def check(workdir):
    m = {"external_parse_ok": 0.0}
    path = os.path.join(workdir, "test_m4.gguf")
    try:
        w = GGUFWriter(path)
        w.add_metadata("general.name", "test")
        w.write()
        if validate_gguf(path):
            m["external_parse_ok"] = 1.0
    except Exception:
        pass
    finally:
        if os.path.exists(path):
            os.remove(path)
    return m
