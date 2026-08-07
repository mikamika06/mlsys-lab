import os
from gguf_writer.writer import GGUFWriter
from gguf_writer.validator import validate_gguf

def check(workdir):
    m = {"runtime_exec_ok": 0.0}
    path = os.path.join(workdir, "test_m5.gguf")
    try:
        w = GGUFWriter(path)
        w.add_metadata("general.architecture", "llama")
        w.write()
        if validate_gguf(path):
            m["runtime_exec_ok"] = 1.0
    except Exception:
        pass
    finally:
        if os.path.exists(path):
            os.remove(path)
    return m
