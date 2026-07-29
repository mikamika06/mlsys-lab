import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))


def gguf_or_none():
    try:
        import gguf
        return gguf
    except ImportError:
        return None


def needs_gguf(out):
    out["_note"] = "this milestone needs the gguf package: pip install gguf"
    return out


def fixture(tmp, name="fx.gguf", alignment=None, rich=False):
    """A container written by the real library, so the learner's reader is checked
    against the format as it actually is and not against our idea of it."""
    import gguf
    path = os.path.join(tmp, name)
    w = gguf.GGUFWriter(path, "labarch")
    w.add_uint32("lab.count", 7)
    w.add_string("lab.name", "привіт")
    w.add_array("lab.list", [3, 1, 4, 1, 5])
    if rich:
        w.add_float32("lab.ratio", 0.25)
        w.add_bool("lab.flag", True)
        w.add_array("lab.words", ["alpha", "beta"])
    if alignment:
        w.add_custom_alignment(alignment)
    w.add_tensor("a.weight", np.arange(12, dtype=np.float32).reshape(3, 4))
    w.add_tensor("b.bias", np.arange(5, dtype=np.float32) * 2)
    if rich:
        w.add_tensor("c.half", (np.arange(8) * 0.5).astype(np.float16).reshape(2, 4))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def oracle_meta(path):
    import gguf
    r = gguf.GGUFReader(path)
    out = {}
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        out[k] = f.contents()
    return out


def oracle_tensors(path):
    import gguf
    r = gguf.GGUFReader(path)
    return [{"name": t.name, "shape": list(t.shape), "data": np.array(t.data)} for t in r.tensors]


def tmpdir():
    return tempfile.mkdtemp(prefix="gguf-lab-")
