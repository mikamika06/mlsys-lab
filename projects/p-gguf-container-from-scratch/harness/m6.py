import os

import numpy as np

import w


def check(workdir):
    out = {"roundtrip_metadata": 0.0, "roundtrip_tensors": 0.0, "alignment_honoured": 0.0}
    gguf = w.gguf_or_none()
    if gguf is None:
        return w.needs_gguf(out)
    from gguf_lab import read_metadata, read_tensor_data, read_tensor_info, write

    tmp = w.tmpdir()
    src = w.fixture(tmp, name="src.gguf", alignment=64, rich=True)
    meta = {k: v for k, v in read_metadata(src).items()
            if k not in ("general.architecture", "general.alignment")}
    tensors = [{"name": t["name"], "data": np.asarray(read_tensor_data(src, t["name"]))}
               for t in read_tensor_info(src)]

    dst = os.path.join(tmp, "copy.gguf")
    write(dst, "labarch", meta, tensors, alignment=64)

    a, b = w.oracle_meta(src), w.oracle_meta(dst)
    keys = set(a) | set(b)
    diff = []
    for k in sorted(keys):
        x, y = a.get(k), b.get(k)
        if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
            same = list(x or []) == list(y or [])
        elif isinstance(x, float) or isinstance(y, float):
            same = x is not None and y is not None and abs(float(x) - float(y)) < 1e-6
        else:
            same = x == y
        if not same:
            diff.append(f"{k}: {x!r} -> {y!r}")
    out["roundtrip_metadata"] = 1.0 if not diff else 0.0
    if diff:
        out["_note"] = "; ".join(diff[:3])

    ta = {t["name"]: t["data"] for t in w.oracle_tensors(src)}
    tb = {t["name"]: t["data"] for t in w.oracle_tensors(dst)}
    same = set(ta) == set(tb) and all(
        np.array_equal(ta[k].ravel().astype("float64"), tb[k].ravel().astype("float64"))
        for k in ta)
    out["roundtrip_tensors"] = 1.0 if same else 0.0

    offs = [t["offset"] for t in read_tensor_info(dst)]
    out["alignment_honoured"] = 1.0 if all(o % 64 == 0 for o in offs) else 0.0
    return out
