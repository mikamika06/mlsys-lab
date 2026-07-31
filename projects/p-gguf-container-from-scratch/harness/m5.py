import os

import numpy as np

import w


def check(workdir):
    out = {"real_reader_accepts": 0.0, "metadata_survives": 0.0, "tensors_survive": 0.0}
    gguf = w.gguf_or_none()
    if gguf is None:
        return w.needs_gguf(out)
    from gguf_lab import write

    tmp = w.tmpdir()
    p = os.path.join(tmp, "mine.gguf")
    meta = {"my.count": 42, "my.name": "test", "my.list": [3, 1, 4]}
    tensors = [{"name": "x.weight", "data": np.arange(6, dtype=np.float32).reshape(2, 3)},
               {"name": "y.bias", "data": (np.arange(4) * 1.5).astype(np.float32)}]
    write(p, "labarch", meta, tensors, alignment=32)

    try:
        r = gguf.GGUFReader(p)
    except Exception as e:  # noqa: BLE001
        out["_note"] = f"the real reader refused the file: {type(e).__name__}: {str(e)[:120]}"
        return out
    out["real_reader_accepts"] = 1.0

    fields = {k: f.contents() for k, f in r.fields.items() if not k.startswith("GGUF.")}
    ok = (fields.get("general.architecture") == "labarch"
          and fields.get("my.count") == 42
          and fields.get("my.name") == "test"
          and list(fields.get("my.list") or []) == [3, 1, 4])
    out["metadata_survives"] = 1.0 if ok else 0.0
    if not ok:
        out["_note"] = f"metadata came back as {fields}"

    got = {t.name: np.array(t.data) for t in r.tensors}
    same = (set(got) == {"x.weight", "y.bias"}
            and np.array_equal(got["x.weight"].ravel(), tensors[0]["data"].ravel())
            and np.array_equal(got["y.bias"].ravel(), tensors[1]["data"].ravel()))
    out["tensors_survive"] = 1.0 if same else 0.0
    return out
