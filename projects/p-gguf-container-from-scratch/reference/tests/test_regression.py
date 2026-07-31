import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, ".")
import gguf_lab.writer as writer
from gguf_lab import read_header, read_tensor_data, read_tensor_info


def _write_sample(path):
    tensors = [{"name": "x.weight", "data": np.arange(12, dtype=np.float32).reshape(3, 4)},
               {"name": "y.bias", "data": (np.arange(5) * 1.5).astype(np.float32)}]
    writer.write(path, "labarch", {"lab.count": 3, "lab.name": "check"},
                 tensors, alignment=32)
    return tensors


def _validate(path, expected):
    h = read_header(path)
    assert h["magic"] == "GGUF"
    infos = read_tensor_info(path)
    assert h["tensor_count"] == len(infos), (
        f"header claims {h['tensor_count']} tensors, the table holds {len(infos)}")
    for t, want in zip(infos, expected):
        assert t["offset"] % 32 == 0, f"{t['name']} starts at {t['offset']}, not on a 32-byte boundary"
        data = np.asarray(read_tensor_data(path, t["name"]))
        assert data.size == want["data"].size, (
            f"{t['name']}: {data.size} elements readable, {want['data'].size} declared")
        assert np.array_equal(data.ravel(), want["data"].ravel())


def test_written_container_is_self_consistent():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.gguf")
        expected = _write_sample(p)
        _validate(p, expected)


def test_declared_tensor_count_matches_the_table():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.gguf")
        _write_sample(p)
        h = read_header(p)
        assert h["tensor_count"] == len(read_tensor_info(p))
