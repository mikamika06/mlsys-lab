import os
import tempfile
from cachefootprint.measure import measure_footprint
from cachefootprint.invalidation import check_path_invalidation
from cachefootprint.compare import compare_bytes


def test_measure_footprint_scaling():
    with tempfile.TemporaryDirectory() as tmp:
        sizes = [1, 2, 4]
        res = measure_footprint(sizes, tmp)
        for s in sizes:
            assert s in res
            assert res[s] > 0


def test_invalidation_behavior():
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        res = check_path_invalidation(tmp1, tmp2)
        assert res["path_keyed"] is True
        assert res["invalidated"] is True


def test_compare_bytes_ratio():
    with tempfile.TemporaryDirectory() as tmp:
        dummy = os.path.join(tmp, "kernel.so")
        with open(dummy, "wb") as f:
            f.write(b"ABC")
        res = compare_bytes(tmp)
        assert res["raw_bytes"] >= res["artifact_bytes"]
        assert res["artifact_bytes"] > 0
