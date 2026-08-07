import sys

sys.path.insert(0, ".")
from mlspec.blobs import detect_duplicate_blobs
from mlspec.diff import diff_package_and_compiled
from mlspec.io import rename_io


def test_rename_io_preserves_nodes():
    spec = {
        "inputs": [{"name": "a"}],
        "outputs": [{"name": "b"}],
        "nodes": [{"inputs": ["a"], "outputs": ["b"]}]
    }
    res = rename_io(spec, {"a": "x"}, {"b": "y"})
    assert res["inputs"][0]["name"] == "x"
    assert res["outputs"][0]["name"] == "y"
    assert res["nodes"][0]["inputs"] == ["x"]
    assert res["nodes"][0]["outputs"] == ["y"]


def test_detect_duplicate_blobs():
    blobs = {"f1": b"data", "f2": b"other", "f3": b"data"}
    dups = detect_duplicate_blobs(blobs)
    assert len(dups) == 2


def test_diff_package():
    d = diff_package_and_compiled({"a": 1}, {"a": 1, "b": 2})
    assert "b" in d["only_in_compiled"]
