import sys
import tempfile
import os

sys.path.insert(0, ".")
from megacache.artifacts import save_artifact, load_artifact
from megacache.keys import find_cache_break


def test_artifact_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "art.pkl")
        art = {"graph": [1, 2, 3], "meta": "test"}
        save_artifact(p, art)
        loaded = load_artifact(p)
        assert loaded == art


def test_cache_break_detection():
    base = {"a": 1, "b": 2}
    new = {"a": 1, "b": 3}
    assert find_cache_break(base, new) == "b"
