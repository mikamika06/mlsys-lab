import sys

sys.path.insert(0, ".")
from ollama_evict.manager import ModelManager


def test_lru_eviction_strictly_removes_oldest():
    mgr = ModelManager(2)
    mgr.request("a")
    mgr.request("b")
    mgr.request("a")
    res = mgr.request("c")
    assert "b" in res["evicted"]
    assert "a" not in res["evicted"]


def test_capacity_never_exceeded():
    mgr = ModelManager(2)
    for m in ["m1", "m2", "m3", "m4", "m5"]:
        res = mgr.request(m)
        assert len(res["loaded"]) <= 2
