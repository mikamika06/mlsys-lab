import sys
sys.path.insert(0, ".")
from mlx_serve import memory, server

def test_memory_stability_invariant():
    res = memory.check_stability([10.0, 10.0, 10.0])
    assert res["memory_stable_ok"] == 1.0

def test_chat_template_invariant():
    res = server.format_chat([{"role": "user", "content": "test"}])
    assert "<|im_start|>" in res
