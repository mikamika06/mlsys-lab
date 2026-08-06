import recompile.guard as g_ref
import recompile.cache as c_ref
import recompile.policy as p_ref

GUARD_TESTS = [
    {"history": [[1, 16]], "shape": [1, 32], "enabled": True},
    {"history": [[1, 32]], "shape": [1, 32], "enabled": True},
    {"history": [], "shape": [1, 16], "enabled": False},
]

CACHE_TESTS = [
    {"cache_files": ["a.bin", "b.bin"], "available": ["a.bin", "b.bin", "c.bin"]},
    {"cache_files": ["a.bin", "b.bin"], "available": ["a.bin"]},
    {"cache_files": [], "available": []},
]

POLICY_TESTS = [
    {"table": {("strict", False): "fail_fast"}, "state": ("strict", False)},
    {"table": {("lax", True): "fallback"}, "state": ("lax", False)},
]
