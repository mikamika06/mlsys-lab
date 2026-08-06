"""Regression tests for triton cache behavior."""

from tritoncache.cache_key import build_cache_key, classify_arg


def test_cache_key_classification():
    sig = {
        "x": {"is_ptr": True},
        "n": {"is_constexpr": False},
        "BLOCK": {"is_constexpr": True},
    }

    class MockTensor:
        def __init__(self, dtype, shape, stride):
            self.dtype = dtype
            self.shape = shape
            self.stride = stride

    t1 = MockTensor("float32", (1024,), (1,))
    args1 = {"x": t1, "n": 1024, "BLOCK": 64}
    args2 = {"x": t1, "n": 2048, "BLOCK": 64}

    key1 = build_cache_key("vec_add", sig, args1)
    key2 = build_cache_key("vec_add", sig, args2)

    assert key1 == key2

    args3 = {"x": t1, "n": 1024, "BLOCK": 128}
    key3 = build_cache_key("vec_add", sig, args3)
    assert key1 != key3
