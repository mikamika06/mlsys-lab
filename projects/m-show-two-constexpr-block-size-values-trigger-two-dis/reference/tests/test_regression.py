import sys

sys.path.insert(0, ".")
from triton_cache.cache_demo import (
    ConstexprWrapper,
    inspect_cache_keys,
    measure_compile_vs_hit_latency,
    trigger_constexpr_error,
)
from triton_cache.kernel import mock_triton_kernel


def test_constexpr_cache_separation():
    keys = inspect_cache_keys(mock_triton_kernel, 32, 64)
    assert len(keys) == 2, f"Expected 2 cache entries, got {len(keys)}"
    assert (32,) in keys, "(32,) not found in cache keys"
    assert (64,) in keys, "(64,) not found in cache keys"


def test_non_constexpr_error_triggered():
    raised, msg = trigger_constexpr_error(mock_triton_kernel, 32)
    assert raised, f"Expected CompilationError to be raised, message: {msg}"


def test_cache_hit_speedup():
    cold, warm = measure_compile_vs_hit_latency(mock_triton_kernel, 128)
    assert warm > 0, "Warm latency must be positive"
    assert cold / warm >= 5.0, f"Cold compile ({cold:.6f}s) should be much slower than warm hit ({warm:.6f}s)"
