import sys


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"distinct_cache_entries": 0.0, "compilation_error_caught": 0.0}

    try:
        from triton_cache.cache_demo import (
            inspect_cache_keys,
            trigger_constexpr_error,
        )
        from triton_cache.kernel import mock_triton_kernel

        keys = inspect_cache_keys(mock_triton_kernel, 16, 64)
        if len(keys) == 2 and (16,) in keys and (64,) in keys:
            out["distinct_cache_entries"] = 1.0
        else:
            out["_note"] = f"Expected 2 cache keys ((16,), (64,)), got: {keys}"

        raised, msg = trigger_constexpr_error(mock_triton_kernel, 32)
        if raised:
            out["compilation_error_caught"] = 1.0
        else:
            out["_note"] = f"Failed to trigger CompilationError: {msg}"

    except Exception as e:
        out["_note"] = f"Milestone 1 check failed with error: {type(e).__name__}: {e}"

    return out
