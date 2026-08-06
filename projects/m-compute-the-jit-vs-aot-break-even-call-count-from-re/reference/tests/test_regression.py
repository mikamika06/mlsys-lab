from aotbreak.breakeven import compute_breakeven


def test_breakeven_threshold_invariant():
    """Verify that preferred mode latency is <= alternative at break_even_calls."""
    profile = {
        "jit_compile_ms": 100.0,
        "jit_exec_ms": 1.0,
        "aot_load_ms": 0.0,
        "aot_exec_ms": 10.0,
    }
    res = compute_breakeven(profile)
    n = int(res["break_even_calls"])

    jit_time = profile["jit_compile_ms"] + n * profile["jit_exec_ms"]
    aot_time = profile["aot_load_ms"] + n * profile["aot_exec_ms"]

    assert res["preferred_mode"] == "jit"
    assert jit_time <= aot_time
    if n > 1:
        prev_n = n - 1
        prev_jit = profile["jit_compile_ms"] + prev_n * profile["jit_exec_ms"]
        prev_aot = profile["aot_load_ms"] + prev_n * profile["aot_exec_ms"]
        assert prev_jit > prev_aot
