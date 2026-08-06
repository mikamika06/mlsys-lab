from hpabudget.cooldown import compute_safe_cooldown_period
from hpabudget.readiness import classify_readiness


def test_readiness_and_cooldown_safety():
    """Verify readiness probes and HPA cooldown budget invariants."""
    logs = ["Process started", "Loading weights"]
    http_status = 200
    engine_state = {"ready": True, "graph_captured": False}
    state = classify_readiness(logs, http_status, engine_state)
    assert state == "process_up"

    phase_times = {
        "process_bootstrap": 10.0,
        "weight_loading": 20.0,
        "torch_compile": 60.0,
        "cudagraph_capture": 15.0,
    }
    cooldown = compute_safe_cooldown_period(
        phase_times=phase_times,
        warm_compile_cache=True,
        cache_speedup_factor=4.0,
        safety_margin_pct=20.0,
    )

    unwarmed_cooldown = compute_safe_cooldown_period(
        phase_times=phase_times,
        warm_compile_cache=False,
        cache_speedup_factor=4.0,
        safety_margin_pct=20.0,
    )

    assert cooldown >= 72
    assert cooldown < unwarmed_cooldown
