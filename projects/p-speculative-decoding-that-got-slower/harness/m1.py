import numpy as np


def check(workdir):
    from specdec.tracker import AcceptanceTracker

    out = {
        "tracker_api_ok": 0.0,
        "acceptance_accuracy": 0.0,
        "per_domain_tracking": 0.0
    }

    tracker = AcceptanceTracker(window_size=10)

    try:
        r0 = tracker.get_acceptance_rate("unknown")
        if isinstance(r0, (int, float)):
            out["tracker_api_ok"] = 1.0
    except Exception:
        return out

    tracker.record("domain_a", 3, 5)
    tracker.record("domain_a", 4, 5)

    rate_a = tracker.get_acceptance_rate("domain_a")
    expected_a = (0.6 + 0.8) / 2.0
    if abs(rate_a - expected_a) < 1e-6:
        out["acceptance_accuracy"] = 1.0

    tracker.record("domain_b", 1, 5)
    rate_b = tracker.get_acceptance_rate("domain_b")
    global_rate = tracker.get_acceptance_rate(None)

    expected_global = (0.6 + 0.8 + 0.2) / 3.0
    if abs(rate_b - 0.2) < 1e-6 and abs(global_rate - expected_global) < 1e-6:
        out["per_domain_tracking"] = 1.0

    return out
