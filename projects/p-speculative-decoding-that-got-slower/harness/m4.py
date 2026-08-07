def check(workdir):
    from specdec.tracker import AcceptanceTracker
    from specdec.model import SpeculativeModel
    from specdec.policy import AdaptivePolicy

    out = {
        "gate_decision_correct": 0.0,
        "throttling_prevents_loss": 0.0
    }

    tracker = AcceptanceTracker(window_size=20)
    model = SpeculativeModel(target_step_cost=10.0, draft_step_cost=1.0, overhead_per_draft=0.1)
    policy = AdaptivePolicy(model, tracker, min_speedup=1.05)

    for _ in range(10):
        tracker.record("good_domain", 4, 5)
        tracker.record("bad_domain", 0, 5)

    g_good, act_good = policy.decide("good_domain", batch_size=1)
    g_bad, act_bad = policy.decide("bad_domain", batch_size=1)

    if act_good and g_good > 0 and not act_bad and g_bad == 0:
        out["gate_decision_correct"] = 1.0

    g_batch, act_batch = policy.decide("good_domain", batch_size=64)
    if not act_batch and g_batch == 0:
        out["throttling_prevents_loss"] = 1.0

    return out
