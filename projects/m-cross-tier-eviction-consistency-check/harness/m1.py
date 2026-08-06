"""Milestone 1 harness check."""

import ref

def check(workdir):
    from eviction.checker import check_cross_tier_consistency

    scenarios = ref.generate_scenarios()
    out = {"scenarios_matched": 0.0}
    matched = 0

    for i, (t0_state, t1_state, expected_valid, expected_v_count) in enumerate(scenarios):
        try:
            valid, violations = check_cross_tier_consistency(t0_state, t1_state)
            if valid == expected_valid and len(violations) == expected_v_count:
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"Scenario {i} failed: expected valid={expected_valid}, got {valid}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Scenario {i} raised {type(e).__name__}: {str(e)}"

    out["scenarios_matched"] = float(matched)
    return out
