import ref

def check(workdir):
    from vllm_cost.model import compute_spot_expected_cost
    scenarios = ref.generate_scenarios()
    ok = 0
    for sc in scenarios:
        base_cost, p_preempt, restart_h, loss_frac, req_h = sc["spot"]
        got = compute_spot_expected_cost(base_cost, p_preempt, restart_h, loss_frac, req_h)
        want = ref.compute_spot_expected_cost(base_cost, p_preempt, restart_h, loss_frac, req_h)
        if abs(got - want) / max(1.0, abs(want)) < 1e-5:
            ok += 1
    return {"spot_cost_matched": 1.0 if ok == len(scenarios) else 0.0}
