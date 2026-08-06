import ref

def check(workdir):
    from vllm_cost.model import compute_breakeven_volume
    scenarios = ref.generate_scenarios()
    ok = 0
    for sc in scenarios:
        fixed, api, self_var = sc["breakeven"]
        got = compute_breakeven_volume(fixed, api, self_var)
        want = ref.compute_breakeven_volume(fixed, api, self_var)
        if (got == float('inf') and want == float('inf')) or (isinstance(got, float) and abs(got - want) / max(1.0, abs(want)) < 1e-7):
            ok += 1
    return {"breakeven_matched": 1.0 if ok == len(scenarios) else 0.0}
