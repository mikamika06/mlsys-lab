import ref


def check(workdir):
    from ctxcost.rope import simulate_rope_schedule
    from ctxcost.evals import compare_strategies

    out = {"schedules_matched": 0.0}
    ok = 0
    for i, sc in enumerate(ref.ROPE_SCENARIOS):
        want = ref.simulate_rope_schedule(sc["stages"], sc["initial_base"], sc["target_base"])
        got = simulate_rope_schedule(sc["stages"], sc["initial_base"], sc["target_base"])
        if isinstance(got, list) and len(got) == len(want) and all(abs(g - w) < 1e-3 for g, w in zip(got, want)):
            ok += 1

    for i, sc in enumerate(ref.EVAL_SCENARIOS):
        want = ref.compare_strategies(sc["abf_ppl"], sc["yarn_ppl"])
        got = compare_strategies(sc["abf_ppl"], sc["yarn_ppl"])
        if got == want:
            ok += 1

    out["schedules_matched"] = float(ok)
    return out
