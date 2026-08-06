import ref


def check(workdir):
    from ctxcost.estimator import estimate_cost

    out = {"costs_matched": 0.0, "configs": float(len(ref.SCENARIOS))}
    ok = 0
    for i, sc in enumerate(ref.SCENARIOS):
        want = ref.estimate_cost(sc["base_tokens"], sc["target_length"], sc["base_length"], sc["alpha"])
        got = estimate_cost(sc["base_tokens"], sc["target_length"], sc["base_length"], sc["alpha"])
        if abs(got - want) < 1e-4:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"scenario {i}: got {got}, want {want}"
    out["costs_matched"] = float(ok)
    return out
