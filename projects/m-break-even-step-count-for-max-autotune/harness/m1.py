import ref


def check(workdir):
    from autotune.delta import compute_delta

    cases = ref.generate_fixtures()
    ok = 0
    for base, mode in cases:
        want = compute_delta_ref(base, mode)
        got = compute_delta(base, mode)
        if got == want:
            ok += 1
    return {"delta_match": 1.0 if ok == len(cases) else 0.0}


def compute_delta_ref(base_config, target_mode):
    out = dict(base_config)
    if target_mode == "default":
        pass
    elif target_mode == "reduce-overhead":
        out["triton.cudagraphs"] = True
    elif target_mode == "max-autotune":
        out["max_autotune"] = True
        out["triton.cudagraphs"] = True
    return {k: out[k] for k in sorted(out.keys()) if out[k] != base_config.get(k)}
