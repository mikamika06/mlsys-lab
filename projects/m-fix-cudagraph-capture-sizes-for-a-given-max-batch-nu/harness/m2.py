import ref


def check(workdir):
    from speculative.eagle import find_optimal_eagle_config

    out = {"optimal_config_matched": 0.0}
    budget = 3000 * 1024
    want = ref.find_optimal_eagle_config(ref.EAGLE_CONFIGS, budget)
    got = find_optimal_eagle_config(ref.EAGLE_CONFIGS, budget)
    if got == want:
        out["optimal_config_matched"] = 1.0
    else:
        out["_note"] = f"got {got}, reference {want}"
    return out
