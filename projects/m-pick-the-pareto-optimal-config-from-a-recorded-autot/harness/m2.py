import ref


def check(workdir):
    from autotune.sweep import load_sweep
    from autotune.pareto import compute_pareto
    from autotune.select import select_best

    raw = ref.raw_sweep_text()
    configs = load_sweep(raw)

    want_pareto = ref.compute_pareto(configs)
    got_pareto = compute_pareto(configs)

    want_best = ref.select_best(configs, 65536)
    got_best = select_best(configs, 65536)

    out = {"pareto_match": 0.0, "argmin_index": 0.0}

    norm_want = [c["id"] for c in want_pareto]
    norm_got = [c["id"] for c in got_pareto]

    if norm_got == norm_want:
        out["pareto_match"] = 1.0
    else:
        out["_note"] = f"pareto mismatch: got {norm_got}, want {norm_want}"

    if got_best == want_best:
        out["argmin_index"] = 1.0
    else:
        out["_note"] = f"argmin mismatch: got {got_best}, want {want_best}"

    return out
