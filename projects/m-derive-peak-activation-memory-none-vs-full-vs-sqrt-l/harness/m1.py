import ref


def check(workdir):
    from ckpt import analysis

    out = {"memory_cases_matched": 0.0}
    ok = 0
    for layers, base_mem, strategy, seg in ref.TEST_CASES_MEM:
        want = ref.peak_activation_memory(layers, base_mem, strategy, seg)
        try:
            got = analysis.peak_activation_memory(layers, base_mem, strategy, seg)
        except Exception as e:
            out["_note"] = f"raised {type(e).__name__} for ({layers}, {base_mem}, {strategy}, {seg})"
            return out
        if got is not None and abs(got - want) < 1e-5:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"mismatch for ({layers}, {base_mem}, {strategy}): got {got}, want {want}"

    opt_want = ref.optimal_segment_size(25, 10.0)
    try:
        opt_got = analysis.optimal_segment_size(25, 10.0)
        if opt_got == opt_want:
            ok += 1
    except Exception:
        pass

    out["memory_cases_matched"] = float(ok)
    return out
