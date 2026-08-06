import ref


def check(workdir):
    from layout.chooser import check_memory_fit
    from layout.imbalance import straggler_factor

    out = {"memory_checks_matched": 0.0, "straggler_factors_matched": 0.0}

    mem_ok = True
    for cfg in ref.CONFIGS:
        for vram in ref.VRAM_TESTS:
            for tp, pp, _ in ref.CANDIDATE_LAYOUTS:
                want = ref.check_memory_fit(cfg, tp, pp, vram)
                got = check_memory_fit(cfg, tp, pp, vram)
                if want != got:
                    mem_ok = False
                    out["_note"] = f"check_memory_fit mismatch for tp={tp}, pp={pp}, vram={vram}"
                    break
            if not mem_ok:
                break
        if not mem_ok:
            break

    if mem_ok:
        out["memory_checks_matched"] = 1.0

    strag_ok = True
    for hist in ref.HISTOGRAMS:
        want = ref.straggler_factor(hist)
        got = straggler_factor(hist)
        if abs(want - got) > 1e-5:
            strag_ok = False
            out["_note"] = f"straggler_factor mismatch: want {want}, got {got}"
            break

    if strag_ok:
        out["straggler_factors_matched"] = 1.0

    return out
