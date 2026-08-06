import ref

def check(workdir):
    from preemption.crossover import find_crossover
    out = {"crossover_matched": 0.0}
    ok = 0
    total = len(ref.MODELS) * len(ref.SYSTEMS)
    for m in ref.MODELS:
        for s in ref.SYSTEMS:
            want = ref.find_crossover(m, s)
            got = find_crossover(m, s)
            if got == want:
                ok += 1
            else:
                out["_note"] = f"mismatch for model {m}, system {s}: got {got}, want {want}"
                return out
    if ok == total:
        out["crossover_matched"] = 1.0
    return out
