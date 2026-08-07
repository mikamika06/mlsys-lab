import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from checkpointing.sim import optimal_segments

    out = {"optimal_match": 0.0}
    ok = 0
    for n in ref.N_LAYERS_LIST:
        want = ref.optimal_segments(n)
        try:
            got = optimal_segments(n)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"optimal_segments({n}): got {got}, want {want}"
        except Exception:
            pass

    out["optimal_match"] = 1.0 if ok == len(ref.N_LAYERS_LIST) else 0.0
    return out
