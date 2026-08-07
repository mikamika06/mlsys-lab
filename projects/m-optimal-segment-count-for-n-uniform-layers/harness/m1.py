import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from checkpointing.sim import baseline, simulate_checkpointing

    out = {"baseline_match": 0.0, "sim_match": 0.0}
    ok_b = 0
    ok_s = 0

    for cfg in ref.CONFIGS:
        n, s, mem, f, b = cfg
        wb = ref.baseline(n, mem, f, b)
        try:
            gb = baseline(n, mem, f, b)
            if gb == wb:
                ok_b += 1
        except Exception:
            pass

        ws = ref.simulate_checkpointing(n, s, mem, f, b)
        try:
            gs = simulate_checkpointing(n, s, mem, f, b)
            if gs == ws:
                ok_s += 1
            elif "_note" not in out:
                out["_note"] = f"sim({n}, {s}) got {gs}, want {ws}"
        except Exception:
            pass

    out["baseline_match"] = 1.0 if ok_b == len(ref.CONFIGS) else 0.0
    out["sim_match"] = 1.0 if ok_s == len(ref.CONFIGS) else 0.0
    return out
