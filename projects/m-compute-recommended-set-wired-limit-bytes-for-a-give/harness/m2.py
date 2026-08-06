import ref
import sys

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from limits.compute import tune_limits
        from limits.simulate import simulate_rss
    except ImportError:
        return {"tune_matched": 0.0, "sim_matched": 0.0, "_note": "failed to import limits"}

    gb = 1024 ** 3
    mb = 1024 ** 2
    memsize = 32 * gb
    model_size = 14 * gb

    out = {"tune_matched": 0.0, "sim_matched": 0.0}

    w_want, c_want = ref.tune_limits(memsize, model_size)
    try:
        w_got, c_got = tune_limits(memsize, model_size)
    except NotImplementedError:
        out["_note"] = "tune_limits not implemented"
        return out

    if w_want == w_got and c_want == c_got:
        out["tune_matched"] = 1.0
    else:
        out["_note"] = f"tune_limits mismatch: got {(w_got, c_got)}, expected {(w_want, c_want)}"
        return out

    sim_want = ref.simulate_rss(50, c_want, model_size, 16 * mb)
    try:
        sim_got = simulate_rss(50, c_want, model_size, 16 * mb)
    except NotImplementedError:
        out["_note"] = "simulate_rss not implemented"
        return out

    if sim_want == sim_got:
        out["sim_matched"] = 1.0
    else:
        out["_note"] = "simulate_rss output diverged from reference"

    return out
