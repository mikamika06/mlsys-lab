import numpy as np
import ref


def check(workdir):
    from threadsweep.profile import analyze_profile_gap

    np.random.seed(123)
    op_times = [5.2, 3.1, 12.4, 2.0]
    wall_clock = 28.5

    want = ref.compute_profile_gap(op_times, wall_clock)
    out = {"gap_match": 0.0}
    try:
        got = analyze_profile_gap(op_times, wall_clock)
    except Exception as e:
        out["_note"] = f"execution failed: {type(e).__name__}: {str(e)[:100]}"
        return out

    if isinstance(got, dict) and "gap" in got and np.isclose(got["gap"], want["gap"], atol=1e-3):
        out["gap_match"] = 1.0
    else:
        out["_note"] = f"got {got}, want gap approx {want['gap']}"
    return out
