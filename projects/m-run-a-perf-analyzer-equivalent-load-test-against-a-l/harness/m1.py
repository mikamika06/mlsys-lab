import ref
import numpy as np

def check(workdir):
    from loadtest.simulator import generate_schedule
    out = {"schedule_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.generate_schedule(cfg["concurrency"], cfg["num_requests"], cfg["seed"])
        got = generate_schedule(cfg["concurrency"], cfg["num_requests"], cfg["seed"])
        got = np.asarray(got, dtype=float)
        if np.allclose(got, want, rtol=1e-5, atol=1e-5):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: schedules differ"
    out["schedule_matched"] = float(ok)
    return out
