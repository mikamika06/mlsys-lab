import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from attnsink.drift import compute_drift

    Q, K, V = ref.make_inputs(seq_len=512, d_k=32, d_v=32, seed=999)
    sink_size = 4
    window_size = 32

    want_res = ref.compute_drift(Q, K, V, sink_size, window_size)
    try:
        got_res = compute_drift(Q, K, V, sink_size, window_size)
    except Exception as e:
        return {"rel_err": 1.0, "sink_beats_win": 0.0, "_note": f"Exception raised: {type(e).__name__}: {e}"}

    errs = []
    for key in ["win_rel_err", "sink_rel_err"]:
        errs.append(abs(got_res[key] - want_res[key]))
    for key in ["drift_by_pos", "win_drift_by_pos", "full_out", "win_out", "sink_out"]:
        denom = np.linalg.norm(want_res[key]) + 1e-12
        errs.append(float(np.linalg.norm(got_res[key] - want_res[key]) / denom))

    max_err = max(errs)
    sink_beats_win = 1.0 if got_res["sink_rel_err"] < got_res["win_rel_err"] else 0.0

    return {"rel_err": max_err, "sink_beats_win": sink_beats_win}
