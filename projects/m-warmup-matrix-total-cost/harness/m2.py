import ref
import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"sim_p50_err": 0.0, "sim_p99_err": 0.0, "tune_match": 0.0}
    arr, seq = ref.generate_arrivals(200, rate=0.1, seed=123)

    try:
        import serving.queue as q

        got_sim = q.simulate(arr, seq, 4, 15.0)
        want_sim = ref.ref_simulate(arr, seq, 4, 15.0)

        if abs(got_sim["p50"] - want_sim["p50"]) < 1e-3:
            out["sim_p50_err"] = 1.0
        if abs(got_sim["p99"] - want_sim["p99"]) < 1e-3:
            out["sim_p99_err"] = 1.0

        got_tune = q.tune_batching(arr, seq, 200.0)
        want_tune = ref.ref_tune_batching(arr, seq, 200.0)
        if got_tune == want_tune:
            out["tune_match"] = 1.0

    except Exception as e:
        out["_note"] = f"M2 failed: {e}"

    return out
