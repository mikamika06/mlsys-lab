import ref
import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"lat_err": 0.0, "warmup_err": 0.0}
    try:
        import serving.models as m

        got_lat = m.profile_latency(8, 256)
        want_lat = ref.ref_profile_latency(8, 256)
        if abs(got_lat - want_lat) < 1e-5:
            out["lat_err"] = 1.0

        b_sizes = [1, 2, 4, 8, 16]
        s_lens = [32, 64, 128, 256, 512]
        got_warmup = m.warmup_cost(b_sizes, s_lens)
        want_warmup = ref.ref_warmup_cost(b_sizes, s_lens)
        if abs(got_warmup - want_warmup) < 1e-5:
            out["warmup_err"] = 1.0

    except Exception as e:
        out["_note"] = f"M1 failed: {e}"

    return out
