import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"rel_err": 1.0}
    try:
        from macroofline.bandwidth import (
            achieved_bandwidth_gbps,
            bandwidth_utilization_pct,
            bytes_transferred,
        )

        errs = []
        for run in ref.SAMPLE_RUNS:
            m, n, k = run["m"], run["n"], run["k"]
            sec = run["elapsed_sec"]
            itemsize = run.get("itemsize", 2)

            ref_b = ref.bytes_transferred(m, n, k, itemsize)
            got_b = bytes_transferred(m, n, k, itemsize)
            errs.append(abs(got_b - ref_b) / (abs(ref_b) + 1e-12))

            ref_bw = ref.achieved_bandwidth_gbps(ref_b, sec)
            got_bw = achieved_bandwidth_gbps(got_b, sec)
            errs.append(abs(got_bw - ref_bw) / (abs(ref_bw) + 1e-12))

            ref_pct = ref.bandwidth_utilization_pct(ref_bw, ref.M4_MAX_PEAK_GBPS)
            got_pct = bandwidth_utilization_pct(got_bw, ref.M4_MAX_PEAK_GBPS)
            errs.append(abs(got_pct - ref_pct) / (abs(ref_pct) + 1e-12))

        out["rel_err"] = max(errs) if errs else 1.0
    except Exception as e:
        out["_note"] = f"{type(e).__name__}: {e}"
    return out
