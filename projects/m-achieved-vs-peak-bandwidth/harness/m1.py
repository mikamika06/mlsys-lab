import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from bandwidth.tracker import (
            compute_achieved_bandwidth,
            compute_bandwidth_efficiency,
            compute_bytes_transferred,
        )
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Failed to import tracker functions: {e}"}

    max_err = 0.0

    for cfg in ref.CONFIGS:
        want_bytes = ref.compute_bytes_transferred(cfg)
        try:
            got_bytes = compute_bytes_transferred(cfg)
        except Exception as e:
            return {"rel_err": 1.0, "_note": f"compute_bytes_transferred raised {e}"}

        for k in ("naive_bytes", "tiled_bytes"):
            w = want_bytes[k]
            g = got_bytes.get(k, 0.0)
            err = abs(g - w) / max(abs(w), 1e-9)
            max_err = max(max_err, err)

        exec_time = 0.0025
        peak_gbps = 1500.0

        want_bw = ref.compute_achieved_bandwidth(want_bytes["tiled_bytes"], exec_time)
        got_bw = compute_achieved_bandwidth(got_bytes["tiled_bytes"], exec_time)
        err = abs(got_bw - want_bw) / max(abs(want_bw), 1e-9)
        max_err = max(max_err, err)

        want_eff = ref.compute_bandwidth_efficiency(want_bw, peak_gbps)
        got_eff = compute_bandwidth_efficiency(got_bw, peak_gbps)
        err = abs(got_eff - want_eff) / max(abs(want_eff), 1e-9)
        max_err = max(max_err, err)

    return {"rel_err": float(max_err)}
