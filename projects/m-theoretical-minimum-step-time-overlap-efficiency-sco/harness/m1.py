import ref


def check(workdir):
    from overlap.reconstruct import reconstruct_timeline

    out = {"reconstruction_rel_err": 1.0}
    max_err = 0.0

    for events in ref.TRACES:
        want = ref.reconstruct_timeline(events)
        got = reconstruct_timeline(events)

        for key in ["total_span", "compute_only", "comm_only", "overlapped", "idle"]:
            w_val = want[key]
            g_val = got.get(key, 0.0)
            denom = abs(w_val) if abs(w_val) > 1e-9 else 1.0
            err = abs(g_val - w_val) / denom
            if err > max_err:
                max_err = err

    out["reconstruction_rel_err"] = float(max_err)
    return out
