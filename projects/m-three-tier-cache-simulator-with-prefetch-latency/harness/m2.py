import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from cachesim.simulator import simulate

    out = {"advanced_latency_rel_err": 0.0, "advanced_penalty_rel_err": 0.0}
    lat_errs = []
    pen_errs = []

    for trace in ref.TRACES:
        for policy in ["always", "reuse_2", "size_aware"]:
            for mode in ["wb", "wt"]:
                want = ref.simulate(trace, 1024, 2048, policy, mode)
                got = simulate(trace, 1024, 2048, policy, mode)

                l_err = abs(got["latency_ns"] - want["latency_ns"]) / max(want["latency_ns"], 1)
                lat_errs.append(l_err)

                p_err = abs(got["write_penalty_ns"] - want["write_penalty_ns"]) / max(want["write_penalty_ns"], 1)
                pen_errs.append(p_err)

    out["advanced_latency_rel_err"] = max(lat_errs) if lat_errs else 1.0
    out["advanced_penalty_rel_err"] = max(pen_errs) if pen_errs else 1.0

    return out
