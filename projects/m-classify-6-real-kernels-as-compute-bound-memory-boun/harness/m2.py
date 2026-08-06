import ref


def check(workdir):
    from kernelperf.bandwidth import compute_achieved_gbps, cross_check_bandwidth
    out = {"bandwidth_match": 0.0}
    ok = True
    for k in ref.KERNELS:
        got_gbps = compute_achieved_gbps(k["dram_pct"], ref.PEAK_DRAM_BW_GBPS)
        want_gbps = ref.reference_achieved_gbps(k["dram_pct"], ref.PEAK_DRAM_BW_GBPS)
        if abs(got_gbps - want_gbps) > 1e-3:
            ok = False
        cross = cross_check_bandwidth(k["bytes"], k["time_ns"], ref.PEAK_DRAM_BW_GBPS)
        want_cross = ref.reference_cross_check(k["bytes"], k["time_ns"])
        if abs(cross - want_cross) > 1e-3:
            ok = False
    if ok:
        out["bandwidth_match"] = 1.0
    return out
