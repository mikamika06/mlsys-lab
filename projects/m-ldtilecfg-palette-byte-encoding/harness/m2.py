import ref


def check(workdir):
    from amx.model import amx_vs_avx512_throughput, tmul_time_share

    out = {"speedup_rel_err": 1.0, "share_rel_err": 1.0}

    speedup_errs = []
    for params in ref.THROUGHPUT_FIXTURES:
        want = ref.amx_vs_avx512_throughput(params)
        got = amx_vs_avx512_throughput(params)

        for k in ["amx_peak_tflops", "avx512_peak_tflops", "derived_ceiling", "measured_speedup"]:
            w_val = want[k]
            g_val = got.get(k, 0.0)
            err = abs(g_val - w_val) / max(1e-9, abs(w_val))
            speedup_errs.append(err)

    share_errs = []
    for params in ref.TIME_SHARE_FIXTURES:
        want = ref.tmul_time_share(params)
        got = tmul_time_share(params)

        for k in ["compute_cycles", "mem_cycles", "total_cycles", "tmul_compute_share", "tile_io_share"]:
            w_val = want[k]
            g_val = got.get(k, 0.0)
            err = abs(g_val - w_val) / max(1e-9, abs(w_val))
            share_errs.append(err)

    out["speedup_rel_err"] = float(max(speedup_errs)) if speedup_errs else 1.0
    out["share_rel_err"] = float(max(share_errs)) if share_errs else 1.0

    return out
