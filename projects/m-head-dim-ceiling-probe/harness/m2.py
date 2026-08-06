import ref


def check(workdir):
    from probe import throughput, fp8_gating

    out = {"throughput_match": 0.0, "fp8_reason_match": 0.0}
    t_ok = 0
    f_ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want_t = ref.estimate_throughput(cfg)
        got_t = throughput.estimate_throughput(cfg)
        if got_t == want_t:
            t_ok += 1

        want_f = ref.check_fp8_availability(cfg)
        got_f = fp8_gating.check_fp8_availability(cfg)
        if got_f == want_f:
            f_ok += 1

    out["throughput_match"] = 1.0 if t_ok == len(ref.CONFIGS) else 0.0
    out["fp8_reason_match"] = 1.0 if f_ok == len(ref.CONFIGS) else 0.0
    return out
