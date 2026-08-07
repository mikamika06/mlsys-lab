import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    from preflight.profiler import compare_throughput_to_fp16, estimate_throughput

    out = {"throughput_matched": 0.0}
    ok = 0
    total = len(ref.CONFIGS)

    for i, cfg in enumerate(ref.CONFIGS):
        m_cfg = cfg["model"]
        q_cfg = cfg["quant"]
        bs = cfg["batch_size"]
        sl = cfg["seq_len"]

        want_tps = ref.estimate_throughput(m_cfg, q_cfg, bs, sl)
        got_tps = estimate_throughput(m_cfg, q_cfg, bs, sl)

        want_cmp = ref.compare_throughput_to_fp16(m_cfg, q_cfg, bs, sl)
        got_cmp = compare_throughput_to_fp16(m_cfg, q_cfg, bs, sl)

        tps_ok = abs(want_tps - got_tps) < 1e-3
        cmp_ok = (
            isinstance(got_cmp, dict)
            and got_cmp.get("is_faster") == want_cmp["is_faster"]
            and abs(got_cmp.get("speedup_ratio", 0) - want_cmp["speedup_ratio"]) < 1e-3
        )

        if tps_ok and cmp_ok:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got cmp={got_cmp}, want cmp={want_cmp}"

    if ok == total:
        out["throughput_matched"] = 1.0
    return out
