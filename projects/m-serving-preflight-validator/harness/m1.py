import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    from preflight.validator import estimate_memory_bytes, validate_fit

    out = {"fits_matched": 0.0}
    ok = 0
    total = len(ref.CONFIGS)

    for i, cfg in enumerate(ref.CONFIGS):
        m_cfg = cfg["model"]
        q_cfg = cfg["quant"]
        g_cfg = cfg["gpu"]

        want_bytes = ref.estimate_memory_bytes(m_cfg, q_cfg, g_cfg["num_gpus"])
        got_bytes = estimate_memory_bytes(m_cfg, q_cfg, g_cfg["num_gpus"])

        want_fit = ref.validate_fit(m_cfg, q_cfg, g_cfg)
        got_fit = validate_fit(m_cfg, q_cfg, g_cfg)

        bytes_ok = abs(want_bytes - got_bytes) < 1.0
        fit_ok = (
            isinstance(got_fit, dict)
            and got_fit.get("fits") == want_fit["fits"]
            and abs(got_fit.get("estimated_bytes_per_gpu", 0) - want_fit["estimated_bytes_per_gpu"]) < 1.0
        )

        if bytes_ok and fit_ok:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got fit={got_fit}, want fit={want_fit}"

    if ok == total:
        out["fits_matched"] = 1.0
    return out
