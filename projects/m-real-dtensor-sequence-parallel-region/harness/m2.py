import ref


def check(workdir):
    from sp.memory import measure_memory

    out = {"memory_measured": 0.0, "ratio_valid": 0.0}
    cfg = ref.CONFIGS[0]
    try:
        m_tp = measure_memory(cfg, "tp_only")
        m_sp = measure_memory(cfg, "tp_sp")
    except Exception as e:
        out["_note"] = f"measure_memory failed: {e}"
        return out

    ref_tp = ref.measure_memory(cfg, "tp_only")
    ref_sp = ref.measure_memory(cfg, "tp_sp")

    if m_tp == ref_tp and m_sp == ref_sp:
        out["memory_measured"] = 1.0
    else:
        out["_note"] = f"got tp={m_tp}, sp={m_sp}, want tp={ref_tp}, sp={ref_sp}"
        return out

    if m_sp < m_tp and (m_tp / m_sp) >= cfg["tp_size"] * 0.9:
        out["ratio_valid"] = 1.0
    else:
        out["_note"] = f"ratio invalid: tp={m_tp}, sp={m_sp}"
    return out
