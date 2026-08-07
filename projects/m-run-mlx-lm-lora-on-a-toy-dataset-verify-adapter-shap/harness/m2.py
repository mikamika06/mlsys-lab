import ref


def check(workdir):
    from loratools.memory import measure_peak_rss

    out = {"rss_measured": 0.0}
    try:
        base = 100.0
        full_val = measure_peak_rss(base, False)
        qlora_val = measure_peak_rss(base, True)
        want_full = ref.expected_peak_rss(base, False)
        want_qlora = ref.expected_peak_rss(base, True)
        if abs(full_val - want_full) < 1e-5 and abs(qlora_val - want_qlora) < 1e-5:
            out["rss_measured"] = 1.0
        else:
            out["_note"] = f"got full={full_val}, qlora={qlora_val}; want full={want_full}, qlora={want_qlora}"
    except Exception as e:
        out["_note"] = f"error in m2: {type(e).__name__}: {str(e)[:120]}"
    return out
