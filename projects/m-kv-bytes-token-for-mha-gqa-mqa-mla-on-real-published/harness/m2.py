import ref


def check(workdir):
    from kvbytes.measure import measure_growth
    from kvbytes.mla import mla_bytes_per_token
    out = {"growth_match": 0.0, "mla_match": 0.0}

    cfg_std = ref.CONFIGS[1]
    want_growth = ref.measure_growth(cfg_std, 10)
    try:
        got_growth = measure_growth(cfg_std, 10)
    except Exception:
        got_growth = []

    cfg_mla = ref.CONFIGS[3]
    want_mla = ref.mla_bytes_per_token(cfg_mla)
    try:
        got_mla = mla_bytes_per_token(cfg_mla)
    except Exception:
        got_mla = -1

    out["growth_match"] = 1.0 if got_growth == want_growth else 0.0
    out["mla_match"] = 1.0 if got_mla == want_mla else 0.0
    return out
