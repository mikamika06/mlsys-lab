import ref

def check(workdir):
    from bnb_ledger.ledger import get_ledger, predict_memory_footprint
    out = {"configs_matched": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        want_ledger = ref.get_ledger(cfg)
        got_ledger = get_ledger(cfg)
        want_mem = ref.predict_memory_footprint(cfg, 1000000)
        got_mem = predict_memory_footprint(cfg, 1000000)
        if abs(want_ledger["bits_per_param"] - got_ledger["bits_per_param"]) < 1e-6 and want_mem == got_mem:
            ok += 1
    out["configs_matched"] = float(ok)
    return out
