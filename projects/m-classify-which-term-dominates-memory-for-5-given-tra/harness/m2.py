import ref

def check(workdir):
    from actmem.profile import measure_forward_activation_memory
    cfg = ref.CONFIGS[0]
    want = ref.measure_peak_memory(cfg)
    got = measure_forward_activation_memory(cfg)
    if got <= 0:
        return {"rel_err": 1.0, "_note": "non-positive memory measured"}
    err = abs(got - want) / want
    return {"rel_err": float(err)}
