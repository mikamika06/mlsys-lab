import ref

def check(workdir):
    from moe_sim.metrics import active_fraction, vram_cost

    out = {"fraction_match": 0.0, "vram_match": 0.0}

    try:
        if abs(active_fraction(ref.CFG) - ref.active_fraction(ref.CFG)) < 1e-6:
            out["fraction_match"] = 1.0
    except NotImplementedError:
        pass

    try:
        match = True
        for ngl in [0, 10, 32]:
            for c in [0, 2, 8]:
                if vram_cost(ref.CFG, ngl, c) != ref.vram_cost(ref.CFG, ngl, c):
                    match = False
        if match:
            out["vram_match"] = 1.0
    except NotImplementedError:
        pass

    return out
