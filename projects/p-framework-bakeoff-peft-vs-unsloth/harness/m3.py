def check(workdir):
    from bakeoff.engine import BakeoffEngine
    cfg = {"seed": 42}
    eng1 = BakeoffEngine(cfg)
    eng2 = BakeoffEngine(cfg)
    eng1.step("baseline")
    eng2.step("baseline")
    w1 = eng1.get_weights("baseline")
    w2 = eng2.get_weights("baseline")
    import numpy as np
    if not np.allclose(w1, w2):
        return {"weights_match": 0.0}
    return {"weights_match": 1.0}
