import ref


def check(workdir):
    from eagle.integration import EagleIntegration
    from eagle.head import DraftHead
    m = {"memory_saved": 0.0}
    try:
        head = DraftHead(4, 10)
        integ = EagleIntegration(None, head)
        mem = integ.estimate_memory_mb()
        if isinstance(mem, (int, float)) and mem < 500.0:
            m["memory_saved"] = 1.0
    except Exception:
        pass
    return m
