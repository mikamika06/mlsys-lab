import ref


def check(workdir):
    from eagle.integration import EagleIntegration
    from eagle.head import DraftHead
    m = {"speedup_ok": 0.0}
    try:
        head = DraftHead(4, 10)
        integ = EagleIntegration(None, head)
        sp = integ.compute_speedup(10.0, 5.0)
        if sp >= 1.5:
            m["speedup_ok"] = 1.0
    except Exception:
        pass
    return m
